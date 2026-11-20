"""Accent classifier: waveform -> frame features -> masked pooling -> logits.

Backbones
  ``mel_cnn``  log-mel + small 1-D CNN trained from scratch (fast on CPU)
  ``hf``       WavLM / wav2vec2 / HuBERT; chosen hidden layers fused by a learned
               softmax-weighted sum, unused upper layers dropped

Padding never reaches the accent embedding. Every backbone returns a frame
mask derived from the true sample lengths, every pooling op is masked, and
for encoders whose CNN uses GroupNorm over time (WavLM Base/Base+,
wav2vec2-base), where an attention mask is not enough, each distinct length
is encoded without padding.

    out = model(waveforms, attention_mask)        # waveforms [B, T], attention_mask [B, T]
    out["logits"]      # [B, num_accents]  (raw logits: softmax only at inference)
    out["embedding"]   # [B, 2 * D]        pooled mean+std accent embedding
    out["frame_mask"]  # [B, F]            True = real frame
    out["attention"]   # [B, F]            pooling weights (None for mean pooling)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from .config import Config, ModelConfig
from .dataset import SpecAugment


# ------------------------------------------------------------------------- helpers
def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """``[B]`` lengths -> ``[B, max_len]`` boolean mask (True = valid)."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean_std(
    x: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and std over valid frames of ``x [B, T, D]`` (optionally attention-weighted).

    Padded frames get exactly zero weight, so batching never changes the result."""
    x = x.float()
    if weights is None:
        w = mask.float()
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1.0)
    else:
        w = weights.float() * mask.float()
    w = w.unsqueeze(-1)
    mean = (w * x).sum(dim=1)
    var = (w * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
    return mean, torch.sqrt(var.clamp(min=eps))


# ---------------------------------------------------------------------- mel frontend
def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 20.0,
    f_max: float | None = None,
) -> torch.Tensor:
    """HTK-style triangular mel filters, ``[n_fft // 2 + 1, n_mels]``."""
    f_max = f_max or sample_rate / 2

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    hz = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2))
    freqs = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)[:, None]
    left, center, right = hz[:-2][None], hz[1:-1][None], hz[2:][None]
    up = (freqs - left) / np.maximum(center - left, 1e-9)
    down = (right - freqs) / np.maximum(right - center, 1e-9)
    return torch.tensor(np.clip(np.minimum(up, down), 0.0, None), dtype=torch.float32)


class LogMel(nn.Module):
    """Log-mel spectrogram with ``center=False``: a frame only ever covers real
    samples, so the valid frames of a clip do not depend on padding."""

    def __init__(self, sample_rate: int, n_mels: int = 80, win_ms: float = 25, hop_ms: float = 10):
        super().__init__()
        self.n_fft = int(round(sample_rate * win_ms / 1000))
        self.hop = int(round(sample_rate * hop_ms / 1000))
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.register_buffer(
            "fb", mel_filterbank(sample_rate, self.n_fft, n_mels), persistent=False
        )

    def num_frames(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths - self.n_fft, self.hop, rounding_mode="floor").clamp(min=0) + 1

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.shape[-1] < self.n_fft:  # shorter than one window: pad (the frame is masked)
            wave = nn.functional.pad(wave, (0, self.n_fft - wave.shape[-1]))
        spec = torch.stft(
            wave.float(),
            self.n_fft,
            self.hop,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spec.abs().pow(2).transpose(1, 2)  # [B, frames, freq]
        return torch.log(power @ self.fb + 1e-6)  # [B, frames, n_mels]


class ConvBlock(nn.Module):
    """Conv1d -> LayerNorm (per frame) -> GELU -> dropout, zeroing invalid frames."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.padding = dilation * (kernel - 1) // 2
        self.kernel, self.stride, self.dilation = kernel, stride, dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, stride, self.padding, dilation)
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)
        self.residual = c_in == c_out and stride == 1

    def out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        span = self.dilation * (self.kernel - 1) + 1
        return torch.div(lengths + 2 * self.padding - span, self.stride, rounding_mode="floor") + 1

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.conv(x).transpose(1, 2)  # [B, T, C]
        y = self.drop(nn.functional.gelu(self.norm(y))).transpose(1, 2)
        if self.residual:
            y = y + x
        lengths = self.out_lengths(lengths).clamp(min=1)
        mask = lengths_to_mask(lengths, y.shape[-1])
        return y * mask[:, None, :].to(y.dtype), lengths


class MelCNNBackbone(nn.Module):
    """Log-mel -> masked per-utterance mean normalisation -> 4 conv blocks (20 ms frames)."""

    def __init__(self, sample_rate: int, n_mels: int, hidden: int, dropout: float):
        super().__init__()
        self.frontend = LogMel(sample_rate, n_mels)
        self.blocks = nn.ModuleList(
            [
                ConvBlock(n_mels, hidden, 5, dropout=dropout),
                ConvBlock(hidden, hidden, 3, stride=2, dropout=dropout),
                ConvBlock(hidden, hidden, 3, dropout=dropout),
                ConvBlock(hidden, hidden, 3, dilation=2, dropout=dropout),
            ]
        )
        self.out_dim = hidden

    def frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        lengths = self.frontend.num_frames(sample_lengths).clamp(min=1)
        for block in self.blocks:
            lengths = block.out_lengths(lengths).clamp(min=1)
        return lengths

    def forward(
        self,
        wave: torch.Tensor,
        lengths: torch.Tensor,
        spec_augment: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.frontend(wave)  # [B, T, M]
        frames = self.frontend.num_frames(lengths).clamp(min=1, max=x.shape[1])
        mask = lengths_to_mask(frames, x.shape[1])
        mean, _ = masked_mean_std(x, mask)
        x = (x - mean.unsqueeze(1)) * mask.unsqueeze(-1)  # CMN over real frames only
        if spec_augment is not None:
            x = spec_augment(x, frames)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x, frames = block(x, frames)
        return x.transpose(1, 2), lengths_to_mask(frames, x.shape[-1])


# --------------------------------------------------------------------- HF backbones
HF_CLASSES = {
    "wavlm": ("WavLMConfig", "WavLMModel"),
    "wav2vec2": ("Wav2Vec2Config", "Wav2Vec2Model"),
    "hubert": ("HubertConfig", "HubertModel"),
}


class HFBackbone(nn.Module):
    """Self-supervised speech encoder with a learned weighted sum of chosen layers.

    Layers above ``max(layers)`` are removed (they cannot influence the output,
    would waste compute, and would hold parameters that never get a gradient).
    HF SpecAugment and LayerDrop are disabled by default; SpecAugment is applied
    to the fused features instead, within each clip's valid frames."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        import transformers

        config_name, model_name = HF_CLASSES[cfg.hf_arch]
        overrides = {"layerdrop": 0.0, "mask_time_prob": 0.0, "mask_feature_prob": 0.0}
        overrides.update(cfg.hf_overrides)
        if cfg.pretrained:
            model = transformers.AutoModel.from_pretrained(cfg.hf_name, **overrides)
        else:
            config = getattr(transformers, config_name)(**overrides)
            model = getattr(transformers, model_name)(config)
        config = model.config
        self.layers = sorted(set(cfg.layers))
        top = max(self.layers)
        if top > config.num_hidden_layers:
            raise ValueError(
                f"model.layers={cfg.layers} but {cfg.hf_name} has {config.num_hidden_layers} layers"
            )
        if top < config.num_hidden_layers:
            model.encoder.layers = model.encoder.layers[:top]
            config.num_hidden_layers = top
            if config.do_stable_layer_norm:  # the final LN belongs after the *last* original layer
                config.accent_final_norm_removed = True
        if getattr(config, "accent_final_norm_removed", False):
            model.encoder.layer_norm = nn.Identity()
        self.model = model
        self.group_norm = config.feat_extract_norm == "group"
        self.layer_weights = nn.Parameter(torch.zeros(len(self.layers)))
        self.out_dim = config.hidden_size
        if cfg.freeze_feature_encoder or cfg.freeze_backbone:
            for p in model.feature_extractor.parameters():
                p.requires_grad_(False)
        if cfg.freeze_backbone:
            for p in model.parameters():
                p.requires_grad_(False)

    def frame_lengths(self, sample_lengths: torch.Tensor) -> torch.Tensor:
        lengths = self.model._get_feat_extract_output_lengths(sample_lengths)
        return lengths.long().clamp(min=1)

    def _hidden_states(self, wave: torch.Tensor, lengths: torch.Tensor) -> list[torch.Tensor]:
        """Selected hidden states, each ``[B, F_max, D]``."""
        if not self.group_norm:
            mask = lengths_to_mask(lengths, wave.shape[1]).long()
            out = self.model(wave, attention_mask=mask, output_hidden_states=True)
            return [out.hidden_states[i] for i in self.layers]
        # GroupNorm normalises over the whole (padded) time axis: encode each length unpadded.
        f_max = int(self.frame_lengths(lengths.max().reshape(1))[0])
        outs: list[torch.Tensor] | None = None
        for length in torch.unique(lengths).tolist():
            idx = (lengths == length).nonzero(as_tuple=True)[0]
            hs = self.model(wave[idx, :length], output_hidden_states=True).hidden_states
            if outs is None:
                outs = [hs[0].new_zeros(wave.shape[0], f_max, hs[0].shape[-1]) for _ in self.layers]
            for k, layer in enumerate(self.layers):
                outs[k][idx, : hs[layer].shape[1]] = hs[layer]
        return outs  # type: ignore[return-value]

    def forward(
        self,
        wave: torch.Tensor,
        lengths: torch.Tensor,
        spec_augment: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = self._hidden_states(wave, lengths)
        weights = torch.softmax(self.layer_weights, dim=0)
        feats = sum(w * h for w, h in zip(weights, states, strict=True))
        frames = self.frame_lengths(lengths).clamp(max=feats.shape[1])
        mask = lengths_to_mask(frames, feats.shape[1])
        feats = feats * mask.unsqueeze(-1).to(feats.dtype)
        if spec_augment is not None:
            feats = spec_augment(feats, frames)
        return feats, mask


# ------------------------------------------------------------------------- pooling
class AttentiveStatsPooling(nn.Module):
    """Masked attentive statistics pooling (Okabe et al., 2018) with ECAPA-style
    global context: attention sees each frame plus the utterance mean and std."""

    def __init__(self, dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(3 * dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )
        self.out_dim = 2 * dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        mean, std = masked_mean_std(x, mask)
        context = torch.cat(
            [x, mean.unsqueeze(1).expand_as(x), std.unsqueeze(1).expand_as(x)], dim=-1
        )
        scores = self.attention(context).squeeze(-1).float()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1) * mask  # padded frames: exactly 0
        mean, std = masked_mean_std(x, mask, weights)
        return torch.cat([mean, std], dim=-1), weights


class MeanStdPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = 2 * dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        mean, std = masked_mean_std(x, mask)
        return torch.cat([mean, std], dim=-1), None


class MeanPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out_dim = dim

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, None]:
        mean, _ = masked_mean_std(x, mask)
        return mean, None


# ------------------------------------------------------------------------ classifier
class AccentClassifier(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        num_classes: int,
        sample_rate: int = 16000,
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()
        if cfg.backbone == "mel_cnn":
            self.backbone: nn.Module = MelCNNBackbone(
                sample_rate, cfg.n_mels, cfg.hidden_dim, cfg.dropout
            )
        elif cfg.backbone == "hf":
            self.backbone = HFBackbone(cfg)
        else:
            raise ValueError(f"Unknown backbone '{cfg.backbone}'")
        dim = self.backbone.out_dim
        if cfg.pooling == "attentive_stats":
            self.pooling: nn.Module = AttentiveStatsPooling(dim, cfg.attention_dim)
        elif cfg.pooling == "mean_std":
            self.pooling = MeanStdPooling(dim)
        else:
            self.pooling = MeanPooling(dim)
        pooled = self.pooling.out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(pooled),
            nn.Dropout(cfg.dropout),
            nn.Linear(pooled, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, num_classes),
        )
        self.spec_augment = spec_augment
        self.num_classes = num_classes
        self.pretrained_backbone = cfg.backbone == "hf" and cfg.pretrained

    def forward(
        self,
        waveforms: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if lengths is None:
            if attention_mask is not None:
                lengths = attention_mask.long().sum(dim=1)
            else:
                lengths = torch.full(
                    (waveforms.shape[0],), waveforms.shape[1], device=waveforms.device
                )
        augment = self.spec_augment if self.training else None
        feats, frame_mask = self.backbone(waveforms, lengths.to(waveforms.device), augment)
        embedding, attention = self.pooling(feats, frame_mask)
        logits = self.head(embedding)
        return {
            "logits": logits,
            "embedding": embedding,
            "frame_mask": frame_mask,
            "attention": attention,
        }

    def param_groups(self, lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
        """Differential learning rates: pretrained transformer weights get
        ``backbone_lr``; pooling, head, layer weights (and mel_cnn) get ``lr``.
        Biases, norms and the layer-mixing weights get no weight decay."""
        pretrained = self.pretrained_backbone
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_backbone = pretrained and name.startswith("backbone.model.")
            no_decay = p.ndim <= 1 or name.endswith("layer_weights")
            groups.setdefault((is_backbone, no_decay), []).append(p)
        out = []
        for (is_backbone, no_decay), params in groups.items():
            out.append(
                {
                    "params": params,
                    "lr": backbone_lr if is_backbone else lr,
                    "weight_decay": 0.0 if no_decay else weight_decay,
                    "name": ("backbone" if is_backbone else "head") + ("_no_decay" * no_decay),
                }
            )
        return out


def build_model(cfg: Config, num_classes: int) -> AccentClassifier:
    g = cfg.augment
    spec = None
    if g.enabled and (g.spec_time_masks > 0 or g.spec_freq_masks > 0):
        spec = SpecAugment(
            g.spec_time_masks, g.spec_time_width, g.spec_freq_masks, g.spec_freq_width
        )
    return AccentClassifier(cfg.model, num_classes, cfg.audio.sample_rate, spec)


def checkpoint_payload(model: AccentClassifier, cfg: Config) -> dict[str, Any]:
    """What a checkpoint needs to rebuild the model offline (no Hub download)."""
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "accents": list(cfg.data.accents),
        "hf_config": None,
    }
    if isinstance(model.backbone, HFBackbone):
        payload["hf_config"] = model.backbone.model.config.to_dict()
    return payload


def load_accent_model(path: str, device: Any = "cpu") -> tuple[AccentClassifier, Config, dict]:
    """Load ``best.pt``: rebuild the architecture from the stored config (HF
    backbones from the stored HF config, so nothing is downloaded), then load
    the weights strictly."""
    from .utils import load_checkpoint

    ckpt = load_checkpoint(path, map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    if cfg.model.backbone == "hf":
        if not ckpt.get("hf_config"):
            raise ValueError(f"{path} has an hf backbone but no stored hf_config")
        cfg.model.pretrained = False
        cfg.model.hf_overrides = dict(ckpt["hf_config"])
    model = build_model(cfg, len(ckpt["accents"]))
    model.load_state_dict(ckpt["model"], strict=True)
    return model.to(device).eval(), cfg, ckpt


def model_summary(model: AccentClassifier) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total / 1e6:.2f} M parameters ({trainable / 1e6:.2f} M trainable)"


def expected_frames(model: AccentClassifier, num_samples: int) -> int:
    """Number of valid output frames for a clip of ``num_samples`` samples."""
    return int(model.backbone.frame_lengths(torch.tensor([num_samples]))[0])
