"""Typed configuration: ``config.yaml`` plus command-line overrides.

    cfg = load_config("config.yaml", ["train.lr=3e-4", "model.layers=[4,8]"])

Every leaf key is a CLI flag (``--train.lr``) and, where the name is not
ambiguous, also a short flag (``--lr``). Override values are parsed as YAML,
so lists and mappings work (``--set gop.rivals={v: [w]}``), and are then
checked against the field's declared type. Unknown keys fail loudly with a
suggestion instead of being ignored.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import os
import types
from dataclasses import dataclass, field
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml

DEFAULT_ACCENTS = [
    "Tamil",
    "Telugu",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Bengali",
]

# Target phone -> substitutions typical of Indian-English L1 transfer (eSpeak IPA symbols).
DEFAULT_RIVALS = {
    "v": ["w", "ʋ"],
    "w": ["v", "ʋ"],
    "θ": ["t", "t̪"],
    "ð": ["d", "d̪"],
    "f": ["p", "pʰ"],
    "z": ["s", "dʒ"],
    "s": ["ʃ"],
    "ʃ": ["s"],
    "ɡ": ["k"],
}


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    mono: bool = True
    trim_silence: bool = True
    trim_top_db: float = 40.0
    normalize: bool = True
    min_duration: float = 0.5
    max_duration: float = 8.0
    silence_rms_db: float = -60.0


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    audio_root: str | None = None
    accents: list[str] = field(default_factory=lambda: list(DEFAULT_ACCENTS))
    val_frac: float = 0.15
    test_frac: float = 0.15
    test_sources: list[str] = field(default_factory=list)
    split_file: str | None = None
    verify_audio: bool = True
    num_workers: int = 2
    bucket_by_length: bool = True


@dataclass
class AugmentConfig:
    enabled: bool = True
    noise_prob: float = 0.5
    noise_snr_db: list[float] = field(default_factory=lambda: [10.0, 30.0])
    noise_dir: str | None = None
    gain_prob: float = 0.0
    gain_db: list[float] = field(default_factory=lambda: [-6.0, 6.0])
    spec_time_masks: int = 2
    spec_time_width: float = 0.05
    spec_freq_masks: int = 2
    spec_freq_width: float = 0.08


@dataclass
class ModelConfig:
    backbone: str = "hf"
    hf_name: str = "microsoft/wavlm-base-plus"
    hf_arch: str = "wavlm"
    pretrained: bool = True
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    layers: list[int] = field(default_factory=lambda: [6, 9])
    freeze_backbone: bool = False
    freeze_feature_encoder: bool = True
    n_mels: int = 80
    hidden_dim: int = 256
    pooling: str = "attentive_stats"
    attention_dim: int = 128
    dropout: float = 0.2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr: float = 1e-3
    backbone_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.01
    max_grad_norm: float = 5.0
    amp: bool = True
    loss: str = "ce"
    focal_gamma: float = 2.0
    class_weights: str = "auto"
    label_smoothing: float = 0.0
    patience: int = 6
    min_delta: float = 0.001
    select_metric: str = "macro_f1"
    out_dir: str = "runs/default"
    resume: str | None = None
    log_every_n_steps: int = 20
    calibrate: bool = True
    stop_after_epochs: int | None = None


@dataclass
class EvalConfig:
    batch_size: int = 16
    segment_seconds: float = 6.0
    segment_hop_seconds: float = 3.0
    contested_agreement: float = 0.6
    ece_bins: int = 15


@dataclass
class GOPConfig:
    phoneme_model: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    g2p: str = "espeak"
    lexicon: str | None = None
    language: str = "en-us"
    exclude_blank: bool = True
    score_center: float = 0.0
    score_scale: float = 2.0
    flag_threshold: float = 0.5
    word_aggregate: str = "mean"
    unknown_phones: str = "skip"
    rivals: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_RIVALS.items()}
    )


SECTIONS = ("audio", "data", "augment", "model", "train", "eval", "gop")


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    deterministic: bool = False
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    gop: GOPConfig = field(default_factory=GOPConfig)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        cfg = cls()
        for key, value in _flatten(data):
            set_by_path(cfg, key, value)
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    # ----------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Raise ``ValueError`` listing every invalid setting; return soft warnings."""
        errors: list[str] = []
        warnings: list[str] = []

        def check(cond: bool, msg: str) -> None:
            if not cond:
                errors.append(msg)

        def one_of(value: str, allowed: tuple[str, ...], key: str) -> None:
            check(value in allowed, f"{key}={value!r} must be one of {allowed}")

        a, d, g, m, t, e, p = (
            self.audio,
            self.data,
            self.augment,
            self.model,
            self.train,
            self.eval,
            self.gop,
        )
        check(a.sample_rate >= 4000, "audio.sample_rate must be >= 4000")
        check(a.min_duration > 0, "audio.min_duration must be > 0")
        check(
            a.max_duration >= a.min_duration,
            "audio.max_duration must be >= min_duration",
        )
        check(a.trim_top_db > 0, "audio.trim_top_db must be > 0")

        check(len(d.accents) >= 2, "data.accents needs at least two classes")
        check(len(set(d.accents)) == len(d.accents), "data.accents contains duplicates")
        check(
            0 <= d.val_frac < 1 and 0 <= d.test_frac < 1,
            "data.val_frac/test_frac in [0, 1)",
        )
        check(d.val_frac + d.test_frac < 1, "data.val_frac + data.test_frac must be < 1")
        check(d.num_workers >= 0, "data.num_workers must be >= 0")

        for name in ("noise_snr_db", "gain_db"):
            lo_hi = getattr(g, name)
            check(
                len(lo_hi) == 2 and lo_hi[0] <= lo_hi[1],
                f"augment.{name} must be [low, high] with low <= high",
            )
        for name in ("noise_prob", "gain_prob"):
            check(0 <= getattr(g, name) <= 1, f"augment.{name} must be in [0, 1]")
        for name in ("spec_time_width", "spec_freq_width"):
            check(0 <= getattr(g, name) < 1, f"augment.{name} must be in [0, 1)")
        check(
            g.spec_time_masks >= 0 and g.spec_freq_masks >= 0,
            "augment.spec_*_masks >= 0",
        )
        gain_matters = m.backbone == "hf" and not a.normalize
        if g.enabled and g.gain_prob > 0 and not gain_matters:
            warnings.append(
                "augment.gain_prob has no effect here: per-utterance normalisation "
                "(audio.normalize) and the mel_cnn log-mel mean normalisation both remove "
                "a constant gain. It only matters for hf backbones with audio.normalize=false."
            )

        one_of(m.backbone, ("hf", "mel_cnn"), "model.backbone")
        one_of(m.hf_arch, ("wavlm", "wav2vec2", "hubert"), "model.hf_arch")
        one_of(m.pooling, ("attentive_stats", "mean_std", "mean"), "model.pooling")
        check(
            len(m.layers) >= 1 and min(m.layers) >= 0,
            "model.layers must be non-negative",
        )
        check(m.hidden_dim > 0 and m.attention_dim > 0, "model dims must be > 0")
        check(0 <= m.dropout < 1, "model.dropout must be in [0, 1)")
        if m.backbone == "mel_cnn" and m.layers != ModelConfig().layers:
            warnings.append("model.layers is ignored by the mel_cnn backbone")

        check(
            t.epochs >= 1 and t.batch_size >= 1,
            "train.epochs and batch_size must be >= 1",
        )
        check(t.grad_accum_steps >= 1, "train.grad_accum_steps must be >= 1")
        check(t.lr > 0 and t.backbone_lr > 0, "learning rates must be > 0")
        check(0 <= t.warmup_ratio < 1, "train.warmup_ratio must be in [0, 1)")
        check(0 <= t.min_lr_ratio <= 1, "train.min_lr_ratio must be in [0, 1]")
        check(t.max_grad_norm > 0, "train.max_grad_norm must be > 0")
        one_of(t.loss, ("ce", "focal"), "train.loss")
        one_of(t.class_weights, ("auto", "none"), "train.class_weights")
        one_of(t.select_metric, ("macro_f1", "uar", "accuracy"), "train.select_metric")
        check(0 <= t.label_smoothing < 1, "train.label_smoothing must be in [0, 1)")
        check(t.focal_gamma >= 0, "train.focal_gamma must be >= 0")
        check(t.patience >= 1, "train.patience must be >= 1")
        if t.loss == "focal" and t.label_smoothing > 0:
            warnings.append("train.label_smoothing is ignored by the focal loss")
        if t.stop_after_epochs is not None:
            check(t.stop_after_epochs >= 1, "train.stop_after_epochs must be >= 1")

        check(e.batch_size >= 1, "eval.batch_size must be >= 1")
        check(
            e.segment_seconds >= a.min_duration,
            "eval.segment_seconds must be >= min_duration",
        )
        check(
            0 < e.segment_hop_seconds <= e.segment_seconds,
            "eval.segment_hop_seconds must be in (0, segment_seconds]",
        )
        check(
            0 <= e.contested_agreement <= 1,
            "eval.contested_agreement must be in [0, 1]",
        )
        check(e.ece_bins >= 2, "eval.ece_bins must be >= 2")

        one_of(p.g2p, ("espeak", "lexicon"), "gop.g2p")
        one_of(p.word_aggregate, ("mean", "min"), "gop.word_aggregate")
        one_of(p.unknown_phones, ("skip", "error"), "gop.unknown_phones")
        check(p.score_scale > 0, "gop.score_scale must be > 0")
        check(0 <= p.flag_threshold <= 1, "gop.flag_threshold must be in [0, 1]")

        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))
        return warnings


# --------------------------------------------------------------------------- helpers
def _flatten(data: dict[str, Any]) -> list[tuple[str, Any]]:
    """Dotted ``section.key`` pairs. Dict-valued *fields* (e.g. ``gop.rivals``) stay leaves."""
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    out: list[tuple[str, Any]] = []
    for key, value in data.items():
        if key in SECTIONS:
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"Config section '{key}' must be a mapping")
            out += [(f"{key}.{sub}", v) for sub, v in value.items()]
        else:
            out.append((key, value))
    return out


def _field_types(obj: Any) -> dict[str, Any]:
    return get_type_hints(type(obj))


def config_keys() -> list[str]:
    """All dotted leaf keys, e.g. ``['seed', ..., 'train.lr', ...]``."""
    cfg = Config()
    keys = [f.name for f in dataclasses.fields(cfg) if f.name not in SECTIONS]
    for section in SECTIONS:
        keys += [f"{section}.{f.name}" for f in dataclasses.fields(getattr(cfg, section))]
    return keys


def _resolve(cfg: Config, key: str) -> tuple[Any, str, Any]:
    """Return ``(owner object, attribute name, declared type)`` for a dotted key."""
    keys = config_keys()
    if key not in keys:
        close = difflib.get_close_matches(key, keys, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown config key '{key}'.{hint}")
    if "." in key:
        section, name = key.split(".", 1)
        owner = getattr(cfg, section)
    else:
        owner, name = cfg, key
    return owner, name, _field_types(owner)[name]


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    if get_origin(tp) in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0], len(args) < len(get_args(tp))
    return tp, False


def _coerce(value: Any, tp: Any, key: str) -> Any:
    """Check / convert ``value`` to the declared type ``tp`` (raises ``ValueError``)."""
    inner, optional = _unwrap_optional(tp)
    if value is None or (
        optional and isinstance(value, str) and value.strip().lower() in ("null", "none", "")
    ):
        if optional:
            return None
        raise ValueError(f"{key} cannot be null")
    if inner is Any:
        return value
    origin = get_origin(inner)
    if origin is list:
        (elem,) = get_args(inner)
        if isinstance(value, str):  # convenience: "6,9" or "a, b"
            value = [v.strip() for v in value.strip("[]").split(",") if v.strip()]
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{key} expects a list, got {value!r}")
        return [_coerce(v, elem, f"{key}[{i}]") for i, v in enumerate(value)]
    if origin is dict:
        _, val_t = get_args(inner)
        if not isinstance(value, dict):
            raise ValueError(f"{key} expects a mapping, got {value!r}")
        return {str(k): _coerce(v, val_t, f"{key}.{k}") for k, v in value.items()}
    if inner is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.lower() in (
            "true",
            "false",
            "yes",
            "no",
            "1",
            "0",
        ):
            return value.lower() in ("true", "yes", "1")
        raise ValueError(f"{key} expects true/false, got {value!r}")
    if inner is int:
        if isinstance(value, bool):
            raise ValueError(f"{key} expects an integer, got {value!r}")
        if isinstance(value, float) and value.is_integer():
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} expects an integer, got {value!r}") from None
    if inner is float:
        if isinstance(value, bool):
            raise ValueError(f"{key} expects a number, got {value!r}")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} expects a number, got {value!r}") from None
    if inner is str:
        if isinstance(value, (dict, list, bool)):
            raise ValueError(f"{key} expects a string, got {value!r}")
        return str(value)
    raise TypeError(f"Unsupported config type {tp!r} for {key}")


def set_by_path(cfg: Config, key: str, value: Any) -> None:
    owner, name, tp = _resolve(cfg, key)
    setattr(owner, name, _coerce(value, tp, key))


def parse_override(text: str) -> tuple[str, Any]:
    """``"train.lr=3e-4"`` -> ``("train.lr", "3e-4")`` (value parsed later, by type)."""
    if "=" not in text:
        raise ValueError(f"Override '{text}' must look like key=value")
    key, raw = text.split("=", 1)
    return key.strip(), raw.strip()


def _parse_raw(raw: Any, tp: Any) -> Any:
    """Turn a CLI string into a Python value: plain text for str fields, YAML otherwise."""
    if not isinstance(raw, str):
        return raw
    inner, _ = _unwrap_optional(tp)
    if inner is str:
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
            return raw[1:-1]
        return raw
    try:
        return yaml.safe_load(raw) if raw != "" else None
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: Config, overrides: list[str] | None) -> Config:
    for text in overrides or []:
        key, raw = parse_override(text)
        _, _, tp = _resolve(cfg, key)
        set_by_path(cfg, key, _parse_raw(raw, tp))
    return cfg


def resolve_config_path(path: str | None) -> str | None:
    """``None`` means: ``./config.yaml`` if it exists, else built-in defaults."""
    if path is None:
        return "config.yaml" if os.path.isfile("config.yaml") else None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def load_config(
    path: str | None = None, overrides: list[str] | None = None, validate: bool = True
) -> Config:
    path = resolve_config_path(path)
    cfg = Config()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = Config.from_dict(data)
    apply_overrides(cfg, overrides)
    if validate:
        cfg.validate()
    return cfg


# ------------------------------------------------------------------------------ CLI
def short_aliases(
    priority: tuple[str, ...] = (
        "train",
        "model",
        "data",
        "audio",
        "augment",
        "eval",
        "gop",
    ),
    exclude: set[str] | None = None,
) -> dict[str, str]:
    """Map dotted keys to short flag names (``train.lr`` -> ``lr``).

    A name shared by several sections goes to the first section in ``priority``.
    Names equal to a top-level key or listed in ``exclude`` get no alias."""
    exclude = set(exclude or ())
    top = {k for k in config_keys() if "." not in k}
    taken: dict[str, str] = {}
    for section in priority:
        for key in config_keys():
            if not key.startswith(section + "."):
                continue
            name = key.split(".", 1)[1]
            if name in top or name in exclude or name in SECTIONS or name in taken:
                continue
            taken[name] = key
    return {key: name for name, key in taken.items()}


def add_config_arguments(
    parser: argparse.ArgumentParser,
    priority: tuple[str, ...] = (
        "train",
        "model",
        "data",
        "audio",
        "augment",
        "eval",
        "gop",
    ),
    exclude: set[str] | None = None,
) -> argparse.ArgumentParser:
    """Add ``--config``, ``--set KEY=VALUE`` and one flag per config key."""
    parser.add_argument(
        "--config", default=None, help="YAML config (default: ./config.yaml if present)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override any key, e.g. --set model.layers=[4,8] (repeatable)",
    )
    group = parser.add_argument_group("config overrides (any key in config.yaml)")
    aliases = short_aliases(priority, exclude)
    for key in config_keys():
        flags = [f"--{key}"]
        if key in aliases:
            flags.append(f"--{aliases[key]}")
        group.add_argument(
            *flags, dest="cfg__" + key.replace(".", "__"), default=None, metavar="VALUE"
        )
    return parser


def config_from_args(args: argparse.Namespace, validate: bool = True) -> Config:
    """Build the config from ``--config``, flag overrides and ``--set`` (``--set`` wins)."""
    overrides = []
    for key in config_keys():
        value = getattr(args, "cfg__" + key.replace(".", "__"), None)
        if value is not None:
            overrides.append(f"{key}={value}")
    overrides += list(args.set or [])
    return load_config(args.config, overrides, validate=validate)
