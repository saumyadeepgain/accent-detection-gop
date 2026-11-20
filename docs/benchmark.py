"""Measure the numbers quoted in the README (CPU, no downloads).

The accent and phoneme encoders are built at their real sizes (WavLM Base+;
wav2vec2 XLSR-53 large) with random weights: only compute time is measured,
which does not depend on the weight values. Writes docs/benchmark.json.

    python docs/benchmark.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aligner import forced_align
from src.audio import normalize_waveform, pad_batch
from src.config import load_config
from src.model import build_model

SR = 16000
torch.manual_seed(0)
rng = np.random.default_rng(0)


def timed(fn, repeat=3):
    fn()  # warm-up
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def clip(seconds):
    return normalize_waveform(rng.standard_normal(int(seconds * SR)).astype(np.float32))


results = {
    "machine": {
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
    }
}

# --- accent model, WavLM Base+ architecture (12 layers, 768-d), layers [6, 9] -> 9 kept
cfg = load_config(
    None,
    [
        "model.backbone=hf",
        "model.pretrained=false",
        "model.layers=[6,9]",
        "model.hf_overrides={feat_extract_norm: layer, do_stable_layer_norm: false}",
    ],
)
model = build_model(cfg, 7).eval()
x = torch.from_numpy(clip(6.0))[None]
with torch.inference_mode():
    results["accent_wavlm_base_plus_6s_clip_seconds"] = round(timed(lambda: model(x)), 3)
results["accent_wavlm_base_plus_params_M"] = round(
    sum(p.numel() for p in model.parameters()) / 1e6, 1
)

# --- padding invariance of that model: 2 s alone vs padded next to 5 s
two, five = clip(2.0), clip(5.0)
with torch.inference_mode():
    alone = model(torch.from_numpy(two)[None])["logits"][0]
    b = pad_batch([two, five])
    padded = model(b["waveforms"], b["attention_mask"])["logits"][0]
results["padding_invariance_max_abs_logit_diff"] = float((alone - padded).abs().max())

# --- mel_cnn accent model
mel = build_model(load_config(None, ["model.backbone=mel_cnn"]), 7).eval()
with torch.inference_mode():
    results["accent_mel_cnn_6s_clip_seconds"] = round(timed(lambda: mel(x)), 4)

# --- phoneme model, wav2vec2 XLSR-53 large architecture (24 layers, 1024-d), 5 s utterance
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC  # noqa: E402

ctc = Wav2Vec2ForCTC(
    Wav2Vec2Config(
        vocab_size=392,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
        conv_bias=True,
    )
).eval()
x5 = torch.from_numpy(clip(5.0))[None]
with torch.inference_mode():
    results["phoneme_xlsr53_large_5s_seconds"] = round(timed(lambda: ctc(x5).logits, 2), 3)
results["phoneme_xlsr53_large_params_M"] = round(sum(p.numel() for p in ctc.parameters()) / 1e6, 1)

# --- CTC forced alignment: 30 s of audio (1500 frames), 300 phones, 392-token vocabulary
lp = torch.log_softmax(torch.randn(1500, 392), dim=-1)
targets = list(rng.integers(1, 392, 300))
results["forced_align_30s_300_phones_seconds"] = round(
    timed(lambda: forced_align(lp, targets, 0)), 3
)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.json"), "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
