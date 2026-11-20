"""Produce docs/sample_report.json, the example shown in the README.

Everything goes through the real code (``assess_pronunciation``,
``predict_waves``), but the inputs are synthetic so that it runs offline:

* pronunciation: frame posteriors in which /w/ in "world" was pronounced as
  /v/ (the scenario of ``tests/test_gop.py``), aligned and scored by src.gop;
* accent: the ``make demo`` model (trained on synthetic tones, not speech) on
  one synthetic clip, so the probabilities only show the format.

    make demo && python docs/make_sample_report.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.aligner import LexiconG2P  # noqa: E402
from src.config import GOPConfig  # noqa: E402
from src.evaluate import predict_waves  # noqa: E402
from src.gop import assess_pronunciation  # noqa: E402
from src.model import load_accent_model  # noqa: E402
from src.synthetic import synth_clip  # noqa: E402
from src.utils import jsonable  # noqa: E402

TOKENS = ["<pad>", "h", "ə", "l", "oʊ", "w", "ɜː", "d", "v", "ʋ"]
VOCAB = {t: i for i, t in enumerate(TOKENS)}
# (token or None = blank, frames): "hello world", /w/ realised as /v/
TIMELINE = [
    (None, 12),
    ("h", 2),
    (None, 1),
    ("ə", 2),
    ("l", 2),
    (None, 1),
    ("oʊ", 3),
    (None, 9),
    ("w", 2),
    ("ɜː", 3),
    ("l", 2),
    ("d", 2),
    (None, 10),
]


class SyntheticRecognizer:
    vocab, blank_id, frame_shift, sample_rate = VOCAB, 0, 0.02, 16000

    def log_probs(self, wave):
        frames = sum(n for _, n in TIMELINE)
        p = np.full((frames, len(TOKENS)), 0.01)
        t = 0
        for tok, n in TIMELINE:
            for _ in range(n):
                if tok is None:
                    p[t, 0] = 0.95
                elif tok == "w":  # mispronounced: mostly /v/
                    p[t, VOCAB["w"]], p[t, VOCAB["v"]] = 0.15, 0.75
                else:
                    p[t, VOCAB[tok]] = 0.9
                t += 1
        return torch.log(torch.tensor(p / p.sum(1, keepdims=True), dtype=torch.float32))


def main() -> None:
    lexicon = LexiconG2P({"hello": ["h", "ə", "l", "oʊ"], "world": ["w", "ɜː", "l", "d"]})
    wave = np.zeros(sum(n for _, n in TIMELINE) * 320, dtype=np.float32)
    pron = assess_pronunciation(
        wave, "Hello world", SyntheticRecognizer(), lexicon, GOPConfig(rivals={"w": ["v", "ʋ"]})
    )
    report = {
        "file": "hello_world (synthetic posteriors)",
        "duration": round(len(wave) / 16000, 2),
        "sample_rate": 16000,
        "channels": 1,
        "error": None,
        "warnings": [],
        "pronunciation": pron,
    }
    ckpt = os.path.join(ROOT, "runs", "demo", "run", "best.pt")
    if os.path.isfile(ckpt):
        model, cfg, meta = load_accent_model(ckpt)
        clip = synth_clip(np.random.default_rng(1), 16000, 2.0, class_index=0)
        res = predict_waves(model, [clip], cfg, "cpu", meta.get("temperature", 1.0))[0]
        probs = dict(
            sorted(
                zip(meta["accents"], res["probs"].round(4).tolist(), strict=True),
                key=lambda kv: -kv[1],
            )
        )
        label = next(iter(probs))
        report["accent"] = {
            "label": label,
            "confidence": probs[label],
            "probabilities": probs,
            "segments": res["segments"],
            "agreement": res["agreement"],
            "verdict": res["verdict"],
            "temperature": meta.get("temperature", 1.0),
        }
    out = os.path.join(ROOT, "docs", "sample_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            jsonable(
                {
                    "models": {
                        "accent_checkpoint": "runs/demo/run/best.pt",
                        "phoneme_model": "(synthetic posteriors)",
                        "g2p": "lexicon",
                        "flag_threshold": 0.5,
                    },
                    "results": [report],
                }
            ),
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
