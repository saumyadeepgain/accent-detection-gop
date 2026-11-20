"""Synthetic accent-like audio for tests, CI and the ``make demo`` smoke run.

These clips are **not speech**. Each "accent" class gets its own pitch range,
two resonances and a syllable rate, and each speaker gets a personal pitch
offset and level. That is enough for the pipeline to learn something, so the
tests can check that splitting, training, evaluation and inference work end
to end. It says nothing about real accents: never report metrics measured on
this data as results.

    python -m src.synthetic data/synthetic --speakers 4 --utterances 3
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Sequence

import numpy as np
import soundfile as sf

from .config import DEFAULT_ACCENTS


def synth_clip(
    rng: np.random.Generator,
    sample_rate: int,
    duration: float,
    class_index: int,
    speaker_offset: float = 0.0,
) -> np.ndarray:
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    f0 = 110.0 + 18.0 * class_index + speaker_offset + 5.0 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    f1, f2 = 350.0 + 70.0 * class_index, 1100.0 + 170.0 * class_index
    mean_f0 = float(f0.mean())
    wave = np.zeros(n)
    for k in range(1, 13):
        freq = k * mean_f0
        amp = (
            0.05
            + np.exp(-(((freq - f1) / 150.0) ** 2))
            + 0.6 * np.exp(-(((freq - f2) / 250.0) ** 2))
        )
        wave += amp * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    rate = 3.0 + 0.35 * class_index  # "syllables" per second
    envelope = (0.5 * (1 + np.sin(2 * np.pi * rate * t + rng.uniform(0, 2 * np.pi)))) ** 1.5
    wave = envelope * wave + 0.01 * rng.standard_normal(n)
    return (0.5 * wave / (np.abs(wave).max() + 1e-9)).astype(np.float32)


def make_dataset(
    root: str,
    accents: Sequence[str] = DEFAULT_ACCENTS,
    speakers_per_accent: int = 4,
    utterances_per_speaker: int = 3,
    sample_rate: int = 16000,
    durations: tuple[float, float] = (1.5, 3.0),
    seed: int = 0,
    sources: Sequence[str] = ("CorpusA", "CorpusB"),
    text: str = "please call stella",
) -> str:
    """Write WAVs plus ``manifest.csv`` under ``root``; returns the manifest path.

    Every third clip is written at 22.05 kHz stereo so the resampling and
    down-mixing code paths are exercised too."""
    rng = np.random.default_rng(seed)
    rows = []
    counter = 0
    for c, accent in enumerate(accents):
        for s in range(speakers_per_accent):
            speaker = f"{accent[:3].lower()}{s:02d}"
            source = sources[s % len(sources)] if sources else ""
            offset, level = rng.uniform(-8, 8), rng.uniform(0.4, 1.0)
            for u in range(utterances_per_speaker):
                duration = rng.uniform(*durations)
                rel = os.path.join("audio", accent, f"{speaker}_{u}.wav")
                path = os.path.join(root, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                if counter % 3 == 2:
                    sr = 22050
                    clip = synth_clip(rng, sr, duration, c, offset) * level
                    sf.write(path, np.stack([clip, 0.8 * clip], axis=1), sr)
                else:
                    clip = synth_clip(rng, sample_rate, duration, c, offset) * level
                    sf.write(path, clip, sample_rate)
                rows.append([rel, speaker, accent, source, f"{speaker}_{u}", text])
                counter += 1
    manifest = os.path.join(root, "manifest.csv")
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "speaker_id", "accent", "source", "utt_id", "text"])
        writer.writerows(rows)
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write a small synthetic dataset (not speech).")
    ap.add_argument("root")
    ap.add_argument("--speakers", type=int, default=4, help="speakers per accent")
    ap.add_argument("--utterances", type=int, default=3, help="utterances per speaker")
    ap.add_argument("--accents", nargs="+", default=list(DEFAULT_ACCENTS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    path = make_dataset(args.root, args.accents, args.speakers, args.utterances, seed=args.seed)
    print(path)


if __name__ == "__main__":
    main()
