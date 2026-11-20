"""Manifest parsing, speaker-disjoint splits, augmentation and batching."""

import csv

import numpy as np
import pytest
import torch
from conftest import SR, TEST_ACCENTS

from src.config import AugmentConfig
from src.dataset import (
    BucketBatchSampler,
    LeakageError,
    Record,
    SpecAugment,
    WaveformAugment,
    assert_speaker_disjoint,
    collate_batch,
    load_split,
    read_manifest,
    save_split,
    speaker_disjoint_split,
)


def records(n_speakers=6, utts=4, accents=("A", "B", "C"), sources=("X",)):
    out = []
    for a in accents:
        for s in range(n_speakers):
            src = sources[s % len(sources)]
            for u in range(utts):
                out.append(Record(f"{a}{s}_{u}.wav", f"{src}/{a}{s}", a, src, f"{a}{s}_{u}"))
    return out


def speakers(recs):
    return {r.speaker for r in recs}


# ---------------------------------------------------------------------- splits
def test_split_is_speaker_disjoint_and_stratified():
    recs = records()
    splits = speaker_disjoint_split(recs, ["A", "B", "C"], 0.2, 0.2, seed=0)
    assert_speaker_disjoint(splits)
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not speakers(splits[a]) & speakers(splits[b])
    for accent in "ABC":  # every accent present in every split
        for name in ("train", "val", "test"):
            assert any(r.accent == accent for r in splits[name])
    # all utterances of a speaker stay together
    assert sum(len(v) for v in splits.values()) == len(recs)
    for recs_ in splits.values():
        expected = {r.utt_id for r in recs if r.speaker in speakers(recs_)}
        assert {r.utt_id for r in recs_} == expected


def test_split_is_deterministic_for_a_seed():
    a = speaker_disjoint_split(records(), ["A", "B", "C"], 0.2, 0.2, seed=7)
    b = speaker_disjoint_split(records(), ["A", "B", "C"], 0.2, 0.2, seed=7)
    assert {k: speakers(v) for k, v in a.items()} == {k: speakers(v) for k, v in b.items()}


def test_cross_corpus_holdout():
    recs = records(sources=("NISP", "Svarah"))
    splits = speaker_disjoint_split(recs, ["A", "B", "C"], 0.2, 0.2, 0, test_sources=["Svarah"])
    assert {r.source for r in splits["test"]} == {"Svarah"}
    assert {r.source for r in splits["train"] + splits["val"]} == {"NISP"}


def test_same_id_in_two_corpora_is_two_speakers(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text(
        "path,speaker_id,accent,source\na.wav,spk1,Tamil,NISP\nb.wav,spk1,Hindi,Svarah\n",
        encoding="utf-8",
    )
    recs = read_manifest(str(path))
    assert [r.speaker for r in recs] == ["NISP/spk1", "Svarah/spk1"]


def test_speaker_with_two_accent_labels_is_rejected():
    recs = records(n_speakers=2, utts=1, accents=("A",)) + [Record("z.wav", "X/A0", "B", "X", "z")]
    with pytest.raises(ValueError, match="several accent labels"):
        speaker_disjoint_split(recs, ["A", "B"], 0.2, 0.2, 0)


def test_unknown_accent_label_is_rejected():
    with pytest.raises(ValueError, match="not in data.accents"):
        speaker_disjoint_split(records(accents=("A", "Z")), ["A", "B"], 0.2, 0.2, 0)


def test_saved_split_round_trips_and_leakage_is_caught(tmp_path):
    recs = records()
    splits = speaker_disjoint_split(recs, ["A", "B", "C"], 0.2, 0.2, 0)
    path = tmp_path / "split.csv"
    save_split(splits, str(path))
    again = load_split(str(path), recs)
    assert {k: len(v) for k, v in again.items()} == {k: len(v) for k, v in splits.items()}
    # tamper: move one utterance of a training speaker into the test split
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    victim = next(r for r in rows if r["split"] == "train")
    victim["split"] = "test"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(LeakageError):
        load_split(str(path), recs)


# -------------------------------------------------------------------- manifest
def test_manifest_with_bom_and_relative_paths(tmp_path):
    (tmp_path / "audio").mkdir()
    path = tmp_path / "m.csv"
    path.write_text(
        "﻿path,speaker_id,accent,text\naudio/a.wav,s1,Tamil,hello there\n", encoding="utf-8"
    )
    (rec,) = read_manifest(str(path))
    assert rec.path == str(tmp_path / "audio" / "a.wav") and rec.text == "hello there"


@pytest.mark.parametrize(
    "content, message",
    [
        ("path,accent\na.wav,Tamil\n", "missing column"),
        ("path,speaker_id,accent\na.wav,,Tamil\n", "non-empty"),
        ("path,speaker_id,accent\na.wav,s1,Tamil\na.wav,s2,Hindi\n", "duplicate"),
    ],
)
def test_bad_manifests_fail_clearly(tmp_path, content, message):
    path = tmp_path / "m.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_manifest(str(path))


def test_synthetic_manifest_loads(synthetic_data):
    recs = read_manifest(synthetic_data)
    assert len(recs) == len(TEST_ACCENTS) * 4 * 3
    assert {r.accent for r in recs} == set(TEST_ACCENTS)


# ---------------------------------------------------------------- augmentation
def test_additive_noise_hits_the_requested_snr(rng):
    cfg = AugmentConfig(noise_prob=1.0, noise_snr_db=[10.0, 10.0], gain_prob=0.0)
    aug = WaveformAugment(cfg, SR)
    clean = np.sin(2 * np.pi * 200 * np.arange(SR) / SR).astype(np.float32)
    noisy = aug(clean, rng)
    noise = noisy - clean
    snr = 10 * np.log10(np.mean(clean**2) / np.mean(noise**2))
    assert snr == pytest.approx(10.0, abs=0.1)


def test_augmentation_leaves_silence_alone(rng):
    aug = WaveformAugment(AugmentConfig(noise_prob=1.0, gain_prob=1.0), SR)
    assert np.all(aug(np.zeros(SR, dtype=np.float32), rng) == 0)


def test_gain_stays_within_range(rng):
    cfg = AugmentConfig(noise_prob=0.0, gain_prob=1.0, gain_db=[-6.0, 6.0])
    x = np.ones(100, dtype=np.float32)
    gains = [WaveformAugment(cfg, SR)(x, rng)[0] for _ in range(50)]
    assert 10 ** (-6 / 20) - 1e-6 <= min(gains) and max(gains) <= 10 ** (6 / 20) + 1e-6


def test_spec_augment_masks_only_valid_frames_and_only_in_training():
    torch.manual_seed(0)
    spec = SpecAugment(time_masks=3, time_width=0.3, freq_masks=2, freq_width=0.2).train()
    x = torch.ones(2, 20, 10)
    lengths = torch.tensor([20, 8])
    y = spec(x, lengths)
    assert torch.all(y[1, 8:] == 1)  # padded frames are never touched
    assert (y == 0).any()
    spec.eval()
    assert torch.equal(spec(x, lengths), x)


# -------------------------------------------------------------------- batching
def test_collate_pads_and_keeps_ids():
    items = [
        {"waveform": torch.ones(5), "label": 1, "speaker_id": "s1", "utt_id": "u1"},
        {"waveform": torch.ones(3), "label": 0, "speaker_id": "s2", "utt_id": "u2"},
    ]
    batch = collate_batch(items)
    assert batch["waveforms"].shape == (2, 5) and batch["lengths"].tolist() == [5, 3]
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert batch["labels"].tolist() == [1, 0] and batch["utt_ids"] == ["u1", "u2"]


def test_bucket_sampler_covers_every_index_once_and_groups_lengths():
    durations = np.random.default_rng(0).uniform(1, 10, size=103)
    sampler = BucketBatchSampler(durations, batch_size=8, seed=0, pool_factor=4)
    batches = list(sampler)
    flat = sorted(i for b in batches for i in b)
    assert flat == list(range(103)) and len(batches) == len(sampler)
    spread = np.mean([np.ptp(durations[b]) for b in batches if len(b) > 1])
    assert spread < 0.5 * np.ptp(durations)  # batches are much tighter than random
    sampler.set_epoch(1)
    assert list(sampler) != batches  # a new order every epoch
    sampler.set_epoch(0)
    assert list(sampler) == batches  # ...and reproducible
