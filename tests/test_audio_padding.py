"""Padding invariance: a clip must get the same accent logits and embedding
whether it is scored alone or padded inside a batch next to a longer clip."""

import numpy as np
import pytest
import torch
from conftest import SR, make_cfg, mel_overrides, tiny_hf_overrides

from src.audio import normalize_waveform, pad_batch
from src.model import AttentiveStatsPooling, MeanStdPooling, build_model, expected_frames

TOL = 1e-4

BACKBONES = {
    "mel_cnn": mel_overrides(),
    "wavlm_layer_norm": tiny_hf_overrides("layer"),  # attention-mask path
    "wavlm_group_norm": tiny_hf_overrides("group"),  # per-length (unpadded) path
}


@pytest.fixture
def clips(rng):
    two = normalize_waveform(rng.standard_normal(2 * SR).astype(np.float32))
    five = normalize_waveform(rng.standard_normal(5 * SR).astype(np.float32))
    return two, five


@pytest.mark.parametrize("backbone", list(BACKBONES))
@pytest.mark.parametrize("position", [0, 1])
def test_two_second_clip_alone_vs_padded_next_to_five_seconds(backbone, position, clips):
    torch.manual_seed(0)
    model = build_model(make_cfg(*BACKBONES[backbone]), num_classes=7).eval()
    two, five = clips
    with torch.no_grad():
        alone = model(torch.from_numpy(two)[None])
        waves = [two, five] if position == 0 else [five, two]
        batch = pad_batch(waves)
        assert batch["waveforms"].shape == (2, 5 * SR)
        assert int(batch["attention_mask"][position].sum()) == 2 * SR
        padded = model(batch["waveforms"], batch["attention_mask"])

    torch.testing.assert_close(padded["logits"][position], alone["logits"][0], atol=TOL, rtol=0)
    torch.testing.assert_close(
        padded["embedding"][position], alone["embedding"][0], atol=TOL, rtol=0
    )
    frames = int(padded["frame_mask"][position].sum())
    assert frames == expected_frames(model, 2 * SR) == alone["frame_mask"].shape[1]
    # attentive pooling puts exactly zero weight on padded frames
    assert torch.all(padded["attention"][position, frames:] == 0)


def test_unmasked_mean_pooling_would_be_corrupted_by_padding(clips):
    """Sanity check of the test itself: a naive ``.mean(1)`` over padded frames
    differs by far more than the tolerance, so the invariance test has teeth."""
    torch.manual_seed(0)
    model = build_model(make_cfg(*mel_overrides()), num_classes=7).eval()
    two, five = clips
    with torch.no_grad():
        feats_alone, _ = model.backbone(torch.from_numpy(two)[None], torch.tensor([2 * SR]))
        b = pad_batch([two, five])
        feats_pad, _ = model.backbone(b["waveforms"], b["lengths"])
    naive_alone = feats_alone[0].mean(0)
    naive_padded = feats_pad[0].mean(0)
    assert (naive_alone - naive_padded).abs().max() > 100 * TOL


@pytest.mark.parametrize("pool_cls", [AttentiveStatsPooling, MeanStdPooling])
def test_pooling_ignores_whatever_is_in_padded_frames(pool_cls):
    torch.manual_seed(0)
    pool = pool_cls(8, 4) if pool_cls is AttentiveStatsPooling else pool_cls(8)
    x = torch.randn(1, 10, 8)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :6] = True
    garbage = x.clone()
    garbage[0, 6:] = 1e4  # huge values where the clip has ended
    clean, _ = pool(x, mask)
    dirty, _ = pool(garbage, mask)
    torch.testing.assert_close(clean, dirty, atol=1e-5, rtol=0)


def test_batch_of_equal_lengths_matches_single_clips(clips):
    """Group-norm backbones encode equal lengths together without padding."""
    torch.manual_seed(0)
    model = build_model(make_cfg(*tiny_hf_overrides("group")), num_classes=3).eval()
    two, _ = clips
    other = normalize_waveform(np.roll(two, 1234))
    with torch.no_grad():
        joint = model(pad_batch([two, other])["waveforms"])["logits"]
        a = model(torch.from_numpy(two)[None])["logits"][0]
        b = model(torch.from_numpy(other)[None])["logits"][0]
    torch.testing.assert_close(joint, torch.stack([a, b]), atol=TOL, rtol=0)
