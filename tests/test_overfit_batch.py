"""Training sanity: one synthetic batch must be learnable, and every trainable
parameter must receive a gradient (no dead branches, no detached graph)."""

import numpy as np
import pytest
import torch
from conftest import SR, make_cfg, mel_overrides, tiny_hf_overrides

from src.audio import normalize_waveform, pad_batch
from src.dataset import class_weights
from src.model import build_model
from src.synthetic import synth_clip
from src.train import FocalLoss, make_loss

NO_AUG = ["augment.enabled=false", "model.dropout=0.0"]


def synthetic_batch(num_classes: int = 3, per_class: int = 2):
    """Variable-length clips (padding is part of what is being tested)."""
    rng = np.random.default_rng(0)
    waves, labels = [], []
    for c in range(num_classes):
        for k in range(per_class):
            duration = 0.8 + 0.3 * k + 0.1 * c
            waves.append(normalize_waveform(synth_clip(rng, SR, duration, c)))
            labels.append(c)
    batch = pad_batch(waves)
    batch["labels"] = torch.tensor(labels)
    return batch


@pytest.mark.parametrize(
    "overrides",
    [mel_overrides(), tiny_hf_overrides("layer"), tiny_hf_overrides("group")],
    ids=["mel_cnn", "hf_layer_norm", "hf_group_norm"],
)
def test_25_steps_on_one_batch_drive_the_loss_down(overrides):
    torch.manual_seed(0)
    cfg = make_cfg(*overrides, *NO_AUG)
    model = build_model(cfg, num_classes=3).train()
    batch = synthetic_batch()
    optimizer = torch.optim.AdamW(model.param_groups(3e-3, 3e-3, 0.0))
    loss_fn = make_loss(cfg.train, None)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    losses = []
    for step in range(25):
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["waveforms"], batch["attention_mask"])
        loss = loss_fn(out["logits"], batch["labels"])
        loss.backward()
        if step == 0:
            missing = [n for n, p in trainable.items() if p.grad is None]
            dead = [
                n for n, p in trainable.items() if p.grad is not None and p.grad.abs().sum() == 0
            ]
            assert not missing, f"no gradient reached: {missing[:5]}"
            assert not dead, f"all-zero gradient: {dead[:5]}"
        optimizer.step()
        losses.append(loss.item())
    assert np.isfinite(losses).all()
    assert losses[-1] < 0.5 * losses[0], f"loss {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_frozen_feature_encoder_and_backbone():
    cfg = make_cfg(*tiny_hf_overrides("layer"))
    model = build_model(cfg, 3)
    encoder = model.backbone.model.feature_extractor
    assert not any(p.requires_grad for p in encoder.parameters())
    assert any(p.requires_grad for p in model.backbone.model.encoder.parameters())

    frozen = build_model(make_cfg(*tiny_hf_overrides("layer"), "model.freeze_backbone=true"), 3)
    assert not any(p.requires_grad for p in frozen.backbone.model.parameters())
    trainable = {n for n, p in frozen.named_parameters() if p.requires_grad}
    assert "backbone.layer_weights" in trainable
    assert all(n.startswith(("head.", "pooling.", "backbone.layer_weights")) for n in trainable)


def test_differential_learning_rates_for_pretrained_backbones():
    cfg = make_cfg(*tiny_hf_overrides("layer"))
    model = build_model(cfg, 3)
    model.pretrained_backbone = True  # as if loaded from the Hub
    groups = {g["name"]: g for g in model.param_groups(1e-3, 1e-5, 0.01)}
    assert groups["backbone"]["lr"] == 1e-5 and groups["head"]["lr"] == 1e-3
    assert groups["head_no_decay"]["weight_decay"] == 0.0
    assigned = sum(len(g["params"]) for g in groups.values())
    assert assigned == sum(1 for p in model.parameters() if p.requires_grad)


def test_unused_upper_layers_are_dropped():
    model = build_model(make_cfg(*tiny_hf_overrides("layer")), 3)  # layers [1, 2] of 3
    assert len(model.backbone.model.encoder.layers) == 2


def test_focal_loss_equals_cross_entropy_when_gamma_is_zero():
    torch.manual_seed(0)
    logits, target = torch.randn(8, 4), torch.randint(0, 4, (8,))
    weight = torch.tensor([1.0, 2.0, 0.5, 1.5])
    ce = torch.nn.functional.cross_entropy(logits, target, weight=weight)
    torch.testing.assert_close(FocalLoss(0.0, weight)(logits, target), ce)
    assert FocalLoss(2.0)(logits, target) < torch.nn.functional.cross_entropy(logits, target)


def test_class_weights_are_inverse_frequency():
    from src.dataset import Record

    recs = [Record("x", f"s{i}", a) for i, a in enumerate(["A"] * 6 + ["B"] * 2)]
    w = class_weights(recs, ["A", "B", "C"])
    torch.testing.assert_close(w, torch.tensor([8 / (3 * 6), 8 / (3 * 2), 0.0]))
