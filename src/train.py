"""Train the accent classifier.

    python -m src.train                                     # settings from config.yaml
    python -m src.train --lr 3e-4 --batch_size 8 --set model.layers=[4,8]
    python -m src.train --resume runs/default/last.pt       # continue an interrupted run

Pipeline: manifest -> audio check -> speaker-disjoint split (saved to
``<out_dir>/split.csv`` and reused) -> length-bucketed batches -> AMP forward
-> class-weighted CE or focal loss -> gradient accumulation and clipping ->
warmup + cosine LR per optimizer step -> validation on whole utterances,
sectioned exactly like inference -> early stopping on macro-F1 / UAR. When
training ends, the softmax temperature is fitted on the validation split and
the test split is evaluated once.

``last.pt`` holds everything needed for an exact resume (model, optimizer,
scheduler, scaler, RNG states, early-stopping state, history); ``best.pt``
holds only what inference needs.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):  # executed as `python src/train.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")

from .config import Config, TrainConfig, add_config_arguments, config_from_args
from .dataset import (
    BucketBatchSampler,
    Record,
    build_train_loader,
    class_weights,
    load_split,
    read_manifest,
    save_split,
    speaker_disjoint_split,
    verify_records,
)
from .evaluate import (
    accent_report,
    classification_metrics,
    fit_temperature,
    negative_log_likelihood,
    predict_records,
    softmax_np,
)
from .model import AccentClassifier, build_model, checkpoint_payload, model_summary
from .utils import (
    LOGGER_NAME,
    get_device,
    get_logger,
    get_rng_state,
    load_checkpoint,
    quiet_transformers,
    save_checkpoint,
    set_rng_state,
    set_seed,
    timestamp,
    write_json,
)

log = logging.getLogger(LOGGER_NAME)


# ---------------------------------------------------------------------------- loss
class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) on logits, with optional class
    weights as alpha. Computed from ``log_softmax`` for numerical stability."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -((1.0 - logp.exp()) ** self.gamma) * logp
        if self.weight is not None:
            w = self.weight[target]
            return (w * loss).sum() / w.sum().clamp(min=1e-12)
        return loss.mean()


def make_loss(cfg: TrainConfig, weights: torch.Tensor | None) -> nn.Module:
    if cfg.loss == "focal":
        return FocalLoss(cfg.focal_gamma, weights)
    return nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)


# ----------------------------------------------------------------------- schedule
def warmup_cosine(step: int, total: int, warmup: int, min_ratio: float) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to ``min_ratio``."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_amp(device: torch.device, enabled: bool) -> tuple[Any, Any, str]:
    """``(autocast factory, GradScaler or None, mode)``. bf16 needs no scaler;
    fp16 does. CPU and MPS train in fp32."""
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext, None, "fp32"
    if torch.cuda.is_bf16_supported():
        return functools.partial(torch.autocast, "cuda", dtype=torch.bfloat16), None, "bf16"
    scaler = torch.amp.GradScaler("cuda")
    return functools.partial(torch.autocast, "cuda", dtype=torch.float16), scaler, "fp16"


@dataclass
class TrainState:
    epoch: int = 0  # epochs completed
    global_step: int = 0  # optimizer steps taken
    best_metric: float = -math.inf
    best_epoch: int = 0
    bad_epochs: int = 0
    skipped_steps: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------- loops
def train_one_epoch(
    model: AccentClassifier,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss_fn: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    autocast: Any,
    scaler: Any,
    state: TrainState,
) -> dict[str, float]:
    """One pass over ``loader``. Gradients are cleared once per optimizer step
    (so accumulation is correct); a non-finite loss or gradient skips the step
    instead of corrupting the weights. Only Python floats are kept for logging."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_items, norms = 0.0, 0, []
    accum = cfg.grad_accum_steps
    num_batches = len(loader)
    for i, batch in enumerate(loader):
        x = batch["waveforms"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with autocast():
            logits = model(x, mask)["logits"]
        loss = loss_fn(logits.float(), y)
        if not torch.isfinite(loss):
            log.warning(f"Non-finite loss at step {state.global_step}; skipping batch")
            optimizer.zero_grad(set_to_none=True)
            state.skipped_steps += 1
            continue
        (scaler.scale(loss / accum) if scaler else loss / accum).backward()
        total_loss += float(loss.item()) * len(y)
        total_items += len(y)
        if (i + 1) % accum and i + 1 != num_batches:
            continue
        if scaler:
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        if scaler:
            scaler.step(optimizer)  # skips internally on inf/NaN gradients
            scaler.update()
        elif math.isfinite(grad_norm):
            optimizer.step()
        else:
            log.warning(f"Non-finite gradient norm at step {state.global_step}; skipping")
            state.skipped_steps += 1
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        state.global_step += 1
        if math.isfinite(grad_norm):
            norms.append(grad_norm)
        if cfg.log_every_n_steps and state.global_step % cfg.log_every_n_steps == 0:
            lrs = ", ".join(f"{g['name']} {g['lr']:.2e}" for g in optimizer.param_groups)
            log.info(
                f"step {state.global_step} | loss {total_loss / max(total_items, 1):.4f} "
                f"| grad-norm {grad_norm:.3f} | lr {lrs}"
            )
    return {
        "loss": total_loss / max(total_items, 1),
        "grad_norm": float(np.mean(norms)) if norms else float("nan"),
    }


def validate(
    model: AccentClassifier, records: list[Record], cfg: Config, device: torch.device
) -> dict[str, Any]:
    """Whole-utterance validation, sectioned exactly like inference (no temperature)."""
    pred = predict_records(model, records, cfg, device)
    probs = softmax_np(pred["logits"])
    metrics = classification_metrics(pred["labels"], probs.argmax(axis=1), cfg.data.accents)
    metrics["nll"] = negative_log_likelihood(probs, pred["labels"])
    return metrics


# ---------------------------------------------------------------------------- data
def prepare_data(cfg: Config) -> dict[str, list[Record]]:
    records = read_manifest(cfg.data.manifest, cfg.data.audio_root)
    log.info(
        f"Manifest: {len(records)} utterances from {len({r.speaker for r in records})} speakers"
    )
    records = verify_records(records, cfg.audio, decode=cfg.data.verify_audio)
    split_file = cfg.data.split_file or os.path.join(cfg.train.out_dir, "split.csv")
    if os.path.isfile(split_file):
        splits = load_split(split_file, records)
        log.info(f"Reusing split {split_file}")
    else:
        splits = speaker_disjoint_split(
            records,
            cfg.data.accents,
            cfg.data.val_frac,
            cfg.data.test_frac,
            cfg.seed,
            cfg.data.test_sources,
        )
        save_split(splits, split_file)
        log.info(f"Saved speaker-disjoint split to {split_file}")
    for name, recs in splits.items():
        per_accent = {a: sum(r.accent == a for r in recs) for a in cfg.data.accents}
        log.info(
            f"{name:5s}: {len(recs):5d} utterances, {len({r.speaker for r in recs}):4d} speakers "
            f"| {per_accent}"
        )
    if not splits["train"]:
        raise ValueError("The training split is empty")
    return splits


# ------------------------------------------------------------------------- training
def _check_resume_compatible(saved: dict[str, Any], cfg: Config) -> None:
    current = cfg.to_dict()
    for section in ("model", "audio"):
        if saved[section] != current[section]:
            diff = {
                k: (saved[section].get(k), v)
                for k, v in current[section].items()
                if saved[section].get(k) != v
            }
            raise ValueError(
                f"Cannot resume: '{section}' settings differ from the checkpoint: {diff}"
            )
    if saved["data"]["accents"] != current["data"]["accents"]:
        raise ValueError("Cannot resume: data.accents differs from the checkpoint")


def fit(
    cfg: Config, splits: dict[str, list[Record]], device: torch.device
) -> tuple[AccentClassifier, TrainState]:
    tc = cfg.train
    os.makedirs(tc.out_dir, exist_ok=True)
    model = build_model(cfg, len(cfg.data.accents)).to(device)
    log.info(f"Model: {cfg.model.backbone} backbone, {model_summary(model)}")
    weights = None
    if tc.class_weights == "auto":
        weights = class_weights(splits["train"], cfg.data.accents).to(device)
        log.info(f"Class weights: {[round(w, 3) for w in weights.tolist()]}")
    loss_fn = make_loss(tc, weights).to(device)

    loader = build_train_loader(cfg, splits["train"], cfg.seed)
    steps_per_epoch = math.ceil(len(loader) / tc.grad_accum_steps)
    total_steps = steps_per_epoch * tc.epochs
    warmup = int(tc.warmup_ratio * total_steps)
    optimizer = torch.optim.AdamW(model.param_groups(tc.lr, tc.backbone_lr, tc.weight_decay))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        functools.partial(
            warmup_cosine, total=total_steps, warmup=warmup, min_ratio=tc.min_lr_ratio
        ),
    )
    autocast, scaler, amp_mode = setup_amp(device, tc.amp)
    state = TrainState()

    if tc.resume:
        ckpt = load_checkpoint(tc.resume, map_location="cpu")
        if "optimizer" not in ckpt:
            raise ValueError(f"{tc.resume} is an inference checkpoint; resume from last.pt")
        _check_resume_compatible(ckpt["config"], cfg)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        state = TrainState(**ckpt["train_state"])
        set_rng_state(ckpt["rng"])
        log.info(f"Resumed from {tc.resume} after epoch {state.epoch} (step {state.global_step})")

    log.info(
        f"Training on {device} ({amp_mode}) for {tc.epochs} epochs, {steps_per_epoch} steps/epoch"
        f", warmup {warmup} steps"
    )
    val_records = splits["val"]
    if not val_records:
        log.warning("Validation split is empty: selecting on training loss, no calibration")
    last_epoch = tc.epochs if tc.stop_after_epochs is None else min(tc.epochs, tc.stop_after_epochs)
    try:
        for epoch in range(state.epoch, last_epoch):
            started = time.time()
            loader.generator.manual_seed(cfg.seed + epoch)  # same order whether resumed or not
            if isinstance(loader.batch_sampler, BucketBatchSampler):
                loader.batch_sampler.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, loader, optimizer, scheduler, loss_fn, device, tc, autocast, scaler, state
            )
            if val_records:
                val = validate(model, val_records, cfg, device)
                metric = val[tc.select_metric]
            else:
                val, metric = {}, -train_stats["loss"]
            state.epoch = epoch + 1
            row = {
                "epoch": state.epoch,
                "train_loss": train_stats["loss"],
                "grad_norm": train_stats["grad_norm"],
                **{k: val[k] for k in ("accuracy", "macro_f1", "uar", "nll") if k in val},
                "lr": [g["lr"] for g in optimizer.param_groups],
                "seconds": round(time.time() - started, 1),
            }
            state.history.append(row)
            improved = metric > state.best_metric + tc.min_delta
            if improved:
                state.best_metric, state.best_epoch, state.bad_epochs = metric, state.epoch, 0
                best = checkpoint_payload(model, cfg)
                best.update({"epoch": state.epoch, "temperature": 1.0, "val_metrics": val})
                save_checkpoint(best, os.path.join(tc.out_dir, "best.pt"))
            else:
                state.bad_epochs += 1
            log.info(
                f"epoch {state.epoch}/{tc.epochs} | train loss {row['train_loss']:.4f} | "
                + (
                    f"val acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f} "
                    f"UAR {val['uar']:.4f} | "
                    if val
                    else ""
                )
                + f"{'new best' if improved else f'no gain {state.bad_epochs}/{tc.patience}'}"
                f" | {row['seconds']}s"
            )
            last = checkpoint_payload(model, cfg)
            last.update(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "train_state": asdict(state),
                    "rng": get_rng_state(),
                }
            )
            save_checkpoint(last, os.path.join(tc.out_dir, "last.pt"))
            write_json(os.path.join(tc.out_dir, "history.json"), state.history)
            if state.bad_epochs >= tc.patience:
                log.info(f"Early stopping: no improvement for {tc.patience} epochs")
                break
    except KeyboardInterrupt:
        log.warning(f"Interrupted; resume with --resume {os.path.join(tc.out_dir, 'last.pt')}")
        raise
    return model, state


MIN_CALIBRATION_UTTERANCES = 30
TEMPERATURE_BOUNDS = (0.05, 20.0)


def calibrated_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Validation-fitted temperature, or 1.0 when the fit is not trustworthy:
    too few utterances, or an optimum at the search bound (which happens when
    the validation set is perfectly separable and NLL keeps falling as T -> 0)."""
    if len(labels) < MIN_CALIBRATION_UTTERANCES:
        log.warning(
            f"Only {len(labels)} validation utterances (< {MIN_CALIBRATION_UTTERANCES}): "
            "skipping temperature calibration (T = 1)"
        )
        return 1.0
    temperature = fit_temperature(logits, labels)
    lo, hi = TEMPERATURE_BOUNDS
    if temperature <= lo * 1.01 or temperature >= hi * 0.99:
        log.warning(f"Temperature fit hit its bound (T = {temperature:.3f}); keeping T = 1")
        return 1.0
    log.info(f"Fitted softmax temperature on validation: T = {temperature:.3f}")
    return temperature


def finalize(
    model: AccentClassifier, cfg: Config, splits: dict[str, list[Record]], device: torch.device
) -> dict[str, Any]:
    """Reload the best weights, fit the temperature on validation, evaluate test once."""
    best_path = os.path.join(cfg.train.out_dir, "best.pt")
    best = load_checkpoint(best_path, map_location="cpu")
    model.load_state_dict(best["model"])
    temperature = 1.0
    results: dict[str, Any] = {"best_epoch": best.get("epoch")}
    if splits["val"]:
        val_pred = predict_records(model, splits["val"], cfg, device)
        if cfg.train.calibrate:
            temperature = calibrated_temperature(val_pred["logits"], val_pred["labels"])
        results["val"] = accent_report(val_pred, cfg, temperature, cfg.train.out_dir, "val")
    best["temperature"] = temperature
    save_checkpoint(best, best_path)
    if splits["test"]:
        test_pred = predict_records(model, splits["test"], cfg, device)
        results["test"] = accent_report(test_pred, cfg, temperature, cfg.train.out_dir, "test")
        t = results["test"]
        log.info(
            f"TEST ({len(splits['test'])} utt): accuracy {t['accuracy']:.4f} | macro-F1 "
            f"{t['macro_f1']:.4f} | UAR {t['uar']:.4f} | ECE {t['ece']:.4f}"
        )
    results["temperature"] = temperature
    write_json(os.path.join(cfg.train.out_dir, "metrics.json"), results)
    return results


def run(cfg: Config) -> dict[str, Any]:
    """Train (or resume) and evaluate. Returns the final metrics."""
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    get_logger(log_file=os.path.join(cfg.train.out_dir, "logs", f"train_{timestamp()}.log"))
    for warning in cfg.validate():
        log.warning(warning)
    quiet_transformers()
    set_seed(cfg.seed, cfg.deterministic)
    device = get_device(cfg.device)
    cfg.save(os.path.join(cfg.train.out_dir, "config.yaml"))
    splits = prepare_data(cfg)
    model, state = fit(cfg, splits, device)
    finished = state.bad_epochs >= cfg.train.patience or state.epoch >= cfg.train.epochs
    if not finished:
        log.info(f"Stopped after epoch {state.epoch} (stop_after_epochs); resume to continue")
        return {"epoch": state.epoch, "history": state.history}
    return finalize(model, cfg, splits, device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train the accent classifier.")
    add_config_arguments(ap)
    args = ap.parse_args(argv)
    get_logger()
    try:
        cfg = config_from_args(args)
        run(cfg)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        log.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
