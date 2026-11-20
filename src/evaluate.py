"""Evaluation for both tasks.

Accent detection: accuracy, macro-F1, balanced accuracy (UAR), per-accent
precision/recall/F1, confusion matrix PNG, ECE and reliability diagram, with
a softmax temperature fitted on the validation split (never on test).

GOP: Pearson / Spearman against human scores when the manifest has a
``human_score`` column; AUC-ROC and equal error rate for mispronunciation
detection when it has ``phone_labels`` (0/1 per canonical phone, 1 =
mispronounced); per-phone score distributions in every case.

    python -m src.evaluate accent --checkpoint runs/default/best.pt --split test
    python -m src.evaluate gop --manifest data/gop.csv --out-dir runs/gop
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):  # executed as `python src/evaluate.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")

from .audio import load_audio, normalize_waveform, pad_batch, segment_bounds
from .config import Config, add_config_arguments, config_from_args
from .dataset import Record, build_eval_loader, load_split, read_manifest
from .utils import LOGGER_NAME, get_device, get_logger, jsonable, write_json

log = logging.getLogger(LOGGER_NAME)


# =========================================================== classification metrics
def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    """Accuracy, macro-F1, UAR (= balanced accuracy, mean recall over classes
    present in ``y_true``) and per-class precision / recall / F1 / support."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, len(labels))
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "uar": float(recall[present].mean()) if present.any() else 0.0,
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "num_samples": len(y_true),
    }


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def negative_log_likelihood(probs: np.ndarray, y_true: Sequence[int]) -> float:
    p = probs[np.arange(len(y_true)), np.asarray(y_true)]
    return float(-np.log(np.clip(p, 1e-12, 1.0)).mean())


def expected_calibration_error(
    probs: np.ndarray, y_true: Sequence[int], n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """ECE with equal-width confidence bins; the last bin includes 1.0 so no
    prediction falls through the cracks."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, bins = 0.0, []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        in_bin = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if not in_bin.any():
            continue
        acc, avg_conf, frac = correct[in_bin].mean(), conf[in_bin].mean(), in_bin.mean()
        ece += frac * abs(acc - avg_conf)
        bins.append(
            {
                "lo": lo,
                "hi": hi,
                "accuracy": acc,
                "confidence": avg_conf,
                "count": int(in_bin.sum()),
            }
        )
    return float(ece), bins


def fit_temperature(logits: np.ndarray, y_true: Sequence[int], max_iter: int = 100) -> float:
    """Single softmax temperature minimising validation NLL (Guo et al., 2017)."""
    x = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(y_true), dtype=torch.long)
    if len(y) == 0:
        return 1.0
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


# ==================================================================== GOP metrics
def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank), 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    return pearson(_ranks(np.asarray(x, dtype=np.float64)), _ranks(np.asarray(y, dtype=np.float64)))


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC via the Mann-Whitney statistic (ties count half). ``labels``: 1 = positive."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def equal_error_rate(scores: Sequence[float], labels: Sequence[int]) -> tuple[float, float]:
    """``(EER, threshold)``: the operating point where false accepts equal false
    rejects, linearly interpolated between neighbouring thresholds."""
    s, y = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    if y.all() or (~y).all():
        return float("nan"), float("nan")
    thresholds = np.unique(s)[::-1]
    fpr = np.array([((s >= t) & ~y).sum() / (~y).sum() for t in thresholds])
    fnr = np.array([((s < t) & y).sum() / y.sum() for t in thresholds])
    diff = fpr - fnr
    k = int(np.argmax(diff >= 0)) if (diff >= 0).any() else len(diff) - 1
    if k == 0 or diff[k] == 0:
        return float((fpr[k] + fnr[k]) / 2), float(thresholds[k])
    a, b = diff[k - 1], diff[k]  # a < 0 <= b: interpolate the crossing
    w = -a / (b - a)
    eer = (1 - w) * (fpr[k - 1] + fnr[k - 1]) / 2 + w * (fpr[k] + fnr[k]) / 2
    return float(eer), float((1 - w) * thresholds[k - 1] + w * thresholds[k])


# ========================================================================= plots
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(cm: np.ndarray, labels: Sequence[str], path: str) -> str:
    """Row-normalised confusion matrix with raw counts in each cell."""
    plt = _plt()
    cm = np.asarray(cm, dtype=np.float64)
    norm = np.divide(
        cm,
        cm.sum(1, keepdims=True),
        out=np.zeros_like(cm),
        where=cm.sum(1, keepdims=True) > 0,
    )
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 1.0 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted accent")
    ax.set_ylabel("True accent")
    ax.set_title("Confusion matrix (row-normalised, counts in brackets)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_reliability(bins: list[dict[str, float]], ece: float, path: str) -> str:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    widths = [b["hi"] - b["lo"] for b in bins]
    ax.bar(
        centers,
        [b["accuracy"] for b in bins],
        width=widths,
        edgecolor="white",
        alpha=0.8,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability (ECE = {ece:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_phone_distributions(scores: dict[str, list[float]], path: str, top_n: int = 25) -> str:
    """Box plot of GOP scores for the most frequent phones, lowest median first."""
    plt = _plt()
    phones = sorted(scores, key=lambda p: -len(scores[p]))[:top_n]
    phones.sort(key=lambda p: float(np.median(scores[p])))
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(phones) + 2), 4.5))
    ax.boxplot([scores[p] for p in phones], showfliers=False)
    ax.set_xticks(range(1, len(phones) + 1), [f"/{p}/\n{len(scores[p])}" for p in phones])
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("GOP score")
    ax.set_title("Per-phone GOP score distribution (n under each phone)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ===================================================================== inference
def segment_clip(wave: np.ndarray, cfg: Config) -> list[np.ndarray]:
    """Split a (trimmed) clip into overlapping sections, each normalised on its own,
    exactly like a training crop."""
    sr = cfg.audio.sample_rate
    bounds = segment_bounds(len(wave), sr, cfg.eval.segment_seconds, cfg.eval.segment_hop_seconds)
    segments = [wave[s:e] for s, e in bounds]
    if cfg.audio.normalize:
        segments = [normalize_waveform(s) for s in segments]
    return segments


@torch.inference_mode()
def predict_waves(
    model: torch.nn.Module,
    waves: Sequence[np.ndarray],
    cfg: Config,
    device: Any,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    """Accent prediction for whole clips: sections are batched across clips,
    section logits are averaged per clip, and ``agreement`` is the fraction of
    sections whose own top class matches the clip-level prediction."""
    model.eval()
    owners, segments = [], []
    for i, wave in enumerate(waves):
        for seg in segment_clip(np.asarray(wave, dtype=np.float32), cfg):
            owners.append(i)
            segments.append(seg)
    logits = []
    bs = cfg.eval.batch_size
    for s in range(0, len(segments), bs):
        batch = pad_batch(segments[s : s + bs])
        out = model(batch["waveforms"].to(device), batch["attention_mask"].to(device))
        logits.append(out["logits"].float().cpu())
    seg_logits = torch.cat(logits).numpy() if logits else np.zeros((0, 0))
    owners_arr = np.asarray(owners)
    results = []
    for i in range(len(waves)):
        rows = seg_logits[owners_arr == i]
        clip_logits = rows.mean(axis=0)
        probs = softmax_np(clip_logits, temperature)
        pred = int(probs.argmax())
        agreement = float((rows.argmax(axis=1) == pred).mean())
        verdict = "contested" if agreement < cfg.eval.contested_agreement else "consistent"
        results.append(
            {
                "logits": clip_logits,
                "probs": probs,
                "pred": pred,
                "segments": len(rows),
                "agreement": agreement,
                "verdict": verdict,
            }
        )
    return results


def predict_records(
    model: torch.nn.Module,
    records: Sequence[Record],
    cfg: Config,
    device: Any,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Run ``predict_waves`` over manifest records (audio decoded by DataLoader workers)."""
    logits, labels, agreement, utt_ids = [], [], [], []
    for items in build_eval_loader(cfg, records):
        results = predict_waves(model, [it["waveform"] for it in items], cfg, device, temperature)
        for it, res in zip(items, results, strict=True):
            logits.append(res["logits"])
            labels.append(it["label"])
            agreement.append(res["agreement"])
            utt_ids.append(it["utt_id"])
    num_classes = len(cfg.data.accents)
    return {
        "logits": np.stack(logits) if logits else np.zeros((0, num_classes)),
        "labels": np.asarray(labels, dtype=np.int64),
        "agreement": np.asarray(agreement),
        "utt_ids": utt_ids,
    }


def accent_report(
    pred: dict[str, Any],
    cfg: Config,
    temperature: float = 1.0,
    out_dir: str | None = None,
    prefix: str = "test",
) -> dict[str, Any]:
    """Metrics (+ optional confusion-matrix / reliability PNGs) from ``predict_records``."""
    labels = list(cfg.data.accents)
    probs = (
        softmax_np(pred["logits"], temperature)
        if len(pred["labels"])
        else np.zeros((0, len(labels)))
    )
    y = pred["labels"]
    metrics = classification_metrics(y, probs.argmax(axis=1), labels) if len(y) else {}
    if len(y):
        ece, bins = expected_calibration_error(probs, y, cfg.eval.ece_bins)
        ece_raw, _ = expected_calibration_error(softmax_np(pred["logits"]), y, cfg.eval.ece_bins)
        metrics.update(
            {
                "nll": negative_log_likelihood(probs, y),
                "ece": ece,
                "ece_uncalibrated": ece_raw,
                "temperature": temperature,
                "contested_fraction": float(
                    (pred["agreement"] < cfg.eval.contested_agreement).mean()
                ),
            }
        )
        if out_dir:
            metrics["confusion_matrix_png"] = plot_confusion_matrix(
                np.asarray(metrics["confusion_matrix"]),
                labels,
                os.path.join(out_dir, f"{prefix}_confusion_matrix.png"),
            )
            metrics["reliability_png"] = plot_reliability(
                bins, ece, os.path.join(out_dir, f"{prefix}_reliability.png")
            )
    return metrics


# ======================================================================= GOP eval
def read_gop_manifest(path: str) -> list[dict[str, str]]:
    """CSV with ``path, text`` and optional ``human_score``, ``phone_labels``, ``utt_id``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GOP manifest not found: {path}")
    root = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [
            {k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)
        ]
    if rows and not {"path", "text"} <= set(rows[0]):
        raise ValueError(f"{path} needs columns 'path' and 'text'")
    for r in rows:
        if not os.path.isabs(r["path"]):
            r["path"] = os.path.normpath(os.path.join(root, r["path"]))
    return rows


def evaluate_gop(
    rows: Sequence[dict[str, str]], recognizer: Any, g2p: Any, cfg: Config, out_dir: str
) -> dict[str, Any]:
    from .gop import assess_pronunciation

    phone_scores: dict[str, list[float]] = defaultdict(list)
    overall, human, det_scores, det_labels = [], [], [], []
    errors, label_mismatch = 0, 0
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gop_results.jsonl"), "w", encoding="utf-8") as out:
        for row in rows:
            try:
                wave, _ = load_audio(row["path"], recognizer.sample_rate)
                report = assess_pronunciation(wave, row["text"], recognizer, g2p, cfg.gop)
            except (OSError, ValueError) as exc:
                report = {
                    "transcript": row["text"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report["path"] = row["path"]
            out.write(json.dumps(jsonable(report), ensure_ascii=False) + "\n")
            if report.get("error"):
                errors += 1
                log.warning(f"{row['path']}: {report['error']}")
                continue
            phones = [p for w in report["words"] for p in w["phones"]]
            for p in phones:
                phone_scores[p["phone"]].append(p["score"])
            if row.get("human_score"):
                overall.append(report["overall_score"])
                human.append(float(row["human_score"]))
            if row.get("phone_labels"):
                labels = [int(v) for v in row["phone_labels"].split()]
                if len(labels) != len(phones):
                    label_mismatch += 1
                    continue
                det_scores += [1.0 - p["score"] for p in phones]  # high = likely mispronounced
                det_labels += labels
    metrics: dict[str, Any] = {"utterances": len(rows), "errors": errors}
    if human:
        metrics.update(
            {
                "pearson": pearson(overall, human),
                "spearman": spearman(overall, human),
                "num_human_scored": len(human),
            }
        )
    if det_labels:
        eer, thr = equal_error_rate(det_scores, det_labels)
        metrics.update(
            {
                "auc_roc": roc_auc(det_scores, det_labels),
                "eer": eer,
                "eer_score_threshold": 1.0 - thr,
                "num_labelled_phones": len(det_labels),
                "label_length_mismatches": label_mismatch,
            }
        )
    metrics["phone_stats"] = {
        p: {"mean": float(np.mean(v)), "std": float(np.std(v)), "count": len(v)}
        for p, v in sorted(phone_scores.items())
    }
    if phone_scores:
        metrics["phone_distribution_png"] = plot_phone_distributions(
            phone_scores, os.path.join(out_dir, "gop_phone_distributions.png")
        )
    write_json(os.path.join(out_dir, "gop_metrics.json"), metrics)
    return metrics


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate the accent classifier or GOP scoring.")
    sub = ap.add_subparsers(dest="task", required=True)
    acc = sub.add_parser("accent", help="accent metrics on a split of the training manifest")
    acc.add_argument("--checkpoint", default="runs/default/best.pt")
    acc.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    acc.add_argument("--split-file", default=None, help="default: split.csv next to the checkpoint")
    acc.add_argument(
        "--manifest",
        default=None,
        help="default: data.manifest stored in the checkpoint",
    )
    acc.add_argument("--out-dir", default=None, help="default: the checkpoint's folder")
    gop = sub.add_parser(
        "gop",
        help="GOP metrics on a manifest with path,text[,human_score,phone_labels]",
    )
    gop.add_argument("--manifest", required=True)
    gop.add_argument("--out-dir", default="runs/gop_eval")
    for p in (acc, gop):
        add_config_arguments(
            p,
            priority=("eval", "gop", "audio", "model", "data", "train", "augment"),
            exclude={"manifest", "split_file", "out_dir", "checkpoint"},
        )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    get_logger()
    try:
        if args.task == "accent":
            return _main_accent(args)
        return _main_gop(args)
    except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
        log.error(str(exc))
        return 1


def _main_accent(args: argparse.Namespace) -> int:
    from .model import load_accent_model

    cli_cfg = config_from_args(args)
    device = get_device(cli_cfg.device)
    model, cfg, ckpt = load_accent_model(args.checkpoint, device)
    cfg.eval, cfg.data.num_workers = (
        cli_cfg.eval,
        cli_cfg.data.num_workers,
    )  # eval knobs from CLI
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    records = read_manifest(args.manifest or cfg.data.manifest, cfg.data.audio_root)
    if args.split == "all":
        chosen = records
    else:
        split_file = args.split_file or os.path.join(ckpt_dir, "split.csv")
        chosen = load_split(split_file, records)[args.split]
    if not chosen:
        raise ValueError(f"Split '{args.split}' is empty")
    temperature = float(ckpt.get("temperature", 1.0))
    log.info(f"Evaluating {len(chosen)} utterances ({args.split}) on {device}, T={temperature:.3f}")
    pred = predict_records(model, chosen, cfg, device, temperature)
    out_dir = args.out_dir or ckpt_dir
    metrics = accent_report(pred, cfg, temperature, out_dir, prefix=args.split)
    write_json(os.path.join(out_dir, f"{args.split}_metrics.json"), metrics)
    log.info(
        f"{args.split}: accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} "
        f"| UAR {metrics['uar']:.4f} | ECE {metrics['ece']:.4f} -> {out_dir}"
    )
    return 0


def _main_gop(args: argparse.Namespace) -> int:
    from .aligner import PhonemeRecognizer, build_g2p
    from .utils import quiet_transformers

    cfg = config_from_args(args)
    quiet_transformers()
    device = get_device(cfg.device)
    recognizer = PhonemeRecognizer(cfg.gop.phoneme_model, device)
    g2p = build_g2p(cfg.gop.g2p, cfg.gop.language, cfg.gop.lexicon)
    rows = read_gop_manifest(args.manifest)
    metrics = evaluate_gop(rows, recognizer, g2p, cfg, args.out_dir)
    summary = {k: v for k, v in metrics.items() if k != "phone_stats"}
    log.info(f"GOP evaluation: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
