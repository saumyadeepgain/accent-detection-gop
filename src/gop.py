"""Goodness of Pronunciation (GOP) from CTC phoneme posteriors.

Inputs are always log-probabilities (``log_softmax``), never raw logits. For
each phone, over the frames where the CTC alignment emits it:

    LPP       = mean_t  log P(target | o_t)
    LPR       = mean_t [log P(target | o_t) - max_{q != target} log P(q | o_t)]
    rival(r)  = mean_t [log P(target | o_t) - log P(r | o_t)]
    score     = sigmoid((LPR - center) / scale)        in (0, 1), 1 = native-like
    flagged   = score < flag_threshold

With ``exclude_blank`` the posteriors are first renormalised over the non-blank
tokens (a stable ``logsumexp``), so the CTC blank, which dominates most
frames, can neither be the competitor nor dilute the target probability.
Scores are aggregated per word (mean or min) and per utterance (0-100).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from .aligner import (
    AlignmentError,
    Span,
    Word,
    forced_align,
    normalize_symbol,
    phonemize,
)
from .config import GOPConfig

LOG_FLOOR = -50.0  # log-probabilities are clamped here: exp(-50) ~ 2e-22, keeps every mean finite


@dataclass
class PhoneScore:
    phone: str
    word_index: int
    start: float  # seconds (frame grid)
    end: float
    frames: int  # frames the score is computed from
    lpp: float
    lpr: float
    score: float
    flagged: bool
    heard_as: str  # most probable non-blank token over the span
    rivals: dict[str, float] = field(default_factory=dict)
    zero_length: bool = False  # the span had no frames; the nearest frame was used


def prepare_log_probs(log_probs: torch.Tensor, blank_id: int, exclude_blank: bool) -> torch.Tensor:
    """Clamp and (optionally) renormalise frame log-probabilities over non-blank tokens."""
    lp = torch.nan_to_num(log_probs.detach().float(), nan=LOG_FLOOR, neginf=LOG_FLOOR)
    lp = lp.clamp(min=LOG_FLOOR)
    if exclude_blank:
        keep = torch.ones(lp.shape[-1], dtype=torch.bool)
        keep[blank_id] = False
        lp = lp - torch.logsumexp(lp[:, keep], dim=-1, keepdim=True)  # log(1 - P(blank)), stable
        lp[:, blank_id] = LOG_FLOOR
        lp = lp.clamp(min=LOG_FLOOR)
    return lp


def score_span(
    lp: torch.Tensor,
    start: int,
    end: int,
    target_id: int,
    blank_id: int,
    rival_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """GOP statistics of one phone over frames ``[start, end)`` of prepared log-probs.

    A zero-length span (possible with external aligners, never with CTC
    alignment) falls back to the nearest frame instead of dividing by zero."""
    num_frames = lp.shape[0]
    if num_frames == 0:
        raise ValueError("score_span needs at least one frame of log-probabilities")
    zero_length = end <= start
    if zero_length:
        start = min(max(start, 0), num_frames - 1)
        end = start + 1
    frames = lp[start:end]
    target = frames[:, target_id]
    others = frames.clone()
    others[:, target_id] = -math.inf
    others[:, blank_id] = -math.inf
    competitor = others.max(dim=-1).values.clamp(min=LOG_FLOOR)  # finite even for tiny vocabs
    mean_lp = frames.mean(dim=0)
    mean_lp[blank_id] = -math.inf
    out = {
        "lpp": float(target.mean()),
        "lpr": float((target - competitor).mean()),
        "heard_as_id": int(mean_lp.argmax()),
        "frames": int(end - start),
        "zero_length": zero_length,
        "rivals": {
            name: float((target - frames[:, rid]).mean()) for name, rid in (rival_ids or {}).items()
        },
    }
    return out


def to_score(lpr: float, center: float = 0.0, scale: float = 2.0) -> float:
    """Logistic map of LPR onto (0, 1); 0.5 means the target ties its best competitor."""
    z = (lpr - center) / scale
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def rival_ids_for(
    symbol: str, rivals: dict[str, list[str]], vocab: dict[str, int]
) -> dict[str, int]:
    """Configured rivals of ``symbol`` that exist in the vocabulary (target excluded)."""
    out: dict[str, int] = {}
    for key, names in rivals.items():
        if normalize_symbol(key, vocab) != symbol:
            continue
        for name in names:
            sym = normalize_symbol(name, vocab)
            if sym is not None and sym != symbol:
                out[sym] = vocab[sym]
    return out


def score_alignment(
    log_probs: torch.Tensor,
    words: Sequence[Word],
    spans: Sequence[Span],
    vocab: dict[str, int],
    blank_id: int,
    frame_shift: float,
    cfg: GOPConfig,
) -> list[PhoneScore]:
    """Phone-level GOP for an alignment of ``words`` (``spans`` in phone order)."""
    id_to_token = {i: t for t, i in vocab.items()}
    lp = prepare_log_probs(log_probs, blank_id, cfg.exclude_blank)
    flat = [(wi, p) for wi, w in enumerate(words) for p in w.phones]
    if len(flat) != len(spans):
        raise ValueError(f"{len(spans)} spans for {len(flat)} phones")
    starts, ends = _display_bounds(spans, [wi for wi, _ in flat])
    scores = []
    for (wi, phone), span, s0, s1 in zip(flat, spans, starts, ends, strict=True):
        stats = score_span(
            lp,
            span.start,
            span.end,
            phone.token_id,
            blank_id,
            rival_ids_for(phone.symbol, cfg.rivals, vocab),
        )
        score = to_score(stats["lpr"], cfg.score_center, cfg.score_scale)
        scores.append(
            PhoneScore(
                phone=phone.symbol,
                word_index=wi,
                start=round(s0 * frame_shift, 3),
                end=round(s1 * frame_shift, 3),
                frames=stats["frames"],
                lpp=round(stats["lpp"], 4),
                lpr=round(stats["lpr"], 4),
                score=round(score, 4),
                flagged=score < cfg.flag_threshold,
                heard_as=id_to_token.get(stats["heard_as_id"], "?"),
                rivals={k: round(v, 4) for k, v in stats["rivals"].items()},
                zero_length=stats["zero_length"],
            )
        )
    return scores


def _display_bounds(spans: Sequence[Span], word_ids: Sequence[int]) -> tuple[list, list]:
    """Readable phone boundaries: CTC emits a phone on only 1-3 frames, so blank
    frames *inside* a word are split between neighbouring phones. Gaps between
    words (pauses) are not assigned to either word."""
    starts = [s.start for s in spans]
    ends = [s.end for s in spans]
    for k in range(len(spans) - 1):
        if word_ids[k] == word_ids[k + 1]:
            # spans[k].end <= mid <= spans[k + 1].start, so no phone loses its own frames
            mid = (spans[k].end + spans[k + 1].start) // 2
            ends[k] = starts[k + 1] = mid
    return starts, ends


def summarize(
    words: Sequence[Word], phones: Sequence[PhoneScore], word_aggregate: str = "mean"
) -> dict[str, Any]:
    """Word- and utterance-level report from phone scores."""
    by_word: dict[int, list[PhoneScore]] = {}
    for p in phones:
        by_word.setdefault(p.word_index, []).append(p)
    word_rows = []
    for wi, word in enumerate(words):
        ps = by_word.get(wi, [])
        values = [p.score for p in ps]
        if values:
            agg = min(values) if word_aggregate == "min" else float(np.mean(values))
            score: float | None = round(agg, 4)
        else:
            score = None
        word_rows.append(
            {
                "word": word.text,
                "start": ps[0].start if ps else None,
                "end": ps[-1].end if ps else None,
                "score": score,
                "flagged": any(p.flagged for p in ps),
                "phones": [_phone_row(p) for p in ps],
                "unscored_phones": list(word.unknown),
            }
        )
    flagged = [
        {
            "word": words[p.word_index].text,
            "phone": p.phone,
            "start": p.start,
            "end": p.end,
            "score": p.score,
            "heard_as": p.heard_as,
        }
        for p in phones
        if p.flagged
    ]
    overall = round(100.0 * float(np.mean([p.score for p in phones])), 1) if phones else None
    return {
        "overall_score": overall,
        "num_phones": len(phones),
        "num_flagged": len(flagged),
        "words": word_rows,
        "flagged": flagged,
    }


def _phone_row(p: PhoneScore) -> dict[str, Any]:
    row = asdict(p)
    row.pop("word_index")
    return row


def assess_pronunciation(
    wave: np.ndarray,
    transcript: str,
    recognizer: Any,
    g2p: Callable[[str], list[tuple[str, list[str]]]],
    cfg: GOPConfig,
) -> dict[str, Any]:
    """Full GOP pipeline for one utterance: G2P -> posteriors -> alignment -> scores.

    ``recognizer`` provides ``vocab``, ``blank_id``, ``frame_shift`` and
    ``log_probs(wave)``. Problems that concern this utterance only (unknown
    words, audio too short, nothing scorable) are returned in ``error``
    instead of raised, so a batch run never dies on one bad file."""
    report: dict[str, Any] = {"transcript": transcript, "error": None}
    try:
        words = phonemize(transcript, g2p, recognizer.vocab, cfg.unknown_phones)
    except KeyError as exc:  # lexicon OOV
        report["error"] = str(exc).strip("'\"")
        return report
    unknown = sorted({u for w in words for u in w.unknown})
    report["unknown_phones"] = unknown
    targets = [p.token_id for w in words for p in w.phones]
    if not targets:
        report["error"] = "No phones to score (empty transcript or all symbols unknown)"
        return report
    try:
        log_probs = recognizer.log_probs(wave)
        spans = forced_align(log_probs, targets, recognizer.blank_id)
    except AlignmentError as exc:
        report["error"] = str(exc)
        return report
    phones = score_alignment(
        log_probs,
        words,
        spans,
        recognizer.vocab,
        recognizer.blank_id,
        recognizer.frame_shift,
        cfg,
    )
    report.update(summarize(words, phones, cfg.word_aggregate))
    duration = len(wave) / float(getattr(recognizer, "sample_rate", 16000))
    report["alignment"] = {
        "frames": int(log_probs.shape[0]),
        "frame_shift": recognizer.frame_shift,
        "phones_per_second": round(len(targets) / max(duration, 1e-9), 2),
    }
    return report
