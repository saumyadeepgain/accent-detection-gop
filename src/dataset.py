"""Manifest, speaker-disjoint splits, accent-safe augmentation and data loaders.

Manifest (CSV, UTF-8): ``path, speaker_id, accent`` plus optional ``source``,
``utt_id``, ``text``, ``gender``, ``human_score``, ``phone_labels``.

Splits are made over **speakers**, never utterances, separately for each
accent; speaker IDs are namespaced by corpus (``NISP/spk01`` and
``Svarah/spk01`` are different people), and ``assert_speaker_disjoint`` runs
on every split, including one reloaded from disk. ``test_sources`` holds out
whole corpora for a cross-corpus test, the honest check that the model did
not learn the microphone instead of the accent.

Augmentation is deliberately accent-safe: additive noise, mild gain, and
SpecAugment masking. No pitch shifting or speed warping, which move vowel
formants and change speaking rhythm, i.e. the accent cues themselves.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from .audio import (
    AudioError,
    audio_info,
    collect_audio_files,
    load_audio,
    normalize_waveform,
    pad_batch,
    random_crop,
    rms_dbfs,
    trim_silence,
)
from .config import AudioConfig, AugmentConfig, Config
from .utils import LOGGER_NAME, make_generator, seed_worker

REQUIRED_COLUMNS = ("path", "speaker_id", "accent")
SPLITS = ("train", "val", "test")


class LeakageError(ValueError):
    """A speaker appears in more than one split."""


@dataclass
class Record:
    path: str
    speaker: str  # namespaced by corpus when a ``source`` column exists
    accent: str
    source: str = ""
    utt_id: str = ""
    duration: float = 0.0  # seconds (after trimming, when audio was verified)
    text: str = ""
    extra: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------------ manifest
def read_manifest(path: str, audio_root: str | None = None) -> list[Record]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Manifest not found: {path}. Create a CSV with columns {', '.join(REQUIRED_COLUMNS)} "
            "(see README: Dataset setup)."
        )
    root = audio_root or os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8-sig", newline="") as f:  # utf-8-sig: tolerate Excel's BOM
        reader = csv.DictReader(f)
        columns = [c.strip() for c in reader.fieldnames or []]
        missing = [c for c in REQUIRED_COLUMNS if c not in columns]
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
        records = []
        for line, row in enumerate(reader, start=2):
            row = {k.strip(): (v or "").strip() for k, v in row.items() if k}
            if not row["path"] or not row["speaker_id"] or not row["accent"]:
                raise ValueError(f"{path}:{line}: path, speaker_id and accent must be non-empty")
            source = row.get("source", "")
            audio = row["path"] if os.path.isabs(row["path"]) else os.path.join(root, row["path"])
            known = {"path", "speaker_id", "accent", "source", "utt_id", "text"}
            records.append(
                Record(
                    path=os.path.normpath(audio),
                    speaker=f"{source}/{row['speaker_id']}" if source else row["speaker_id"],
                    accent=row["accent"],
                    source=source,
                    utt_id=row.get("utt_id") or row["path"],
                    text=row.get("text", ""),
                    extra={k: v for k, v in row.items() if k not in known},
                )
            )
    ids = Counter(r.utt_id for r in records)
    dupes = [u for u, n in ids.items() if n > 1]
    if dupes:
        raise ValueError(f"{path}: duplicate utt_id/path values, e.g. {dupes[:3]}")
    return records


def verify_records(
    records: Sequence[Record], audio_cfg: AudioConfig, decode: bool = True
) -> list[Record]:
    """Drop missing, unreadable, silent and too-short clips (logged with reasons).

    ``decode=True`` decodes every file once (catches corrupted audio and
    silence); ``decode=False`` only reads headers (fast, catches missing files)."""
    log = logging.getLogger(LOGGER_NAME)
    kept, reasons = [], Counter()
    for i, rec in enumerate(records, 1):
        try:
            if not os.path.isfile(rec.path):
                reasons["missing file"] += 1
                continue
            if decode:
                wave, _ = load_audio(rec.path, audio_cfg.sample_rate)
                if audio_cfg.trim_silence:
                    wave, _ = trim_silence(wave, audio_cfg.sample_rate, audio_cfg.trim_top_db)
                if rms_dbfs(wave) < audio_cfg.silence_rms_db:
                    reasons["silent"] += 1
                    continue
                duration = wave.shape[-1] / audio_cfg.sample_rate
            else:
                duration = audio_info(rec.path).duration
            if duration < audio_cfg.min_duration:
                reasons[f"shorter than {audio_cfg.min_duration}s"] += 1
                continue
            rec.duration = duration
            kept.append(rec)
        except (AudioError, OSError, RuntimeError) as exc:
            reasons["unreadable"] += 1
            log.warning(f"Skipping {rec.path}: {exc}")
        if decode and i % 500 == 0:
            log.info(f"Verified {i}/{len(records)} files")
    if reasons:
        detail = ", ".join(f"{n} {why}" for why, n in reasons.items())
        log.warning(f"Dropped {sum(reasons.values())} of {len(records)} clips: {detail}")
    if not kept:
        raise ValueError("No usable audio left after verification")
    return kept


# -------------------------------------------------------------------------- splits
def speaker_disjoint_split(
    records: Sequence[Record],
    accents: Sequence[str],
    val_frac: float,
    test_frac: float,
    seed: int,
    test_sources: Sequence[str] = (),
) -> dict[str, list[Record]]:
    """Allocate whole speakers to train/val/test, stratified by accent.

    Records from ``test_sources`` form the test set on their own (cross-corpus
    evaluation); otherwise ``test_frac`` of each accent's speakers is held out."""
    log = logging.getLogger(LOGGER_NAME)
    unknown = sorted({r.accent for r in records} - set(accents))
    if unknown:
        raise ValueError(f"Accent labels not in data.accents: {unknown}")
    held_out = set(test_sources)
    splits: dict[str, list[Record]] = {s: [] for s in SPLITS}
    by_speaker: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.source in held_out:
            splits["test"].append(r)
        else:
            by_speaker[r.speaker].append(r)
    if held_out and not splits["test"]:
        raise ValueError(f"data.test_sources={sorted(held_out)} matched no records")

    speaker_accent: dict[str, str] = {}
    for spk, recs in by_speaker.items():
        labels = {r.accent for r in recs}
        if len(labels) > 1:
            raise ValueError(f"Speaker {spk} has several accent labels {sorted(labels)}")
        speaker_accent[spk] = labels.pop()

    rng = np.random.default_rng(seed)
    for accent in accents:
        speakers = sorted(s for s, a in speaker_accent.items() if a == accent)
        if not speakers:
            log.warning(f"No training speakers for accent '{accent}'")
            continue
        rng.shuffle(speakers)
        n = len(speakers)
        n_test = 0 if held_out else int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        while n_test + n_val > n - 1 and (n_test or n_val):  # keep >= 1 training speaker
            if n_val >= n_test and n_val:
                n_val -= 1
            else:
                n_test -= 1
        if (val_frac and not n_val) or (test_frac and not held_out and not n_test):
            log.warning(f"Accent '{accent}' has only {n} speaker(s): val/test may lack it")
        groups = {
            "test": speakers[:n_test],
            "val": speakers[n_test : n_test + n_val],
            "train": speakers[n_test + n_val :],
        }
        for split, spks in groups.items():
            for spk in spks:
                splits[split].extend(by_speaker[spk])
    assert_speaker_disjoint(splits)
    return splits


def assert_speaker_disjoint(splits: dict[str, Sequence[Record]]) -> None:
    speakers = {name: {r.speaker for r in recs} for name, recs in splits.items()}
    names = list(speakers)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = speakers[a] & speakers[b]
            if shared:
                raise LeakageError(
                    f"{len(shared)} speaker(s) in both '{a}' and '{b}', e.g. {sorted(shared)[:5]}"
                )


def save_split(splits: dict[str, Sequence[Record]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utt_id", "speaker", "accent", "source", "split"])
        for split, recs in splits.items():
            for r in recs:
                writer.writerow([r.utt_id, r.speaker, r.accent, r.source, split])


def load_split(path: str, records: Sequence[Record]) -> dict[str, list[Record]]:
    """Re-apply a saved split to the current manifest (and re-check leakage)."""
    with open(path, encoding="utf-8", newline="") as f:
        assignment = {row["utt_id"]: row["split"] for row in csv.DictReader(f)}
    splits: dict[str, list[Record]] = {s: [] for s in SPLITS}
    missing = []
    for r in records:
        if r.utt_id in assignment:
            splits[assignment[r.utt_id]].append(r)
        else:
            missing.append(r.utt_id)
    if missing:
        raise ValueError(
            f"{len(missing)} manifest entries are not in {path} (e.g. {missing[:3]}); "
            "delete the split file to re-split"
        )
    assert_speaker_disjoint(splits)
    return splits


def class_weights(records: Sequence[Record], accents: Sequence[str]) -> torch.Tensor:
    """Inverse-frequency weights ``N / (C * count_c)``; absent classes get 0."""
    counts = Counter(r.accent for r in records)
    total, num = len(records), len(accents)
    return torch.tensor(
        [total / (num * counts[a]) if counts[a] else 0.0 for a in accents],
        dtype=torch.float32,
    )


# -------------------------------------------------------------------- augmentation
class WaveformAugment:
    """Additive noise at a random SNR and a random gain (both optional).

    Noise comes from ``noise_dir`` recordings when given (e.g. MUSAN), otherwise
    it is synthetic coloured noise (white to pink). Silent clips are left alone:
    an SNR is undefined without signal power."""

    def __init__(self, cfg: AugmentConfig, sample_rate: int):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.noise_files = collect_audio_files([cfg.noise_dir], True) if cfg.noise_dir else []
        self._cache: dict[str, np.ndarray] = {}

    def _noise(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.noise_files:
            path = self.noise_files[int(rng.integers(len(self.noise_files)))]
            if path not in self._cache:
                self._cache[path] = load_audio(path, self.sample_rate)[0]
            noise = self._cache[path]
            if noise.shape[0] < n:
                noise = np.tile(noise, n // max(noise.shape[0], 1) + 1)
            start = int(rng.integers(0, noise.shape[0] - n + 1))
            return noise[start : start + n].astype(np.float32)
        white = rng.standard_normal(n)
        alpha = rng.uniform(0.0, 1.0)  # 0 = white, 1 = pink
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n)
        freqs[0] = freqs[1] if n > 1 else 1.0
        return np.fft.irfft(spectrum / freqs ** (alpha / 2), n).astype(np.float32)

    def __call__(self, wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.cfg
        wave = np.asarray(wave, dtype=np.float32)
        if cfg.noise_prob > 0 and rng.random() < cfg.noise_prob:
            power = float(np.mean(np.square(wave, dtype=np.float64)))
            if power > 1e-10:
                noise = self._noise(wave.shape[0], rng)
                noise_power = float(np.mean(np.square(noise, dtype=np.float64))) + 1e-12
                snr = rng.uniform(*cfg.noise_snr_db)
                wave = wave + noise * np.sqrt(power / (noise_power * 10 ** (snr / 10)))
        if cfg.gain_prob > 0 and rng.random() < cfg.gain_prob:
            wave = wave * 10 ** (rng.uniform(*cfg.gain_db) / 20)
        return wave.astype(np.float32)


class SpecAugment(nn.Module):
    """SpecAugment-style time and frequency (channel) masking on ``[B, T, F]``
    features, applied inside each clip's valid frames only. The model calls it
    in training mode; it has no parameters."""

    def __init__(self, time_masks: int, time_width: float, freq_masks: int, freq_width: float):
        super().__init__()
        self.time_masks, self.time_width = time_masks, time_width
        self.freq_masks, self.freq_width = freq_masks, freq_width

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        x = x.clone()
        channels = x.shape[-1]
        for b, length in enumerate(lengths.tolist()):
            length = min(int(length), x.shape[1])
            max_t = int(self.time_width * length)
            for _ in range(self.time_masks if max_t > 0 else 0):
                width = int(torch.randint(0, max_t + 1, ()))
                start = int(torch.randint(0, max(1, length - width + 1), ()))
                x[b, start : start + width] = 0
            max_f = int(self.freq_width * channels)
            for _ in range(self.freq_masks if max_f > 0 else 0):
                width = int(torch.randint(0, max_f + 1, ()))
                start = int(torch.randint(0, channels - width + 1, ()))
                x[b, :length, start : start + width] = 0
        return x


# ------------------------------------------------------------------------ datasets
def _item_rng() -> np.random.Generator:
    # Drawn from torch's RNG, which DataLoader seeds per worker -> reproducible runs.
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, ()).item()))


def load_clip(path: str, audio_cfg: AudioConfig) -> np.ndarray:
    """Read -> mono -> resample -> trim edges (no normalisation, no cropping)."""
    wave, _ = load_audio(path, audio_cfg.sample_rate)
    if audio_cfg.trim_silence:
        wave, _ = trim_silence(wave, audio_cfg.sample_rate, audio_cfg.trim_top_db)
    return wave


class AccentDataset(Dataset):
    """Training clips: random crop to ``max_duration``, augmentation, normalisation."""

    def __init__(
        self,
        records: Sequence[Record],
        accents: Sequence[str],
        audio_cfg: AudioConfig,
        augment: WaveformAugment | None = None,
    ):
        self.records = list(records)
        self.index = {a: i for i, a in enumerate(accents)}
        self.audio_cfg = audio_cfg
        self.augment = augment
        self.max_samples = int(audio_cfg.max_duration * audio_cfg.sample_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        rng = _item_rng()
        wave = random_crop(load_clip(rec.path, self.audio_cfg), self.max_samples, rng)
        if self.augment is not None:
            wave = self.augment(wave, rng)
        if self.audio_cfg.normalize:
            wave = normalize_waveform(wave)
        return {
            "waveform": torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32)),
            "label": self.index[rec.accent],
            "speaker_id": rec.speaker,
            "utt_id": rec.utt_id,
        }


class UtteranceDataset(Dataset):
    """Evaluation: whole trimmed utterances; segmentation and normalisation
    happen in ``evaluate.predict_waves`` exactly as at inference time."""

    def __init__(self, records: Sequence[Record], accents: Sequence[str], audio_cfg: AudioConfig):
        self.records = list(records)
        self.index = {a: i for i, a in enumerate(accents)}
        self.audio_cfg = audio_cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        return {
            "waveform": load_clip(rec.path, self.audio_cfg),
            "label": self.index.get(rec.accent, -1),
            "speaker_id": rec.speaker,
            "utt_id": rec.utt_id,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad to the longest clip: ``waveforms [B, T]``, ``attention_mask [B, T]``,
    ``lengths [B]``, ``labels [B]``, plus speaker / utterance IDs."""
    batch = pad_batch([it["waveform"] for it in items])
    batch["labels"] = torch.tensor([it["label"] for it in items], dtype=torch.long)
    batch["speaker_ids"] = [it["speaker_id"] for it in items]
    batch["utt_ids"] = [it["utt_id"] for it in items]
    return batch


def collate_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items


class BucketBatchSampler(Sampler[list[int]]):
    """Shuffled batches of similar duration, so little compute is spent on padding.

    Indices are shuffled, cut into pools of ``batch_size * pool_factor``, each pool
    is sorted by duration and split into batches, and the batch order is shuffled.
    Deterministic for a given ``seed`` and epoch (call ``set_epoch``)."""

    def __init__(
        self,
        durations: Sequence[float],
        batch_size: int,
        seed: int = 0,
        pool_factor: int = 50,
        drop_last: bool = False,
    ):
        self.durations = np.asarray(durations, dtype=np.float64)
        self.batch_size, self.seed, self.pool = (
            batch_size,
            seed,
            batch_size * pool_factor,
        )
        self.drop_last, self.epoch = drop_last, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.durations))
        batches = []
        for s in range(0, len(order), self.pool):
            pool = order[s : s + self.pool]
            pool = pool[np.argsort(self.durations[pool], kind="stable")]
            for b in range(0, len(pool), self.batch_size):
                batch = pool[b : b + self.batch_size].tolist()
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        return [batches[i] for i in rng.permutation(len(batches))]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        n = len(self.durations)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


def build_train_loader(cfg: Config, records: Sequence[Record], seed: int) -> DataLoader:
    augment = WaveformAugment(cfg.augment, cfg.audio.sample_rate) if cfg.augment.enabled else None
    dataset = AccentDataset(records, cfg.data.accents, cfg.audio, augment)
    common: dict[str, Any] = {
        "num_workers": cfg.data.num_workers,
        "collate_fn": collate_batch,
        "worker_init_fn": seed_worker,
        "generator": make_generator(seed),
        "pin_memory": torch.cuda.is_available(),
        # Workers are re-created each epoch from a per-epoch seed, so a resumed run
        # sees exactly the same augmentation randomness as an uninterrupted one.
        "persistent_workers": False,
    }
    if cfg.data.bucket_by_length:
        capped = [
            min(r.duration or cfg.audio.max_duration, cfg.audio.max_duration) for r in records
        ]
        sampler = BucketBatchSampler(capped, cfg.train.batch_size, seed)
        return DataLoader(dataset, batch_sampler=sampler, **common)
    return DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, **common)


def build_eval_loader(cfg: Config, records: Sequence[Record]) -> DataLoader:
    dataset = UtteranceDataset(records, cfg.data.accents, cfg.audio)
    return DataLoader(
        dataset,
        batch_size=max(1, cfg.eval.batch_size // 4),
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_list,
    )
