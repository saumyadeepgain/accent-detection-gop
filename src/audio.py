"""Audio I/O and preprocessing shared by training, evaluation and inference.

    wave, info = load_audio("clip.mp3", target_sr=16000)   # any rate / channels -> 16 kHz mono
    wave, (start, end) = trim_silence(wave, 16000)          # energy-based edge trim
    x = normalize_waveform(wave)                            # zero mean, unit variance
    batch = pad_batch([x1, x2])                             # waveforms + attention_mask + lengths

Everything here is per-utterance and happens *before* padding, so a clip is
preprocessed identically whether it is scored alone or inside a batch.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")


class AudioError(ValueError):
    """The file exists but cannot be used (undecodable, empty, non-finite, too short)."""


@dataclass
class AudioInfo:
    path: str
    sample_rate: int  # the file's own rate, before resampling
    channels: int
    duration: float  # seconds


# --------------------------------------------------------------------------- reading
def read_audio(path: str) -> tuple[np.ndarray, int]:
    """Decode ``path`` to float32 ``[channels, samples]`` at the file's own rate."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)  # [samples, channels]
        data = data.T
    except Exception as exc:  # unsupported container/codec, truncated file, ...
        data, sr = _read_with_librosa(path, exc)
    data = np.ascontiguousarray(data, dtype=np.float32)
    if data.size == 0:
        raise AudioError(f"{path} contains no audio samples")
    if not np.isfinite(data).all():
        raise AudioError(f"{path} contains NaN or infinite samples")
    return data, int(sr)


def _read_with_librosa(path: str, original: Exception) -> tuple[np.ndarray, int]:
    try:
        import librosa  # optional: handles formats libsndfile cannot (via audioread/ffmpeg)
    except ImportError:
        raise AudioError(f"Cannot decode {path}: {type(original).__name__}: {original}") from None
    try:
        data, sr = librosa.load(path, sr=None, mono=False)
    except Exception as exc:
        raise AudioError(f"Cannot decode {path}: {type(exc).__name__}: {exc}") from exc
    return np.atleast_2d(data), int(sr)


def audio_info(path: str) -> AudioInfo:
    """Header-only metadata (fast); falls back to decoding when the header is unusable."""
    try:
        info = sf.info(path)
    except Exception:  # header unreadable: decode to find out (and raise AudioError if broken)
        info = None
    if info is not None and info.frames > 0 and info.samplerate > 0:
        return AudioInfo(path, int(info.samplerate), int(info.channels), info.duration)
    data, sr = read_audio(path)
    return AudioInfo(path, sr, data.shape[0], data.shape[1] / sr)


# ---------------------------------------------------------------------- conversions
def to_mono(wave: np.ndarray) -> np.ndarray:
    """``[channels, samples]`` or ``[samples]`` -> ``[samples]`` (channel average)."""
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        return wave
    if wave.ndim != 2:
        raise AudioError(f"Expected [channels, samples], got shape {wave.shape}")
    return wave.mean(axis=0, dtype=np.float64).astype(np.float32)


def resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Band-limited polyphase resampling (anti-aliasing filter included), last axis."""
    if orig_sr <= 0 or target_sr <= 0:
        raise AudioError(f"Invalid sample rates {orig_sr} -> {target_sr}")
    wave = np.asarray(wave, dtype=np.float32)
    if orig_sr == target_sr or wave.shape[-1] == 0:
        return wave
    g = gcd(int(orig_sr), int(target_sr))
    out = resample_poly(wave, target_sr // g, orig_sr // g, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


def load_audio(
    path: str, target_sr: int = 16000, mono: bool = True
) -> tuple[np.ndarray, AudioInfo]:
    """Read any supported file and return float32 audio at ``target_sr``.

    ``mono=True`` gives ``[samples]``; ``mono=False`` keeps ``[channels, samples]``."""
    data, sr = read_audio(path)
    info = AudioInfo(path, sr, data.shape[0], data.shape[1] / sr)
    wave = to_mono(data) if mono else data
    return resample(wave, sr, target_sr), info


# ------------------------------------------------------------------- level / silence
def rms_dbfs(wave: np.ndarray) -> float:
    """RMS level in dB relative to full scale (an all-zero clip gives -200 dB)."""
    wave = np.asarray(wave)
    if wave.size == 0:
        return -math.inf
    rms = math.sqrt(float(np.mean(np.square(wave, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 1e-10))


def is_silent(wave: np.ndarray, threshold_db: float = -60.0) -> bool:
    return rms_dbfs(wave) < threshold_db


def trim_silence(
    wave: np.ndarray,
    sr: int,
    top_db: float = 40.0,
    frame_seconds: float = 0.025,
    hop_seconds: float = 0.010,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut leading/trailing frames quieter than ``max_frame_db - top_db``.

    Returns the trimmed audio and the kept ``(start, end)`` sample range, so
    callers can map times back to the original file. Silent or very short
    clips are returned unchanged rather than trimmed to nothing."""
    wave = np.asarray(wave, dtype=np.float32)
    n = wave.shape[-1]
    frame = max(1, int(round(sr * frame_seconds)))
    hop = max(1, int(round(sr * hop_seconds)))
    if n < frame:
        return wave, (0, n)
    csum = np.concatenate([[0.0], np.cumsum(np.square(wave, dtype=np.float64))])
    starts = np.arange(0, n - frame + 1, hop)
    energy = (csum[starts + frame] - csum[starts]) / frame  # O(n) memory, any clip length
    db = 10.0 * np.log10(np.maximum(energy, 1e-20))
    if db.max() <= -190.0:  # digital silence: nothing to anchor on
        return wave, (0, n)
    keep = np.flatnonzero(db > db.max() - top_db)
    start, end = int(starts[keep[0]]), min(n, int(starts[keep[-1]]) + frame)
    return wave[start:end], (start, end)


def normalize_waveform(wave: np.ndarray | torch.Tensor, eps: float = 1e-7):
    """Zero mean, unit variance (the wav2vec2 / WavLM feature-extractor convention).

    ``eps`` keeps silent clips finite: an all-zero input stays all-zero."""
    if isinstance(wave, torch.Tensor):
        x = wave.float()
        return (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + eps)
    x = np.asarray(wave, dtype=np.float32)
    if x.size == 0:
        return x
    return ((x - x.mean()) / np.sqrt(x.var() + eps)).astype(np.float32)


# ------------------------------------------------------------------ cropping / batching
def random_crop(wave: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if wave.shape[-1] <= max_samples:
        return wave
    start = int(rng.integers(0, wave.shape[-1] - max_samples + 1))
    return wave[..., start : start + max_samples]


def segment_bounds(
    num_samples: int, sr: int, segment_seconds: float, hop_seconds: float
) -> list[tuple[int, int]]:
    """Overlapping ``(start, end)`` windows covering the whole clip; the last
    window is aligned to the end so no audio is skipped."""
    seg = max(1, int(round(segment_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))
    if num_samples <= seg:
        return [(0, num_samples)]
    starts = list(range(0, num_samples - seg + 1, hop))
    if starts[-1] + seg < num_samples:
        starts.append(num_samples - seg)
    return [(s, s + seg) for s in starts]


def pad_batch(waves: list[Any]) -> dict[str, torch.Tensor]:
    """Right-pad variable-length mono clips.

    Returns ``waveforms [B, T_max]`` (float32, zeros in the padding),
    ``attention_mask [B, T_max]`` (1 = real sample) and ``lengths [B]``."""
    if not waves:
        raise ValueError("pad_batch needs at least one waveform")
    tensors = [torch.as_tensor(np.asarray(w, dtype=np.float32)).reshape(-1) for w in waves]
    lengths = torch.tensor([t.numel() for t in tensors], dtype=torch.long)
    batch = torch.zeros(len(tensors), int(lengths.max()), dtype=torch.float32)
    mask = torch.zeros(len(tensors), int(lengths.max()), dtype=torch.long)
    for i, t in enumerate(tensors):
        batch[i, : t.numel()] = t
        mask[i, : t.numel()] = 1
    return {"waveforms": batch, "attention_mask": mask, "lengths": lengths}


def collect_audio_files(inputs: list[str], recursive: bool = False) -> list[str]:
    """Expand files and folders into a sorted, de-duplicated list of audio paths.

    Explicit files are kept whatever their extension (a broken file should be
    reported, not skipped silently); folder contents are filtered by extension."""
    explicit: list[str] = []
    found: list[str] = []
    for item in inputs:
        if os.path.isfile(item):
            explicit.append(item)
        elif os.path.isdir(item):
            if recursive:
                for root, _, files in os.walk(item):
                    found += [os.path.join(root, f) for f in files]
            else:
                found += [os.path.join(item, f) for f in os.listdir(item)]
        else:
            raise FileNotFoundError(f"Input not found: {item}")
    found = [p for p in found if os.path.isfile(p) and p.lower().endswith(AUDIO_EXTENSIONS)]
    return list(dict.fromkeys(explicit + sorted(found)))
