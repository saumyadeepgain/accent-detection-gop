"""Text -> phones (G2P), the CTC phoneme recogniser, and CTC forced alignment.

The commonest silent GOP bug is a symbol mismatch: G2P speaks ARPAbet, the
acoustic model speaks eSpeak IPA, and an unmapped phone ends up scored as
``<unk>``. Here the acoustic model's ``vocab.json`` is the single authority:
every G2P phone is normalised and checked against it, and anything that does
not resolve is reported instead of being scored.

    recognizer = PhonemeRecognizer("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    words = phonemize("very good", EspeakG2P("en-us"), recognizer.vocab)
    log_probs = recognizer.log_probs(wave_16k)                 # [frames, vocab] log_softmax
    spans = forced_align(log_probs, [p.token_id for w in words for p in w.phones],
                         recognizer.blank_id)
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .audio import normalize_waveform

# ARPAbet (CMUdict, MFA english_us_arpa) -> eSpeak-style IPA, General American.
ARPABET_TO_IPA = {
    "AA": "ɑː", "AE": "æ", "AH": "ʌ", "AO": "ɔː", "AW": "aʊ", "AY": "aɪ", "B": "b",
    "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɜː", "EY": "eɪ", "F": "f",
    "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "iː", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "UH": "ʊ", "UW": "uː", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}  # fmt: skip
# Unstressed variants that eSpeak writes differently.
ARPABET_UNSTRESSED = {"AH0": "ə", "ER0": "ɚ"}
_ARPABET_RE = re.compile(r"^[A-Z]{1,2}[012]?$")

STRESS_MARKS = "ˈˌ"
TIE_BARS = "͜͡"
LENGTH_MARK = "ː"
ASCII_LOOKALIKES = {"g": "ɡ", ":": "ː"}


class AlignmentError(ValueError):
    """The transcript cannot be aligned to the audio (e.g. audio too short)."""


class UnknownPhoneError(ValueError):
    """A G2P phone has no counterpart in the acoustic model vocabulary."""


# ---------------------------------------------------------------------------- data
@dataclass
class Phone:
    symbol: str  # as it appears in the acoustic model vocabulary
    token_id: int
    source: str  # what the G2P produced before normalisation


@dataclass
class Word:
    text: str
    phones: list[Phone] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)  # G2P symbols the model cannot score


@dataclass
class Span:
    """Frames ``[start, end)`` in which the CTC path emits target number ``index``."""

    index: int
    token_id: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


# ------------------------------------------------------------------- symbol handling
def _strip_marks(symbol: str) -> str:
    return "".join(c for c in symbol if c not in STRESS_MARKS and c not in TIE_BARS)


def _strip_diacritics(symbol: str) -> str:
    decomposed = unicodedata.normalize("NFD", symbol)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_symbol(symbol: str, vocab: dict[str, int]) -> str | None:
    """Map a G2P symbol onto the model vocabulary, or ``None`` if impossible.

    Tries, in order: the symbol itself, NFC, without stress marks / tie bars,
    ASCII look-alikes replaced (``g`` -> ``ɡ``), without the length mark, and
    without diacritics. Only the first form found in ``vocab`` is used."""
    candidates = [symbol, unicodedata.normalize("NFC", symbol)]
    base = _strip_marks(unicodedata.normalize("NFC", symbol))
    candidates.append(base)
    ascii_fixed = "".join(ASCII_LOOKALIKES.get(c, c) for c in base)
    candidates += [ascii_fixed, ascii_fixed.replace(LENGTH_MARK, "")]
    candidates.append(_strip_diacritics(ascii_fixed))
    candidates.append(_strip_diacritics(ascii_fixed).replace(LENGTH_MARK, ""))
    for cand in candidates:
        if cand and cand in vocab:
            return cand
    return None


def arpabet_to_ipa(phone: str) -> str:
    """``"AH0"`` -> ``"ə"``, ``"IY1"`` -> ``"iː"``; raises ``KeyError`` for non-ARPAbet."""
    phone = phone.upper()
    if phone in ARPABET_UNSTRESSED:
        return ARPABET_UNSTRESSED[phone]
    return ARPABET_TO_IPA[phone.rstrip("012")]


def tokenize_words(text: str) -> list[str]:
    """Lower-case words (letters, digits, apostrophes); punctuation is dropped."""
    return re.findall(r"[\w']+", unicodedata.normalize("NFC", text).lower())


# ------------------------------------------------------------------------------ G2P
class LexiconG2P:
    """Dictionary lookup. Entries are ``word<TAB>phone phone ...`` in IPA or ARPAbet.

    ARPAbet entries (CMUdict style, e.g. ``HH AH0 L OW1``) are converted to IPA;
    the result is still validated against the acoustic model vocabulary."""

    def __init__(self, lexicon: str | dict[str, list[str]]):
        if isinstance(lexicon, str):
            lexicon = self._read(lexicon)
        self.entries = {w.lower(): list(p) for w, p in lexicon.items()}

    @staticmethod
    def _read(path: str) -> dict[str, list[str]]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Lexicon not found: {path}")
        entries: dict[str, list[str]] = {}
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t") if "\t" in line else line.split(None, 1)
                if len(parts) != 2 or not parts[1].split():
                    raise ValueError(f"{path}:{line_no}: expected 'word<TAB>phones'")
                entries.setdefault(parts[0].lower(), parts[1].split())  # first variant wins
        return entries

    def __call__(self, text: str) -> list[tuple[str, list[str]]]:
        words = tokenize_words(text)
        missing = [w for w in words if w not in self.entries]
        if missing:
            raise KeyError(f"Words missing from the lexicon: {', '.join(sorted(set(missing)))}")
        out = []
        for w in words:
            phones = self.entries[w]
            if all(_ARPABET_RE.match(p) for p in phones):
                phones = [arpabet_to_ipa(p) for p in phones]
            out.append((w, phones))
        return out


class EspeakG2P:
    """eSpeak-ng via ``phonemizer``: the phonemiser the wav2vec2 eSpeak models were
    trained with, so its symbols match their vocabulary by construction."""

    def __init__(self, language: str = "en-us"):
        try:
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator
        except ImportError as exc:
            raise ImportError(
                "g2p=espeak needs `pip install phonemizer` and the espeak-ng system package "
                "(apt install espeak-ng / brew install espeak-ng); or use g2p=lexicon."
            ) from exc
        try:
            self.backend = EspeakBackend(
                language, preserve_punctuation=False, with_stress=False, tie=False
            )
        except RuntimeError as exc:  # espeak-ng library not found
            raise ImportError(f"espeak-ng is not available: {exc}") from exc
        self.separator = Separator(phone=" ", word="", syllable="")

    def __call__(self, text: str) -> list[tuple[str, list[str]]]:
        words = tokenize_words(text)
        if not words:
            return []
        phonemized = self.backend.phonemize(words, separator=self.separator, strip=True)
        return [(w, p.split()) for w, p in zip(words, phonemized, strict=True)]


def build_g2p(kind: str, language: str = "en-us", lexicon: str | None = None) -> Callable:
    if kind == "lexicon":
        if not lexicon:
            raise ValueError("gop.g2p=lexicon needs gop.lexicon=<path to TSV>")
        return LexiconG2P(lexicon)
    if kind == "espeak":
        return EspeakG2P(language)
    raise ValueError(f"Unknown G2P backend '{kind}' (use espeak or lexicon)")


def phonemize(
    text: str,
    g2p: Callable[[str], list[tuple[str, list[str]]]],
    vocab: dict[str, int],
    unknown: str = "skip",
) -> list[Word]:
    """Transcript -> words with phones that exist in ``vocab``.

    ``unknown="skip"`` drops unmappable symbols but records them in
    ``Word.unknown``; ``unknown="error"`` raises ``UnknownPhoneError``."""
    words = []
    for text_word, raw_phones in g2p(text):
        word = Word(text_word)
        for raw in raw_phones:
            symbol = normalize_symbol(raw, vocab)
            if symbol is None:
                if unknown == "error":
                    raise UnknownPhoneError(
                        f"Phone '{raw}' in '{text_word}' is not in the acoustic model vocabulary"
                    )
                word.unknown.append(raw)
                continue
            word.phones.append(Phone(symbol, vocab[symbol], raw))
        words.append(word)
    return words


# ------------------------------------------------------------------ acoustic model
def _resolve_file(name_or_path: str, filename: str) -> str | None:
    if os.path.isdir(name_or_path):
        path = os.path.join(name_or_path, filename)
        return path if os.path.isfile(path) else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(name_or_path, filename)
    except Exception:
        return None


def load_vocab(name_or_path: str, language: str | None = None) -> dict[str, int]:
    """``vocab.json`` of a CTC checkpoint (flat, or nested per language)."""
    path = _resolve_file(name_or_path, "vocab.json")
    if path is None:
        raise FileNotFoundError(f"No vocab.json found for {name_or_path}")
    with open(path, encoding="utf-8") as f:
        vocab = json.load(f)
    if vocab and all(isinstance(v, dict) for v in vocab.values()):  # {lang: {token: id}}
        if language not in vocab:
            raise ValueError(f"{path} is per-language; pick one of {sorted(vocab)}")
        vocab = vocab[language]
    return {str(k): int(v) for k, v in vocab.items()}


class PhonemeRecognizer:
    """A Hugging Face ``Wav2Vec2ForCTC`` phoneme model returning frame log-probabilities.

    Frame shift and receptive field are read from the model's convolution
    config rather than hard-coded, and input normalisation follows the
    checkpoint's ``preprocessor_config.json`` (``do_normalize``)."""

    def __init__(self, name_or_path: str, device: Any = "cpu", language: str | None = None):
        from transformers import Wav2Vec2ForCTC

        self.name = name_or_path
        self.device = torch.device(device)
        self.model = Wav2Vec2ForCTC.from_pretrained(name_or_path).to(self.device).eval()
        self.vocab = load_vocab(name_or_path, language)
        self.id_to_token = {i: t for t, i in self.vocab.items()}
        blank = self.model.config.pad_token_id
        self.blank_id = int(blank if blank is not None else self.vocab.get("<pad>", 0))
        prep: dict[str, Any] = {}
        prep_path = _resolve_file(name_or_path, "preprocessor_config.json")
        if prep_path:
            with open(prep_path, encoding="utf-8") as f:
                prep = json.load(f)
        self.do_normalize = bool(prep.get("do_normalize", True))
        self.sample_rate = int(prep.get("sampling_rate", 16000))
        strides = list(self.model.config.conv_stride)
        kernels = list(self.model.config.conv_kernel)
        hop = math.prod(strides)
        receptive = 1 + sum((k - 1) * math.prod(strides[:i]) for i, k in enumerate(kernels))
        self.frame_shift = hop / self.sample_rate  # seconds between frames (0.02 for wav2vec2)
        self.receptive_field = receptive / self.sample_rate  # seconds seen by one frame
        self.min_samples = receptive

    def num_frames(self, num_samples: int) -> int:
        lengths = self.model._get_feat_extract_output_lengths(torch.tensor([num_samples]))
        return int(lengths[0])

    @torch.inference_mode()
    def log_probs(self, wave: np.ndarray) -> torch.Tensor:
        """``wave``: mono float32 at ``self.sample_rate`` -> ``[frames, vocab]`` log-probs (CPU)."""
        wave = np.asarray(wave, dtype=np.float32).reshape(-1)
        if wave.shape[0] < self.min_samples:
            raise AlignmentError(
                f"Audio too short for the phoneme model: {wave.shape[0]} samples "
                f"< receptive field {self.min_samples}"
            )
        x = torch.from_numpy(normalize_waveform(wave) if self.do_normalize else wave)
        logits = self.model(x.unsqueeze(0).to(self.device)).logits[0]
        vocab_size = logits.shape[-1]
        if vocab_size < len(self.vocab):
            raise ValueError(f"Model has {vocab_size} outputs but vocab.json {len(self.vocab)}")
        return torch.log_softmax(logits.float(), dim=-1).cpu()


# ----------------------------------------------------------------- forced alignment
def min_frames_needed(targets: Sequence[int]) -> int:
    """CTC needs one frame per label plus a blank between identical neighbours."""
    repeats = sum(1 for a, b in zip(targets, targets[1:], strict=False) if a == b)
    return len(targets) + repeats


def forced_align(log_probs: torch.Tensor, targets: Sequence[int], blank_id: int) -> list[Span]:
    """CTC Viterbi alignment: the most likely path that emits ``targets`` in order.

    ``log_probs`` must be log-probabilities ``[frames, vocab]`` (log_softmax).
    Every target gets at least one frame, so spans are never empty. Raises
    ``AlignmentError`` when the audio has too few frames for the sequence."""
    targets = [int(t) for t in targets]
    if not targets:
        return []
    lp = log_probs.detach().float().cpu().numpy().astype(np.float64)
    num_frames, vocab_size = lp.shape
    if any(t == blank_id or not 0 <= t < vocab_size for t in targets):
        raise ValueError("targets must be non-blank token ids inside the vocabulary")
    needed = min_frames_needed(targets)
    if num_frames < needed:
        raise AlignmentError(
            f"{num_frames} frames cannot hold {len(targets)} phones (needs >= {needed}); "
            "the audio is too short for this transcript"
        )
    ext = np.full(2 * len(targets) + 1, blank_id, dtype=np.int64)
    ext[1::2] = targets
    num_states = ext.shape[0]
    emit = lp[:, ext]  # [frames, states]
    can_skip = np.zeros(num_states, dtype=bool)
    can_skip[2:] = (ext[2:] != blank_id) & (ext[2:] != ext[:-2])

    alpha = np.full(num_states, -np.inf)
    alpha[0] = emit[0, 0]
    alpha[1] = emit[0, 1]
    back = np.zeros((num_frames, num_states), dtype=np.int8)
    neg = np.full(2, -np.inf)
    for t in range(1, num_frames):
        prev1 = np.concatenate([neg[:1], alpha[:-1]])
        prev2 = np.where(can_skip, np.concatenate([neg, alpha[:-2]]), -np.inf)
        cand = np.stack([alpha, prev1, prev2])
        arg = cand.argmax(axis=0)
        alpha = cand[arg, np.arange(num_states)] + emit[t]
        back[t] = arg

    state = num_states - 1 if alpha[-1] >= alpha[-2] else num_states - 2
    if not np.isfinite(alpha[state]):
        raise AlignmentError("No valid CTC path (all paths have zero probability)")
    path = np.empty(num_frames, dtype=np.int64)
    for t in range(num_frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])  # int(): NumPy 2 would keep int8 and overflow

    spans = []
    for k, token in enumerate(targets):
        frames = np.flatnonzero(path == 2 * k + 1)
        spans.append(Span(k, token, int(frames[0]), int(frames[-1]) + 1))
    return spans


def spans_from_times(
    intervals: Sequence[tuple[float, float]],
    frame_shift: float,
    num_frames: int,
    frame_offset: float | None = None,
) -> list[tuple[int, int]]:
    """External phone times (e.g. an MFA TextGrid) -> frame ranges ``[start, end)``.

    A phone owns every frame whose centre (``i * frame_shift + frame_offset``,
    default half a frame) lies in ``[t_start, t_end)``. Adjacent phones therefore
    never share a frame, and a phone shorter than a frame can get zero frames;
    ``gop.score_span`` handles that case by falling back to the nearest frame."""
    offset = 0.5 * frame_shift if frame_offset is None else frame_offset
    out = []
    for t0, t1 in intervals:
        start = math.ceil((t0 - offset) / frame_shift - 1e-9)
        end = math.ceil((t1 - offset) / frame_shift - 1e-9)
        start = min(max(start, 0), num_frames)
        out.append((start, min(max(end, start), num_frames)))
    return out
