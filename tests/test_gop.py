"""Alignment + GOP on synthetic phoneme posteriors with a known alignment."""

import math

import numpy as np
import pytest
import torch
from conftest import LEXICON

from src.aligner import (
    AlignmentError,
    LexiconG2P,
    PhonemeRecognizer,
    UnknownPhoneError,
    forced_align,
    min_frames_needed,
    normalize_symbol,
    phonemize,
    spans_from_times,
)
from src.config import GOPConfig
from src.gop import assess_pronunciation, prepare_log_probs, score_span, to_score

TOKENS = ["<pad>", "h", "ə", "l", "oʊ", "w", "ɜː", "d", "v", "ɡ", "t", "s", "ʃ"]
VOCAB = {t: i for i, t in enumerate(TOKENS)}
BLANK = 0
# (token or None for blank, number of frames): "hello world" with pauses
TRUTH = [
    (None, 3),
    ("h", 2),
    (None, 1),
    ("ə", 2),
    ("l", 2),
    (None, 1),
    ("oʊ", 3),
    (None, 4),
    ("w", 2),
    ("ɜː", 3),
    ("l", 2),
    ("d", 2),
    (None, 3),
]
NUM_FRAMES = sum(n for _, n in TRUTH)
LEX = LexiconG2P({w: LEXICON[w].split() for w in ("hello", "world")})


def posteriors(target_prob: float = 0.9, weaken: str | None = None, towards: str = "v"):
    """Log-probabilities ``[frames, vocab]`` matching ``TRUTH``; optionally move
    probability mass of phone ``weaken`` to ``towards``."""
    p = np.full((NUM_FRAMES, len(TOKENS)), 0.01)
    t, spans = 0, []
    for tok, n in TRUTH:
        for _ in range(n):
            if tok is None:
                p[t, BLANK] = 0.95
            elif tok == weaken:
                p[t, VOCAB[tok]] = target_prob
                p[t, VOCAB[towards]] = max(0.9 - target_prob, 0.01)
            else:
                p[t, VOCAB[tok]] = 0.9
            t += 1
        if tok is not None:
            spans.append((tok, t - n, t))
    p /= p.sum(axis=1, keepdims=True)
    return torch.log(torch.tensor(p, dtype=torch.float32)), spans


class FakeRecognizer:
    vocab, blank_id, frame_shift, sample_rate = VOCAB, BLANK, 0.02, 16000

    def __init__(self, log_probs):
        self.lp = log_probs

    def log_probs(self, wave):
        return self.lp


def assess(log_probs, text="hello world", cfg=None):
    wave = np.zeros(NUM_FRAMES * 320, dtype=np.float32)
    return assess_pronunciation(wave, text, FakeRecognizer(log_probs), LEX, cfg or GOPConfig())


def all_phones(report):
    return [p for w in report["words"] for p in w["phones"]]


# ------------------------------------------------------------------- alignment
def test_forced_alignment_recovers_the_true_spans():
    lp, truth = posteriors()
    spans = forced_align(lp, [VOCAB[t] for t, _, _ in truth], BLANK)
    assert [(s.start, s.end) for s in spans] == [(a, b) for _, a, b in truth]
    assert all(s.length >= 1 for s in spans)


def test_long_sequences_align_exactly():
    """Hundreds of phones (> 127 CTC states): a regression test for int8 backpointers."""
    rng = np.random.default_rng(0)
    vocab_size, num_phones = 40, 200
    targets = [int(t) for t in rng.integers(1, vocab_size, num_phones)]
    for i in range(1, num_phones):  # avoid identical neighbours for a simple layout
        if targets[i] == targets[i - 1]:
            targets[i] = targets[i] % (vocab_size - 1) + 1
    p = np.full((3 * num_phones + 1, vocab_size), 0.01)
    truth = []
    for k, tok in enumerate(targets):
        p[3 * k, BLANK] = 0.95
        p[3 * k + 1 : 3 * k + 3, tok] = 0.95
        truth.append((3 * k + 1, 3 * k + 3))
    p[-1, BLANK] = 0.95
    lp = torch.log(torch.tensor(p / p.sum(1, keepdims=True), dtype=torch.float32))
    spans = forced_align(lp, targets, BLANK)
    assert [(s.start, s.end) for s in spans] == truth


def test_repeated_phones_need_a_blank_between_them():
    assert min_frames_needed([3, 3, 3]) == 5
    lp, _ = posteriors()
    with pytest.raises(AlignmentError):
        forced_align(lp[:2], [3, 3], BLANK)
    spans = forced_align(lp[:5], [3, 3], BLANK)  # feasible: l _ l (+ blanks)
    assert spans[0].end <= spans[1].start - 1


def test_audio_too_short_for_transcript_is_reported_not_raised():
    lp, _ = posteriors()
    report = assess(lp[:5])
    assert report["error"] and "too short" in report["error"]


# ------------------------------------------------------------------------- GOP
def test_all_scores_finite_and_in_range():
    lp, _ = posteriors()
    report = assess(lp)
    assert report["error"] is None
    phones = all_phones(report)
    assert len(phones) == 8
    for p in phones:
        assert all(math.isfinite(p[k]) for k in ("lpp", "lpr", "score"))
        assert 0.0 < p["score"] < 1.0
        assert p["start"] <= p["end"]
    words = report["words"]
    assert all(math.isfinite(w["score"]) for w in words)
    assert words[0]["end"] <= words[1]["start"]  # the pause is not assigned to either word
    assert 0 <= report["overall_score"] <= 100


def test_lowering_target_probability_lowers_its_gop():
    previous = None
    for target_prob in (0.9, 0.6, 0.3, 0.05):
        lp, _ = posteriors(target_prob, weaken="w", towards="v")
        w = all_phones(assess(lp))[4]
        assert w["phone"] == "w"
        if previous is not None:
            assert w["lpp"] < previous["lpp"]
            assert w["lpr"] < previous["lpr"]
            assert w["score"] < previous["score"]
        previous = w
    assert previous["flagged"] and previous["heard_as"] == "v"


def test_rival_contrast_is_reported():
    lp, _ = posteriors(0.05, weaken="w", towards="v")
    report = assess(lp, cfg=GOPConfig(rivals={"w": ["v", "ʋ"]}))
    w = all_phones(report)[4]
    assert set(w["rivals"]) == {"v"}  # ʋ is not in this vocabulary: skipped, not mis-scored
    assert w["rivals"]["v"] < 0  # the rival beat the target
    assert report["flagged"][0]["phone"] == "w"


def test_zero_length_span_falls_back_to_nearest_frame():
    lp, _ = posteriors()
    prepared = prepare_log_probs(lp, BLANK, exclude_blank=True)
    for start in (0, 7, NUM_FRAMES):  # first frame, middle, past the end
        stats = score_span(prepared, start, start, VOCAB["ə"], BLANK)
        assert stats["zero_length"] and stats["frames"] == 1
        assert math.isfinite(stats["lpp"]) and math.isfinite(stats["lpr"])


def test_external_times_to_frames_never_overlap_and_may_be_empty():
    spans = spans_from_times([(0.0, 0.05), (0.05, 0.06), (0.06, 0.2)], 0.02, 10)
    assert spans == [(0, 2), (2, 3), (3, 10)]
    short = spans_from_times([(0.031, 0.039)], 0.02, 10)  # shorter than a frame
    assert short[0][0] == short[0][1]


def test_exact_zero_probabilities_stay_finite():
    lp, _ = posteriors()
    lp = lp.clone()
    lp[:, VOCAB["ʃ"]] = -math.inf  # log(0)
    report = assess(lp)
    assert all(math.isfinite(p["lpr"]) for p in all_phones(report))


def test_gop_depends_on_log_probabilities_not_logit_offsets():
    """Scores come from log_softmax, so shifting every logit of a frame by a
    constant (which raw-logit GOP would not survive) changes nothing."""
    lp, _ = posteriors()
    shifted = torch.log_softmax(lp + torch.linspace(0, 50, NUM_FRAMES)[:, None], dim=-1)
    a, b = all_phones(assess(lp)), all_phones(assess(shifted))
    for x, y in zip(a, b, strict=True):
        assert x["lpr"] == pytest.approx(y["lpr"], abs=1e-4)


def test_blank_frames_do_not_dilute_the_score():
    """With exclude_blank the score of a phone is the same whether its frames
    are dominated by blank or not, as long as non-blank ranks are unchanged."""
    lp, truth = posteriors()
    blanky = lp.clone()
    _, start, end = truth[0]
    blanky[start:end, BLANK] += 4.0  # make blank dominant on the /h/ frames
    blanky = torch.log_softmax(blanky, dim=-1)
    prepared_a = prepare_log_probs(lp, BLANK, True)
    prepared_b = prepare_log_probs(blanky, BLANK, True)
    a = score_span(prepared_a, start, end, VOCAB["h"], BLANK)
    b = score_span(prepared_b, start, end, VOCAB["h"], BLANK)
    assert a["lpr"] == pytest.approx(b["lpr"], abs=1e-4)


def test_score_mapping_is_monotone_and_centred():
    assert to_score(0.0) == pytest.approx(0.5)
    assert to_score(-10) < to_score(-1) < to_score(1) < to_score(10)
    assert to_score(-1e6) == 0.0 and to_score(1e6) == 1.0  # saturates, never overflows


# ------------------------------------------------------------------ G2P / vocab
def test_empty_transcript_and_unknown_words_are_errors_not_crashes():
    lp, _ = posteriors()
    assert "No phones" in assess(lp, text="!!!")["error"]
    assert "missing from the lexicon" in assess(lp, text="hello mars")["error"]


def test_symbols_are_normalised_to_the_model_vocabulary():
    assert normalize_symbol("g", VOCAB) == "ɡ"  # ASCII g -> IPA ɡ (U+0261)
    assert normalize_symbol("ˈh", VOCAB) == "h"  # stress mark
    assert normalize_symbol("ɜ:", VOCAB) == "ɜː"  # ASCII colon length mark
    assert normalize_symbol("q", VOCAB) is None


def test_unknown_phones_are_reported_or_rejected():
    g2p = LexiconG2P({"good": ["ɡ", "ʊ", "d"]})  # ʊ is not in VOCAB
    words = phonemize("good", g2p, VOCAB)
    assert [p.symbol for p in words[0].phones] == ["ɡ", "d"] and words[0].unknown == ["ʊ"]
    with pytest.raises(UnknownPhoneError):
        phonemize("good", g2p, VOCAB, unknown="error")


def test_arpabet_lexicon_is_converted():
    g2p = LexiconG2P({"go": ["G", "OW1"], "the": ["DH", "AH0"]})
    assert g2p("Go, the!") == [("go", ["ɡ", "oʊ"]), ("the", ["ð", "ə"])]


def test_espeak_symbols_exist_in_a_wav2vec2_espeak_vocabulary():
    pytest.importorskip("phonemizer")
    from src.aligner import EspeakG2P

    try:
        g2p = EspeakG2P("en-us")
    except ImportError:
        pytest.skip("espeak-ng not installed")
    words = g2p("very good weather")
    assert words[1] == ("good", ["ɡ", "ʊ", "d"])  # IPA ɡ, as in the model vocabulary
    assert all(p for _, phones in words for p in phones)


# ---------------------------------------------------------- real CTC model path
def test_phoneme_recognizer_end_to_end(tiny_phoneme_model, rng):
    rec = PhonemeRecognizer(tiny_phoneme_model["path"])
    assert rec.frame_shift == pytest.approx(0.02)  # 320-sample hop from the conv config
    assert rec.receptive_field == pytest.approx(0.025)
    wave = (0.1 * rng.standard_normal(2 * 16000)).astype(np.float32)
    lp = rec.log_probs(wave)
    assert lp.shape == (rec.num_frames(len(wave)), len(rec.vocab))
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(lp.shape[0]), atol=1e-4, rtol=0)
    g2p = LexiconG2P(tiny_phoneme_model["lexicon"])
    report = assess_pronunciation(wave, "please call stella", rec, g2p, GOPConfig())
    assert report["error"] is None and report["num_phones"] == 12
    assert all(math.isfinite(p["score"]) for p in all_phones(report))
    with pytest.raises(AlignmentError):
        rec.log_probs(np.zeros(100, dtype=np.float32))  # shorter than the receptive field
