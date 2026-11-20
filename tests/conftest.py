"""Shared fixtures. Everything runs on CPU with synthetic 16 kHz signals and tiny,
randomly initialised models built from configs, so the suite needs no dataset,
no GPU and no network access."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys

import numpy as np
import pytest
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never reach the Hub from tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config

SR = 16000
TEST_ACCENTS = ["Tamil", "Hindi", "Bengali"]

# Small WavLM-style encoder: real conv strides (20 ms frames), 3 transformer layers.
TINY_HF = {
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "conv_dim": [16] * 7,
    "num_conv_pos_embeddings": 16,
    "num_conv_pos_embedding_groups": 2,
}

# Lexicon in the tiny phoneme model's vocabulary (eSpeak-style IPA).
LEXICON = {
    "please": "p l iː z",
    "call": "k ɔː l",
    "stella": "s t ɛ l ə",
    "very": "v ɛ ɹ i",
    "good": "ɡ ʊ d",
    "hello": "h ə l oʊ",
    "world": "w ɜː l d",
}
PHONE_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"] + sorted(
    {p for phones in LEXICON.values() for p in phones.split()} | {"w", "v", "ʋ", "t", "ʃ", "θ"}
)


def tiny_hf_overrides(norm: str = "layer") -> list[str]:
    """Config overrides for a randomly initialised tiny HF backbone."""
    hf = dict(TINY_HF, feat_extract_norm=norm, do_stable_layer_norm=norm == "layer")
    return [
        "model.backbone=hf",
        "model.pretrained=false",
        "model.layers=[1,2]",
        "model.hidden_dim=32",
        "model.attention_dim=16",
        f"model.hf_overrides={json.dumps(hf)}",
    ]


def mel_overrides() -> list[str]:
    return ["model.backbone=mel_cnn", "model.hidden_dim=48", "model.attention_dim=16"]


def make_cfg(*overrides: str, extra: list[str] | None = None):
    return load_config(None, list(overrides) + list(extra or []))


@pytest.fixture
def capture_logs(caplog):
    """``with capture_logs(): ...`` records the project logger into ``caplog``.

    ``get_logger`` turns propagation off, and pytest < 9 captures only through
    the root logger; attaching the handler of the running test phase directly
    works with every pytest version and in any test order."""

    @contextlib.contextmanager
    def capture(level=logging.WARNING, name="accent"):
        logger = logging.getLogger(name)
        handler = caplog.handler
        attach = handler not in logger.handlers
        if attach:
            logger.addHandler(handler)
        try:
            with caplog.at_level(level, logger=name):
                yield caplog
        finally:
            if attach:
                logger.removeHandler(handler)

    return capture


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_data(tmp_path):
    """3 accents x 4 speakers x 3 utterances (fresh copy per test)."""
    from src.synthetic import make_dataset

    return make_dataset(str(tmp_path / "data"), TEST_ACCENTS, 4, 3, durations=(1.0, 2.0))


def write_lexicon(path) -> str:
    with open(path, "w", encoding="utf-8") as f:
        for word, phones in LEXICON.items():
            f.write(f"{word}\t{phones}\n")
    return str(path)


@pytest.fixture(scope="session")
def tiny_phoneme_model(tmp_path_factory):
    """A saved, randomly initialised wav2vec2 CTC phoneme model (+ vocab.json)."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

    d = tmp_path_factory.mktemp("phoneme_model")
    cfg = Wav2Vec2Config(
        vocab_size=len(PHONE_TOKENS),
        pad_token_id=0,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(16,) * 7,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer",
        do_stable_layer_norm=True,
    )
    torch.manual_seed(0)
    Wav2Vec2ForCTC(cfg).save_pretrained(str(d))
    with open(d / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(PHONE_TOKENS)}, f, ensure_ascii=False)
    with open(d / "preprocessor_config.json", "w", encoding="utf-8") as f:
        json.dump({"do_normalize": True, "sampling_rate": SR}, f)
    lexicon = write_lexicon(d / "lexicon.tsv")
    return {"path": str(d), "lexicon": lexicon}


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory):
    """A real best.pt from a short mel_cnn run on synthetic data (shared, read-only)."""
    from src.synthetic import make_dataset
    from src.train import run

    root = tmp_path_factory.mktemp("trained")
    manifest = make_dataset(str(root / "data"), TEST_ACCENTS, 4, 3, durations=(1.0, 2.0))
    out_dir = str(root / "run")
    cfg = make_cfg(
        *mel_overrides(),
        f"data.manifest={manifest}",
        f"data.accents={TEST_ACCENTS}",
        "data.num_workers=0",
        "data.val_frac=0.25",
        "data.test_frac=0.25",
        "train.epochs=3",
        "train.batch_size=4",
        "train.log_every_n_steps=0",
        f"train.out_dir={out_dir}",
        "eval.segment_seconds=1.0",
        "eval.segment_hop_seconds=0.5",
    )
    metrics = run(cfg)
    return {
        "out_dir": out_dir,
        "checkpoint": os.path.join(out_dir, "best.pt"),
        "manifest": manifest,
        "metrics": metrics,
    }
