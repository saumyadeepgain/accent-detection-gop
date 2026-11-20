"""End-to-end: train, resume, evaluate and predict through the real CLIs."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import TEST_ACCENTS, mel_overrides

from src.utils import load_checkpoint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_args(manifest, out_dir, *extra):
    sets = [
        *mel_overrides(),
        f"data.manifest={manifest}",
        f"data.accents={TEST_ACCENTS}",
        "data.num_workers=0",
        "data.val_frac=0.25",
        "data.test_frac=0.25",
        f"train.out_dir={out_dir}",
        "train.batch_size=4",
        "train.log_every_n_steps=0",
        "eval.segment_seconds=1.0",
        "eval.segment_hop_seconds=0.5",
        *extra,
    ]
    return [arg for s in sets for arg in ("--set", s)]


def test_training_writes_every_artifact(trained_run):
    out = trained_run["out_dir"]
    for name in (
        "best.pt",
        "last.pt",
        "split.csv",
        "config.yaml",
        "history.json",
        "metrics.json",
        "test_confusion_matrix.png",
        "val_reliability.png",
    ):
        assert os.path.isfile(os.path.join(out, name)), name
    assert os.listdir(os.path.join(out, "logs"))
    metrics = json.load(open(os.path.join(out, "metrics.json")))
    for key in ("accuracy", "macro_f1", "uar", "ece", "per_class", "confusion_matrix"):
        assert key in metrics["test"]
    best = load_checkpoint(trained_run["checkpoint"])
    assert best["accents"] == TEST_ACCENTS and "optimizer" not in best  # inference-only
    assert "temperature" in best


def test_resume_reproduces_an_uninterrupted_run(synthetic_data, tmp_path):
    from src.train import main

    straight, resumed = str(tmp_path / "straight"), str(tmp_path / "resumed")
    assert main(train_args(synthetic_data, straight, "train.epochs=2")) == 0
    assert (
        main(train_args(synthetic_data, resumed, "train.epochs=2", "train.stop_after_epochs=1"))
        == 0
    )
    first = load_checkpoint(os.path.join(resumed, "last.pt"))
    assert first["train_state"]["epoch"] == 1
    last = os.path.join(resumed, "last.pt")
    assert main(train_args(synthetic_data, resumed, "train.epochs=2", f"train.resume={last}")) == 0
    a = load_checkpoint(os.path.join(straight, "last.pt"))
    b = load_checkpoint(os.path.join(resumed, "last.pt"))
    assert a["train_state"]["epoch"] == b["train_state"]["epoch"] == 2
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=1e-6, rtol=0)


def test_resume_refuses_a_different_model(synthetic_data, tmp_path, trained_run):
    from src.train import main

    last = os.path.join(trained_run["out_dir"], "last.pt")
    out = str(tmp_path / "other")
    args = train_args(synthetic_data, out, f"train.resume={last}", "model.hidden_dim=32")
    assert main(args) == 1


def test_evaluate_accent_cli(trained_run, tmp_path):
    from src.evaluate import main

    out = str(tmp_path / "eval")
    code = main(
        [
            "accent",
            "--checkpoint",
            trained_run["checkpoint"],
            "--split",
            "test",
            "--out-dir",
            out,
            "--set",
            "data.num_workers=0",
            "--set",
            "eval.segment_seconds=1.0",
            "--set",
            "eval.segment_hop_seconds=0.5",
        ]
    )
    assert code == 0
    metrics = json.load(open(os.path.join(out, "test_metrics.json")))
    assert 0.0 <= metrics["macro_f1"] <= 1.0 and metrics["num_samples"] > 0
    assert os.path.isfile(os.path.join(out, "test_confusion_matrix.png"))


def test_predict_json_to_stdout_with_accent_and_gop(trained_run, tiny_phoneme_model, capsys):
    from src.predict import main

    clip = os.path.join(os.path.dirname(trained_run["manifest"]), "audio", "Hindi", "hin00_0.wav")
    code = main(
        [
            "--audio",
            clip,
            "--text",
            "please call stella",
            "--output",
            "-",
            "--accent-checkpoint",
            trained_run["checkpoint"],
            "--phoneme_model",
            tiny_phoneme_model["path"],
            "--g2p",
            "lexicon",
            "--lexicon",
            tiny_phoneme_model["lexicon"],
            "--flag_threshold",
            "0.6",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    report = json.loads(captured.out)  # stdout is pure JSON
    assert "Accent" in captured.err  # the human-readable report went to stderr
    result = report["results"][0]
    accent = result["accent"]
    assert accent["label"] in TEST_ACCENTS
    assert sum(accent["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert accent["verdict"] in ("consistent", "contested")
    pron = result["pronunciation"]
    assert pron["num_phones"] == 12 and 0 <= pron["overall_score"] <= 100
    for word in pron["words"]:
        assert word["start"] <= word["end"]
        for phone in word["phones"]:
            assert phone["flagged"] == (phone["score"] < 0.6)
    assert report["models"]["flag_threshold"] == 0.6


def test_predict_needs_something_to_do(trained_run):
    from src.predict import main

    assert main(["--audio", trained_run["manifest"]]) == 1  # no checkpoint, no transcript


def test_transcript_file_per_clip(tmp_path, tiny_phoneme_model):
    from src.predict import main

    t = np.arange(24000) / 16000
    for name in ("a", "b"):
        sf.write(
            tmp_path / f"{name}.wav", (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 16000
        )
    (tmp_path / "t.tsv").write_text("a.wav\tvery good\nb\thello world\n", encoding="utf-8")
    out = tmp_path / "r.json"
    code = main(
        [
            "--audio",
            str(tmp_path / "a.wav"),
            str(tmp_path / "b.wav"),
            "--text-file",
            str(tmp_path / "t.tsv"),
            "--output",
            str(out),
            "--phoneme_model",
            tiny_phoneme_model["path"],
            "--g2p",
            "lexicon",
            "--lexicon",
            tiny_phoneme_model["lexicon"],
        ]
    )
    assert code == 0
    results = json.loads(out.read_text())["results"]
    assert [r["pronunciation"]["transcript"] for r in results] == ["very good", "hello world"]


def test_evaluate_gop_cli_with_human_scores_and_phone_labels(tmp_path, tiny_phoneme_model):
    from src.evaluate import main

    rng = np.random.default_rng(0)
    rows = ["path,text,human_score,phone_labels"]
    for i in range(6):
        sf.write(
            tmp_path / f"u{i}.wav", (0.1 * rng.standard_normal(24000)).astype(np.float32), 16000
        )
        labels = " ".join(str(v) for v in rng.integers(0, 2, 7))  # very good = 7 phones
        rows.append(f"u{i}.wav,very good,{i + 1},{labels}")
    (tmp_path / "gop.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    out = tmp_path / "gop_eval"
    code = main(
        [
            "gop",
            "--manifest",
            str(tmp_path / "gop.csv"),
            "--out-dir",
            str(out),
            "--phoneme_model",
            tiny_phoneme_model["path"],
            "--g2p",
            "lexicon",
            "--lexicon",
            tiny_phoneme_model["lexicon"],
        ]
    )
    assert code == 0
    metrics = json.load(open(out / "gop_metrics.json"))
    assert metrics["utterances"] == 6 and metrics["errors"] == 0
    for key in ("pearson", "spearman", "auc_roc", "eer"):
        assert key in metrics
    assert metrics["num_labelled_phones"] == 42
    assert (out / "gop_phone_distributions.png").is_file()
    assert len((out / "gop_results.jsonl").read_text().splitlines()) == 6


@pytest.mark.parametrize("script", ["src/train.py", "src/evaluate.py", "src/predict.py"])
def test_scripts_run_directly(script):
    result = subprocess.run(
        [sys.executable, script, "--help"], cwd=REPO, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
