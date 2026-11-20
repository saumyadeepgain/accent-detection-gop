"""Silent, stereo, odd-sample-rate, short and corrupted audio must be handled
gracefully everywhere: loading, preprocessing, the model, GOP and the CLI."""

import json
import math

import numpy as np
import pytest
import soundfile as sf
import torch
from conftest import SR, make_cfg, mel_overrides

from src.audio import (
    AudioError,
    is_silent,
    load_audio,
    normalize_waveform,
    pad_batch,
    resample,
    rms_dbfs,
    trim_silence,
)
from src.dataset import Record, verify_records
from src.model import build_model


def tone(freq: float, seconds: float, sr: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(wave: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
    return float(np.fft.rfftfreq(len(wave), 1 / sr)[spectrum.argmax()])


# ------------------------------------------------------------------- silence
def test_silent_audio_is_detected_and_stays_finite(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(SR, dtype=np.float32), SR)
    wave, info = load_audio(str(path), SR)
    assert is_silent(wave) and rms_dbfs(wave) == -200.0
    assert np.all(np.isfinite(normalize_waveform(wave)))
    assert np.all(normalize_waveform(wave) == 0)
    trimmed, (start, end) = trim_silence(wave, SR)
    assert (start, end) == (0, SR) and len(trimmed) == SR  # not trimmed to nothing


def test_model_on_all_zero_input_gives_finite_outputs():
    model = build_model(make_cfg(*mel_overrides()), 3).eval()
    batch = pad_batch([np.zeros(SR, dtype=np.float32), np.zeros(SR // 2, dtype=np.float32)])
    with torch.no_grad():
        out = model(batch["waveforms"], batch["attention_mask"])
    assert torch.isfinite(out["logits"]).all() and torch.isfinite(out["embedding"]).all()


def test_silent_and_unreadable_clips_are_dropped_from_training(tmp_path, capture_logs):
    from src.config import AudioConfig

    sf.write(tmp_path / "ok.wav", tone(200, 1.0, SR), SR)
    sf.write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32), SR)
    sf.write(tmp_path / "short.wav", tone(200, 0.1, SR), SR)
    (tmp_path / "broken.wav").write_bytes(b"RIFF\x00\x00garbage")
    records = [
        Record(str(tmp_path / n), f"s{i}", "A")
        for i, n in enumerate(["ok.wav", "silent.wav", "short.wav", "broken.wav", "missing.wav"])
    ]
    with capture_logs() as logs:
        kept = verify_records(records, AudioConfig(), decode=True)
    assert [r.path.split("/")[-1] for r in kept] == ["ok.wav"]
    for reason in ("silent", "shorter than", "unreadable", "missing file"):
        assert reason in logs.text


# -------------------------------------------------------------------- stereo
def test_stereo_is_averaged_to_mono(tmp_path):
    left, right = tone(300, 1.0, SR), tone(500, 1.0, SR, amp=0.1)
    sf.write(tmp_path / "stereo.wav", np.stack([left, right], axis=1), SR)
    wave, info = load_audio(str(tmp_path / "stereo.wav"), SR)
    assert info.channels == 2 and wave.ndim == 1
    np.testing.assert_allclose(wave, (left + right) / 2, atol=1e-4)
    multi, _ = load_audio(str(tmp_path / "stereo.wav"), SR, mono=False)
    assert multi.shape == (2, SR)


# --------------------------------------------------------------- sample rate
@pytest.mark.parametrize("orig_sr", [44100, 48000, 22050, 8000])
def test_any_sample_rate_is_resampled_to_16k(tmp_path, orig_sr):
    sf.write(tmp_path / "x.wav", tone(1000, 1.5, orig_sr), orig_sr)
    wave, info = load_audio(str(tmp_path / "x.wav"), SR)
    assert info.sample_rate == orig_sr
    assert abs(len(wave) - int(1.5 * SR)) <= 1  # duration preserved
    assert peak_hz(wave, SR) == pytest.approx(1000, abs=5)  # pitch preserved


def test_resampling_removes_content_above_the_new_nyquist():
    wave = tone(12000, 1.0, 44100)  # above 8 kHz: must not alias into the 16 kHz signal
    out = resample(wave, 44100, SR)
    assert np.sqrt(np.mean(out**2)) < 0.01 * np.sqrt(np.mean(wave**2))


def test_mp3_and_flac_are_supported(tmp_path):
    clip = tone(440, 1.0, 44100)
    for ext in ("mp3", "flac"):
        sf.write(tmp_path / f"x.{ext}", np.stack([clip, clip], axis=1), 44100)
        wave, info = load_audio(str(tmp_path / f"x.{ext}"), SR)
        assert info.channels == 2 and peak_hz(wave, SR) == pytest.approx(440, abs=5)


# ---------------------------------------------------------- corrupted inputs
def test_corrupted_and_non_finite_files_raise_audio_error(tmp_path):
    (tmp_path / "broken.wav").write_bytes(b"RIFF....not audio")
    with pytest.raises(AudioError):
        load_audio(str(tmp_path / "broken.wav"))
    bad = tone(200, 0.5, SR)
    bad[100] = np.nan
    sf.write(tmp_path / "nan.wav", bad, SR, subtype="FLOAT")
    with pytest.raises(AudioError, match="NaN"):
        load_audio(str(tmp_path / "nan.wav"))
    with pytest.raises(FileNotFoundError):
        load_audio(str(tmp_path / "nope.wav"))


# ----------------------------------------------------------------------- CLI
def test_predict_reports_bad_files_and_keeps_going(tmp_path, tiny_phoneme_model, capsys):
    from src.predict import main

    sf.write(tmp_path / "speech.wav", tone(180, 1.5, 44100), 44100)  # 44.1 kHz input
    sf.write(tmp_path / "silent.wav", np.zeros(SR, dtype=np.float32), SR)
    sf.write(tmp_path / "short.wav", tone(180, 0.2, SR), SR)
    (tmp_path / "broken.wav").write_bytes(b"garbage")
    out = tmp_path / "report.json"
    code = main(
        [
            "--audio",
            str(tmp_path),
            "--text",
            "please call stella",
            "--phoneme_model",
            tiny_phoneme_model["path"],
            "--g2p",
            "lexicon",
            "--lexicon",
            tiny_phoneme_model["lexicon"],
            "--output",
            str(out),
        ]
    )
    assert code == 2  # some files failed, the rest were scored
    results = {r["file"].split("/")[-1]: r for r in json.loads(out.read_text())["results"]}
    assert "silent" in results["silent.wav"]["error"]
    assert "too short" in results["short.wav"]["error"]
    assert "AudioError" in results["broken.wav"]["error"]
    good = results["speech.wav"]
    assert good["error"] is None and good["sample_rate"] == 44100
    assert any("resampled" in w for w in good["warnings"])
    phones = [p for w in good["pronunciation"]["words"] for p in w["phones"]]
    assert phones and all(math.isfinite(p["score"]) for p in phones)
    assert "speech.wav" in capsys.readouterr().out  # console report
