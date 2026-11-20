"""Accent + pronunciation report for audio files or whole folders.

    python -m src.predict --audio clip.wav --text "Please call Stella" \\
        --accent-checkpoint runs/default/best.pt --output report.json
    python -m src.predict --audio recordings/ -r --text-file transcripts.tsv --output -

Per file: decode (wav / mp3 / flac / ogg, any sample rate, any channel count)
-> 16 kHz mono, then
  accent  trim silence -> overlapping sections -> classifier -> temperature-
          calibrated probabilities, section agreement, "contested" verdict
  GOP     G2P in the phoneme model's own vocabulary -> CTC posteriors -> forced
          alignment -> phone / word / utterance scores with timestamps -> flags

The JSON report goes to ``--output`` (``-`` = stdout); a readable summary goes
to the console (stderr when stdout carries the JSON). Each part is optional:
without ``--accent-checkpoint`` only GOP runs, without a transcript only the
accent. Exit code 0 = every file scored, 2 = some files failed, 1 = fatal error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, TextIO

import numpy as np

if __package__ in (None, ""):  # executed as `python src/predict.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "src"
    __import__("src")

from .audio import AudioError, collect_audio_files, load_audio, resample, rms_dbfs, trim_silence
from .config import Config, add_config_arguments, config_from_args
from .utils import LOGGER_NAME, get_device, get_logger, jsonable, quiet_transformers

log = logging.getLogger(LOGGER_NAME)


# --------------------------------------------------------------------------- inputs
def read_transcripts(path: str) -> dict[str, str]:
    """TSV ``file<TAB>transcript``; keys match by relative path or by base name."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Transcript file not found: {path}")
    out: dict[str, str] = {}
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{line_no}: expected 'file<TAB>transcript'")
            name, text = line.rstrip("\n").split("\t", 1)
            out[os.path.normpath(name.strip())] = text.strip()
    return out


def transcript_for(path: str, text: str | None, table: dict[str, str] | None) -> str | None:
    if table:
        for key in (os.path.normpath(path), os.path.basename(path)):
            if key in table:
                return table[key]
        stem = os.path.splitext(os.path.basename(path))[0]
        return table.get(stem, text)
    return text


# ----------------------------------------------------------------------- per file
class Scorer:
    """Holds the loaded models; ``score(path, transcript)`` returns one file report."""

    def __init__(
        self,
        cfg: Config,
        accent_checkpoint: str | None,
        need_gop: bool,
    ):
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.accent = None
        self.audio_cfg = cfg.audio
        if accent_checkpoint:
            from .model import load_accent_model

            model, model_cfg, ckpt = load_accent_model(accent_checkpoint, self.device)
            self.accent = model
            self.accents = list(ckpt["accents"])
            self.temperature = float(ckpt.get("temperature", 1.0))
            self.audio_cfg = model_cfg.audio  # preprocessing must match training
            self.accent_cfg = model_cfg
            self.accent_cfg.eval = cfg.eval
        self.recognizer = self.g2p = None
        if need_gop:
            from .aligner import PhonemeRecognizer, build_g2p

            quiet_transformers()
            self.recognizer = PhonemeRecognizer(cfg.gop.phoneme_model, self.device)
            self.g2p = build_g2p(cfg.gop.g2p, cfg.gop.language, cfg.gop.lexicon)

    def score(self, path: str, transcript: str | None) -> dict[str, Any]:
        sr = self.audio_cfg.sample_rate
        report: dict[str, Any] = {"file": path, "error": None, "warnings": []}
        try:
            wave, info = load_audio(path, sr)
        except (AudioError, OSError) as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
            return report
        report.update(
            {
                "duration": round(info.duration, 3),
                "sample_rate": info.sample_rate,
                "channels": info.channels,
            }
        )
        if info.channels > 1:
            report["warnings"].append(f"{info.channels} channels averaged to mono")
        if info.sample_rate != sr:
            report["warnings"].append(f"resampled {info.sample_rate} Hz -> {sr} Hz")
        if info.duration < self.audio_cfg.min_duration:
            report["error"] = f"too short ({info.duration:.2f} s < {self.audio_cfg.min_duration} s)"
            return report
        if rms_dbfs(wave) < self.audio_cfg.silence_rms_db:
            report["error"] = f"silent input (level {rms_dbfs(wave):.0f} dBFS)"
            return report
        if self.accent is not None:
            report["accent"] = self._accent(wave)
        if self.recognizer is not None:
            if transcript:
                gop_wave = wave
                if self.recognizer.sample_rate != sr:
                    gop_wave = resample(wave, sr, self.recognizer.sample_rate)
                from .gop import assess_pronunciation

                report["pronunciation"] = assess_pronunciation(
                    gop_wave, transcript, self.recognizer, self.g2p, self.cfg.gop
                )
                if report["pronunciation"].get("error"):
                    report["warnings"].append(f"GOP: {report['pronunciation']['error']}")
            else:
                report["warnings"].append("no transcript: pronunciation not scored")
        return report

    def _accent(self, wave: np.ndarray) -> dict[str, Any]:
        from .evaluate import predict_waves

        cfg = self.audio_cfg
        if cfg.trim_silence:
            wave, _ = trim_silence(wave, cfg.sample_rate, cfg.trim_top_db)
        res = predict_waves(self.accent, [wave], self.accent_cfg, self.device, self.temperature)[0]
        probs = {a: round(float(p), 4) for a, p in zip(self.accents, res["probs"], strict=True)}
        ranked = dict(sorted(probs.items(), key=lambda kv: -kv[1]))
        label = next(iter(ranked))
        return {
            "label": label,
            "confidence": ranked[label],
            "probabilities": ranked,
            "segments": res["segments"],
            "agreement": round(res["agreement"], 3),
            "verdict": res["verdict"],
            "temperature": round(self.temperature, 3),
        }


# -------------------------------------------------------------------------- console
def format_report(report: dict[str, Any], threshold: float) -> str:
    lines = [f"\n{report['file']}"]
    if report.get("duration") is not None:
        lines[0] += (
            f"  {report['duration']:.1f} s  ({report['sample_rate'] / 1000:g} kHz, "
            f"{report['channels']} ch)"
        )
    if report.get("error"):
        lines.append(f"  ERROR  {report['error']}")
        return "\n".join(lines)
    acc = report.get("accent")
    if acc:
        others = ", ".join(f"{a} {p:.0%}" for a, p in list(acc["probabilities"].items())[1:3])
        lines.append(
            f"  Accent         {acc['label']} {acc['confidence']:.0%}  ({others})  "
            f"{acc['segments']} section(s), agreement {acc['agreement']:.0%} [{acc['verdict']}]"
        )
    pron = report.get("pronunciation")
    if pron and not pron.get("error"):
        lines.append(
            f"  Pronunciation  {pron['overall_score']:.1f} / 100  "
            f"({pron['num_flagged']} of {pron['num_phones']} phones below {threshold:.2f})"
        )
        for w in pron["words"]:
            phones = "  ".join(
                f"{p['phone']} {p['score']:.2f}{'!' if p['flagged'] else ''}" for p in w["phones"]
            )
            score = f"{w['score']:.2f}" if w["score"] is not None else "  - "
            lines.append(f"    {w['word']:<14} {score}   {phones}")
        for f in pron["flagged"]:
            heard = f"  heard as /{f['heard_as']}/" if f["heard_as"] != f["phone"] else ""
            lines.append(
                f"    flagged /{f['phone']}/ in '{f['word']}'  {f['start']:.2f}-{f['end']:.2f} s"
                f"  score {f['score']:.2f}{heard}"
            )
    for warning in report.get("warnings", []):
        lines.append(f"  note: {warning}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Accent and pronunciation (GOP) report for audio files or folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--audio", nargs="+", required=True, help="audio files and/or folders")
    ap.add_argument("-r", "--recursive", action="store_true", help="search folders recursively")
    ap.add_argument("--text", default=None, help="reference transcript (used for every file)")
    ap.add_argument("--text-file", default=None, help="TSV: file<TAB>transcript per file")
    ap.add_argument("--accent-checkpoint", default=None, help="best.pt from training")
    ap.add_argument("--output", "-o", default=None, help="JSON report path ('-' = stdout)")
    add_config_arguments(
        ap,
        priority=("gop", "eval", "audio", "model", "data", "train", "augment"),
        exclude={"audio", "text", "output", "recursive"},
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console: TextIO = sys.stderr if args.output == "-" else sys.stdout
    get_logger(stream=sys.stderr)
    try:
        cfg = config_from_args(args)
        files = collect_audio_files(args.audio, args.recursive)
        if not files:
            raise FileNotFoundError(f"No audio files found in {args.audio}")
        table = read_transcripts(args.text_file) if args.text_file else None
        need_gop = bool(args.text or table)
        if not args.accent_checkpoint and not need_gop:
            raise ValueError("Nothing to do: give --accent-checkpoint and/or --text/--text-file")
        scorer = Scorer(cfg, args.accent_checkpoint, need_gop)
    except (FileNotFoundError, ValueError, KeyError, ImportError, OSError) as exc:
        log.error(str(exc))
        return 1

    results = []
    for path in files:
        report = scorer.score(path, transcript_for(path, args.text, table))
        results.append(report)
        print(format_report(report, cfg.gop.flag_threshold), file=console)
    payload = {
        "models": {
            "accent_checkpoint": args.accent_checkpoint,
            "phoneme_model": cfg.gop.phoneme_model if need_gop else None,
            "g2p": cfg.gop.g2p if need_gop else None,
            "flag_threshold": cfg.gop.flag_threshold,
        },
        "results": results,
    }
    text = json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    elif args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        log.info(f"JSON report -> {args.output}")
    failed = sum(1 for r in results if r.get("error"))
    if failed:
        log.warning(f"{failed} of {len(results)} file(s) could not be scored")
    return 2 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
