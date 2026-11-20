"""Seeding, device selection, logging, JSON and checkpoint helpers."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np
import torch

LOGGER_NAME = "accent"


# ------------------------------------------------------------------------- seeding
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by third-party code
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: derive NumPy/Python seeds from the torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG used by third-party code
    random.seed(seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_rng_state() -> dict[str, Any]:
    """RNG state as plain Python / tensor objects (loadable with ``weights_only=True``)."""
    np_state = np.random.get_state()  # noqa: NPY002 - legacy global RNG used by third-party code
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(_to_tuple(state["python"]))
    name, keys, *rest = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), *rest))  # noqa: NPY002 - legacy global RNG used by third-party code
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _to_tuple(obj: Any) -> Any:
    return tuple(_to_tuple(o) for o in obj) if isinstance(obj, (list, tuple)) else obj


# -------------------------------------------------------------------------- device
def get_device(prefer: str = "auto") -> torch.device:
    """``auto`` picks CUDA, then Apple MPS, then CPU. An unavailable explicit
    device logs a warning and falls back instead of crashing later."""
    log = logging.getLogger(LOGGER_NAME)
    prefer = (prefer or "auto").lower()
    mps = getattr(torch.backends, "mps", None)
    mps_ok = mps is not None and mps.is_available()
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps") if mps_ok else torch.device("cpu")
    if prefer.startswith("cuda"):
        index = int(prefer.split(":")[1]) if ":" in prefer else 0
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return torch.device(prefer)
        log.warning(f"device '{prefer}' requested but not available; falling back.")
        return get_device("auto")
    if prefer == "mps":
        if mps_ok:
            return torch.device("mps")
        log.warning("device 'mps' requested but not available; falling back.")
        return get_device("auto")
    return torch.device("cpu")


# ------------------------------------------------------------------------- logging
def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def get_logger(
    name: str = LOGGER_NAME, log_file: str | None = None, stream: Any = None
) -> logging.Logger:
    """Console (+ optional file) logger. Idempotent: calling again with a new
    ``log_file`` adds that file. ``stream`` defaults to stderr so that CLIs can
    keep stdout for data (JSON)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in logger.handlers):
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if log_file:
        path = os.path.abspath(log_file)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == path for h in logger.handlers
        ):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------- JSON
def jsonable(obj: Any) -> Any:
    """Convert NumPy / torch scalars and arrays (and NaN/inf) into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------- checkpoints
def save_checkpoint(obj: dict[str, Any], path: str) -> None:
    """Atomic save: write to a temp file, then rename (never a half-written checkpoint)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: Any = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by this project (``weights_only=True``: no pickled code)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # corrupted / truncated / foreign file
        raise ValueError(f"Could not read checkpoint {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(ckpt, dict) or "model" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{path} is not a checkpoint written by this project.")
    return ckpt


def quiet_transformers() -> None:
    """Silence Hugging Face progress bars and info logs so CLI output stays readable."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        import transformers

        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
    except ImportError:  # transformers is only needed for the hf backbone and GOP
        return


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
