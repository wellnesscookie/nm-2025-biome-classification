"""Common utilities: paths, seeding, device selection, logging."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


# --- Project paths ---------------------------------------------------------

# src/utils.py -> src/ -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
MAPPING_FILE: Path = DATA_DIR / "mapping"
SPLITS_DIR: Path = PROJECT_ROOT / "splits"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR: Path = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
LOGS_DIR: Path = ARTIFACTS_DIR / "logs"

for _d in (SPLITS_DIR, ARTIFACTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Reproducibility -------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy, and torch (CPU + CUDA + MPS) for reproducibility.

    Full determinism is not guaranteed on GPU/MPS backends, but this makes
    runs reproducible enough for reporting.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Device ----------------------------------------------------------------

def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Order: explicit `prefer` (if available) -> cuda -> mps -> cpu.
    """
    if prefer:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- Logging ---------------------------------------------------------------

def get_logger(name: str = "biome", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger. Idempotent: safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                          datefmt="%H:%M:%S")
    )
    logger.addHandler(h)
    logger.propagate = False
    return logger
