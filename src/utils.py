"""
utils.py
--------
Small shared helpers used across modeling and evaluation scripts.
"""

import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_object(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved object to {path}")


def load_object(path: Path):
    if not Path(path).exists():
        raise FileNotFoundError(f"No object found at {path}")
    return joblib.load(path)


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"
