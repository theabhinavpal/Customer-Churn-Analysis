"""
data_loader.py
--------------
Thin wrapper around pandas I/O with basic logging and validation, so every
other module reads/writes the dataset the same way.
"""

import logging
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame, raising a clear error if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved {path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
