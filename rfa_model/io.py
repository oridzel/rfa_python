"""
io.py

Saving and loading field dictionaries and result tables.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def load_field_npz(path: str | Path) -> dict:
    """
    Load an RFA field dictionary saved as .npz.

    Handles normal arrays plus object/dict entries such as voltages.
    """
    path = Path(path)

    data = np.load(path, allow_pickle=True)

    field = {}

    for key in data.files:
        value = data[key]

        # Convert 0-d object arrays back to Python objects/dicts.
        if value.shape == () and value.dtype == object:
            value = value.item()

        field[key] = value

    return field


def save_field_npz(field: dict, path: str | Path) -> Path:
    """
    Save an RFA field dictionary as .npz.

    This is best for arrays and small metadata dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(path, **field)

    return path