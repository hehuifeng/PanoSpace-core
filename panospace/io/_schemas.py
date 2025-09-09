"""panospace.io._schemas
=======================
Internal column-schema definitions for the `SpatialData.tables` that
PanoSpace expects.

Why keep schemas centralised?
-----------------------------
*   **Define column names and dtypes in one place** - every reader/writer uses the same mapping, eliminating *magic strings* scattered across the code-base.
*   **Facilitate validation** - CI pipelines or users immediately see which required columns are missing or wrongly typed.
*   **Single point of evolution** - when a new field (e.g. `doublet_score`) is introduced you only change it here and all downstream components benefit automatically.

This module is **not** re-exported in ``__all__`` because end users typically
have no reason to access it directly.  When implementing a new adapter or
converter, import the schemas like so::

    from panospace.io._schemas import CELLS_SCHEMA
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Type

import pandas as pd

__all__: list[str] = [
    "CELLS_SCHEMA",
    "SPOTS_SCHEMA",
    "validate_table",
]


@dataclass(frozen=True)
class _TableSchema:
    """A simple column-name → dtype mapping with basic validation helpers."""

    required: Mapping[str, Type]
    optional: Mapping[str, Type] = field(default_factory=dict)

    def all_columns(self) -> Mapping[str, Type]:  # noqa: D401
        """Required ∪ optional mapping."""
        return {**self.required, **self.optional}


# -----------------------------------------------------------------------------
# Cells & Spots schemas
# -----------------------------------------------------------------------------

CELLS_SCHEMA = _TableSchema(
    required={
        "cell_id": str,
        "x": float,  # micron coordinate in CCS
        "y": float,
    },
    optional={
        "z": float,  # optional z-stack coordinate
        "cell_type": str,
        "morphotype": str,  # output of CellViT / StarDist categories
        "spot_id": str,  # Visium spot that encloses this cell, if any
    },
)

SPOTS_SCHEMA = _TableSchema(
    required={
        "spot_id": str,
        "x": float,
        "y": float,
    },
    optional={
        "array_row": int,
        "array_col": int,
        "in_tissue": bool,
        "platform": str,  # Visium, Xenium, etc.
    },
)


# -----------------------------------------------------------------------------
# Validation helper
# -----------------------------------------------------------------------------

def validate_table(df: pd.DataFrame, schema: _TableSchema, name: str | None = None) -> None:
    """Raise ``ValueError`` if *df* violates *schema*.

    Parameters
    ----------
    df
        The DataFrame to inspect.
    schema
        Column name → dtype expectations.
    name
        Optional human-readable table name used in error message.
    """
    _name = f" `{name}`" if name else ""

    missing: set[str] = set(schema.required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns{_name}: {sorted(missing)}")

    # Check dtypes for required columns only (optional ones may be absent)
    for col, expected in schema.required.items():
        if not isinstance(expected, tuple):
            expected_tuple = (expected,)
        else:
            expected_tuple = expected  # type: ignore[assignment]

        actual = df[col].dtype.type
        if not any(issubclass(actual, exp) for exp in expected_tuple):
            raise ValueError(
                f"Column '{col}' in table{_name} has dtype {actual}, "
                f"expected {expected_tuple}",
            )

    # No return (raises on first problem).
