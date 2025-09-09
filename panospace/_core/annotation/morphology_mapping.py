"""panospace._core.annotation.morphology_mapping
================================================
Backend utilities that translate **morphological nucleus classes** - produced
by CellViT, StarDist, HoVer-Net, etc. - into *gene-expression* cell-type
probabilities via an **Optimal Transport (OT)** formulation.

Algorithmic outline
-------------------
1.  For every Visium (or other platform) *spot* *j* we know
    - a vector **h**_j of counts for each morphological class *s* (derived from
      nucleus segmentation).
    - a vector **y**_j of counts for each gene-expression cell type *k*
      (estimated by spot-level deconvolution; see *EnDecon*).
2.  We build two matrices H (n_spots x S) and Y (n_spots x K) and compute a
    *cost matrix* **M**_{s,k} = 1 - cosine(H_:,s, Y_:,k).
3.  Solve the OT problem (Eq. 1 of the paper) to obtain Γ ∈ R^{SxK} that maps
    morphology classes → transcriptomic cell types.
4.  Normalise Γ row-stochastic ⇒ Γ̄, treat each row as
    P(cell-type k | morphology's).
5.  Apply Γ̄ to every detected cell based on its morphology label to obtain an
    initial probabilistic annotation - later refined/combined with
    deconvolution-based probabilities in the high-level wrapper.

Functions
~~~~~~~~~
``learn_morphology_mapping``
    Fit OT and return Γ̄ as a pandas DataFrame.
``annotate_from_morphology``
    Apply Γ̄ to a cell table and return a per-cell probability table ready for
    merging.

The heavy-duty OT solver is provided by *POT* (`pip install pot`).  If POT is
missing we fall back to a crude cosine-similarity arg-max.
"""
from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Dict, List
import logging
import time

import numpy as np
import pandas as pd


from panospace._core import register

logger = logging.getLogger("panospace._core.annotation.morphology")


# -----------------------------------------------------------------------------
# Register back-end
# -----------------------------------------------------------------------------

register("annotation", "morphology",)

__all__ = [
    "learn_morphology_mapping",
    "annotate_from_morphology",
]
