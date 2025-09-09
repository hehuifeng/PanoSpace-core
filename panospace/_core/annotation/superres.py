from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

from ._superres_backend.superres_utils import *

import logging
from panospace._core import register

logger = logging.getLogger(__name__)



def superres_core(
    cells: ad.AnnData,
    adata_vis: ad.AnnData,
    img_dir: str,
    neighb: int=3,
    radius: int=129,
    num_classes: int=9
):