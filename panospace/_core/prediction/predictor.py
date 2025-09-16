"""panospace._core.prediction.predictor
================================================

"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

from .predictor_backend.predictor_utils import GeneExpPredictor

logger = logging.getLogger(__name__)


def predictor_core(
    sc_adata: AnnData,
    spot_adata: AnnData,
    infered_adata: AnnData,
    celltype_list: list[str],
    celltype_column: str = "celltype_major",
) -> Tuple[AnnData, AnnData]:
    start_time = time.time()

    # 1) 初始化预测器
    predictor = GeneExpPredictor(sc_adata, spot_adata, infered_adata)

    # 2) 计算特异性表达
    predictor.ctspecific_spot_gene_exp(celltype_list, celltype_column)

    adata_pred = predictor.do_geneinfer(
        gamma=0.1,       # 保持和你原来一致
        iterations=20,   # 最大迭代次数
        tol=1e-4,        # 早停阈值
        patience=5       # 早停耐心
    )
    # 3) 日志记录
    logger.info(f"Predictor core completed in {time.time() - start_time:.2f} seconds.")
    return adata_pred