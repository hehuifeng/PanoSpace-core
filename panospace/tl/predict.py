"""panospace.tl.predict
=====================

"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData

from tl import _import_backend

logger = logging.getLogger("panospace.tl")

# -----------------------------------------------------------------------------
# Backend registry ----------------------------------------------------------------
# -----------------------------------------------------------------------------




def genexp_predictor(
    sc_adata: AnnData,
    spot_adata: AnnData,
    infered_adata: AnnData,
    celltype_list: list[str],
    celltype_column: str = "celltype_major",
    backend: str = "predictor",
) -> AnnData:
    """基于单细胞数据预测空间转录组的基因表达。

    Parameters
    ----------
    sc_adata
        包含单细胞RNA测序数据的 AnnData 对象。
    spot_adata
        包含空间转录组数据的 AnnData 对象。
    infered_adata
        包含推断空间信息的 AnnData 对象。
    celltype_list
        细胞类型列表，用于指定哪些细胞类型参与预测。
    celltype_column
        指定 `sc_adata.obs` 中包含细胞类型信息的列名。默认值为 'celltype_major'。
    backend
        指定用于预测的后端实现。当前仅支持 'predictor'。

    Returns
    -------
    AnnData
        返回一个包含预测基因表达数据的 AnnData 对象。

    Raises
    ------
    ValueError
        如果指定的后端名称不受支持。
    """
    start_time = time.time()
    logger.info(f"Starting gene expression prediction using backend '{backend}'")

    backend_func = _import_backend(backend)

    adata_pred = backend_func(
        sc_adata=sc_adata,
        spot_adata=spot_adata,
        infered_adata=infered_adata,
        celltype_list=celltype_list,
        celltype_column=celltype_column,
    )

    logger.info(f"Gene expression prediction completed in {time.time() - start_time:.2f} seconds.")
    return adata_pred
