from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix

from ._spatialDWLS_backend.spatialDWLS_utils import *

import logging


logger = logging.getLogger(__name__)


def annotate_cells_core(
    sc_adata: ad.AnnData,
    adata_vis: ad.AnnData,
    celltype_key: str,
    target_sum: int = 1e4,
    n_top_genes: int = 2000,
    flavor: str = 'seurat',
    n_neighbors: int = 15,
    n_pca: int = 10,
    resolution: float = 0.4,
    method: str = 'wilcoxon',
    n_genes: int = 100,

):
    adata_vis.raw = adata_vis.copy()  # 保留原始数据
    sc.pp.normalize_total(adata_vis, target_sum=target_sum)
    sc.pp.log1p(adata_vis)
    sc.pp.highly_variable_genes(adata_vis, n_top_genes=n_top_genes, flavor=flavor)
    adata_vis = adata_vis[:, adata_vis.var.highly_variable]

    sc.pp.pca(adata_vis, svd_solver='arpack')
    sc.pp.neighbors(adata_vis, n_neighbors=n_neighbors, n_pcs=n_pca)
    sc.tl.umap(adata_vis)
    sc.tl.leiden(adata_vis, resolution=resolution)

    sc_adata.raw = sc_adata.copy()
    sc.pp.normalize_total(sc_adata, target_sum=target_sum)
    sc.pp.log1p(sc_adata)
    sc.pp.highly_variable_genes(sc_adata, n_top_genes=n_top_genes, flavor=flavor)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]

    sc.pp.pca(sc_adata, svd_solver='arpack')

    # 加入 celltype_final 注释
    sc_adata.obs["leiden_clus"] = sc_adata.obs[celltype_key].astype(str)
    sc.tl.rank_genes_groups(
        sc_adata,
        groupby='leiden_clus',
        method=method,  # Giotto 用 scran，这里用 wilcoxon 近似
        n_genes=n_genes
    )

    # 提取 marker 基因
    marker_df = pd.DataFrame({
        group: sc_adata.uns['rank_genes_groups']['names'][group]
        for group in sc_adata.uns['rank_genes_groups']['names'].dtype.names
    })
    Sig_scran = marker_df.melt()['value'].dropna().unique()

    norm_exp = np.expm1(sc_adata.X.toarray())
    Sig_exp = []

    # 按细胞类型取均值
    for cluster in sc_adata.obs['leiden_clus'].unique():
        cluster_mask = sc_adata.obs['leiden_clus'] == cluster
        cluster_mean = norm_exp[cluster_mask,:].mean(axis=0)
        Sig_exp.append(cluster_mean)

    Sig_exp = np.array(Sig_exp).T  # shape: genes x clusters
    Sig_exp_df = pd.DataFrame(Sig_exp,
                            index=sc_adata.var_names,
                            columns=sc_adata.obs['leiden_clus'].unique())

    
    # Sig_exp_df = Sig_exp_df.loc[Sig_exp_df.index.isin(Sig_scran)] # 只保留 marker 基因
    expr_df = pd.DataFrame(
        adata_vis.raw.X.A if hasattr(adata_vis.raw.X, "A") else adata_vis.raw.X,  # 转换稀疏矩阵
        index=adata_vis.raw.obs_names,
        columns=adata_vis.raw.var_names
    ).T
    log_expr_df = adata_vis.to_df().T

    results = runDWLSDeconv(expr_df=expr_df, log_expr_df=log_expr_df, cluster_info=adata_vis.obs['leiden'].tolist(), ct_exp_df=Sig_exp_df)
    return results

