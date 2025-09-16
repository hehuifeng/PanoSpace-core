# -*- coding: utf-8 -*-
import numpy as np
import scanpy as sc  # 保留依赖（不强制使用 to_df）
from typing import Optional, Union, Literal, List, Dict
from collections import defaultdict

import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, csc_matrix, diags, vstack

from sklearn.neighbors import KDTree
import anndata as ad

from ...._utils.utils import radius_membership_sparse


# ---------------------------------------------------------------------
# 2) 工具：按类别（cell type）计算 scRNA 的类型均值（稀疏）
#    返回：顺序与 celltype_list 对齐的 (K x G) CSR 均值矩阵
# ---------------------------------------------------------------------
def compute_celltype_means_sparse(
    sc_adata: ad.AnnData,
    celltype_list: List[str],
    celltype_column: str = "celltype_major",
) -> sp.csr_matrix:
    """Compute mean expression profiles for each requested cell type.

    The implementation avoids materialising dense intermediate arrays by
    constructing a sparse one-hot encoding of the requested cell type ordering
    and using sparse matrix multiplications.  Missing cell types are handled by
    inserting empty rows whose values remain zero after normalisation.

    Parameters
    ----------
    sc_adata
        Annotated single-cell expression matrix containing the reference
        profiles. ``sc_adata.X`` is expected to be convertible to ``csr``.
    celltype_list
        Ordered list of cell-type names whose mean expression should be
        returned.  The order of the list defines the row order in the output
        matrix.
    celltype_column
        Observation column in ``sc_adata.obs`` that stores the categorical cell
        type annotation.  The column is coerced into a categorical series so
        that unused categories in ``celltype_list`` remain present.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix with shape ``(len(celltype_list), n_genes)`` where each
        row stores the mean expression of the respective cell type.

    Notes
    -----
    The mean expression is computed as ``G.T @ X`` where ``G`` is the sparse
    cell-type indicator matrix.  Rows belonging to cell types that are not
    present in ``sc_adata`` remain zero because their counts are set to one to
    avoid division-by-zero during normalisation.
    """
    # 用稀疏操作计算 group-mean： (onehot^T @ X) / counts
    labels = sc_adata.obs[celltype_column].astype("category")
    # 仅保留需要的类别顺序
    labels = labels.cat.set_categories(celltype_list)
    valid = labels.notna().to_numpy()
    X = sc_adata.X.tocsr() if not sp.isspmatrix_csr(sc_adata.X) else sc_adata.X
    X = X[valid]

    codes = labels[valid].cat.codes.to_numpy()  # 0..K-1
    K = len(celltype_list)
    rows = np.arange(X.shape[0], dtype=np.int64)
    G = sp.csr_matrix((np.ones_like(rows), (rows, codes)), shape=(X.shape[0], K))

    sum_by_type = G.T @ X  # (K x G), 稀疏
    counts = np.asarray(G.sum(axis=0)).ravel().astype(np.float64)  # (K,)
    counts[counts == 0] = 1.0
    inv_counts = diags(1.0 / counts)
    means = inv_counts @ sum_by_type  # (K x G) CSR
    return means.tocsr()


# ---------------------------------------------------------------------
# 3) 图构建：Delaunay + inverse 距离；添加自环；行归一化成随机游走
# ---------------------------------------------------------------------
def construct_graph_delaunay_inverse(coords: np.ndarray) -> sp.csr_matrix:
    """Construct a row-stochastic affinity matrix from 2-D coordinates.

    The function performs a Delaunay triangulation of the input coordinates to
    obtain an undirected neighbourhood graph.  Each undirected edge is assigned
    a symmetric weight equal to the inverse Euclidean distance between the two
    incident nodes.  Self-loops are added with weights equal to the mean weight
    of the node's incident edges (or ``1`` for isolated nodes) and the matrix is
    subsequently row-normalised to form a random-walk transition matrix.

    Parameters
    ----------
    coords
        ``(n, d)`` array of spatial coordinates.  Only the first two dimensions
        are used by ``scipy.spatial.Delaunay`` and ``n`` may be zero or one.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic sparse matrix describing the random-walk graph.  The
        matrix contains self-loops and is symmetric before the final
        normalisation.

    Notes
    -----
    Degenerate situations are handled explicitly: an empty input yields an
    empty matrix and a single coordinate produces a ``1x1`` identity matrix.
    A small ``1e-6`` constant is added to distances to avoid division by zero
    for coincident points.
    """
    if coords.shape[0] == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)
    if coords.shape[0] == 1:
        return sp.eye(1, dtype=np.float32, format="csr")

    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                dist = np.linalg.norm(coords[u] - coords[v])
                edges.add((u, v, dist))

    rows, cols, data = [], [], []
    n = coords.shape[0]
    for u, v, d in edges:
        w = 1.0 / (d + 1e-6)
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([w, w])

    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    # 添加自环（防止孤点 & 提升收敛稳定性），自环权重取邻边权均值（若无邻居则置1）
    deg = np.asarray(W.sum(axis=1)).ravel()
    self_w = np.where(deg > 0, deg / np.maximum((W != 0).sum(axis=1).A.ravel(), 1), 1.0)
    W = W + diags(self_w.astype(np.float32))

    # 行归一化：变为随机游走矩阵 D^{-1} W
    rowsum = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    rowsum[rowsum <= 1e-12] = 1.0
    W = diags(1.0 / rowsum) @ W
    return W.tocsr()


# ---------------------------------------------------------------------
# 4) 主类：稀疏实现 + 传播公式保持不变
# ---------------------------------------------------------------------
class GeneExpPredictor(object):
    """Sparse backend used to propagate reference expression to new cells.

    The class orchestrates the computation of cell-type mean profiles, derives
    spot-specific expression contributions and diffuses the expression to the
    nuclei that should be inferred.  All heavy computations operate on sparse
    matrices to keep the memory footprint manageable for large data sets.
    """
    def __init__(self, sc_adata: ad.AnnData, spot_adata: ad.AnnData, infered_adata: ad.AnnData):
        """Initialise the predictor with the reference and target data sets.

        Parameters
        ----------
        sc_adata
            Single-cell reference used to compute the mean expression profile
            for each cell type.
        spot_adata
            Spatial transcriptomics data providing the observed spot-level
            expression and the estimated cell-type mixture proportions.  The
            matrix is expected to expose the ``spatial`` embedding and a
            ``radius`` value inside ``.uns``.
        infered_adata
            AnnData object representing the nuclei for which expression should
            be inferred.  The object must include a ``spatial`` embedding and a
            ``pred_cell_type`` column in ``.obs``.
        """
        self.sc_adata = sc_adata
        self.spot_adata = spot_adata
        self.infered_adata = infered_adata

        # 基因交集（保持顺序的一致性）
        self.spot_adata, self.sc_adata = self._find_common_genes(self.spot_adata, self.sc_adata)
        self.infered_adata = self.infered_adata[:, self.sc_adata.var_names].copy()

        # 占位
        self.cell_type_means = None  # (K x G) CSR
        self.ct_spot_expr: Dict[str, sp.csr_matrix] = {}  # 每种类型在所有 spot 上的特异表达 (S x G)

    @staticmethod
    def _find_common_genes(adata1: ad.AnnData, adata2: ad.AnnData):
        """Return copies of the inputs restricted to their shared genes.

        Parameters
        ----------
        adata1, adata2
            Input AnnData objects that will be sliced to the intersection of
            their ``var_names``.

        Returns
        -------
        tuple of AnnData
            The sliced copies of ``adata1`` and ``adata2`` with aligned genes.
        """
        common_genes = adata1.var_names.intersection(adata2.var_names)
        return adata1[:, common_genes].copy(), adata2[:, common_genes].copy()

    # ------------------------------
    # 2') 计算“cell type 特异的 spot 表达”：避免三维张量
    #     数学与原式一致：把 Y[i,j] 按 beta_i(k)*mu_k(j) / sum_k 分摊给每个 k
    # ------------------------------
    def ctspecific_spot_gene_exp(self, celltype_list: List[str], celltype_column: str = "celltype_major"):
        """Compute type-specific spot expression contributions.

        Parameters
        ----------
        celltype_list
            Ordered list of cell types to process.  The same order is used when
            reading mixture proportions from ``spot_adata.obs``.
        celltype_column
            Name of the cell-type annotation column in ``sc_adata`` used to
            compute the mean expression per type.

        Notes
        -----
        The method mirrors the dense implementation from the original backend
        but keeps the intermediate matrices sparse.  For each cell type the
        method re-weights the observed spot expression according to the
        contribution that the type makes to a gene in a spot and stores the
        transformed expression matrices in :attr:`ct_spot_expr`.
        """
        # 计算 cell type 均值 (K x G) 稀疏
        self.cell_type_means = compute_celltype_means_sparse(self.sc_adata, celltype_list, celltype_column)
        mu = self.cell_type_means.toarray()  # K x G（K通常较小，密集化可接受）

        # beta：S x K，从 spot_adata.obs 读取（列名为各 cell type）
        beta = np.vstack([self.spot_adata.obs[ct].to_numpy() for ct in celltype_list]).T  # S x K (float)
        beta = beta.astype(np.float64, copy=False)

        # Y：S x G，保持稀疏 CSR
        Y = self.spot_adata.X.tocsr() if not sp.isspmatrix_csr(self.spot_adata.X) else self.spot_adata.X
        S, G = Y.shape
        K = len(celltype_list)

        # 预取结构
        indptr, indices, data = Y.indptr, Y.indices, Y.data

        # 为每个 cell type 构建一个稀疏矩阵，和 Y 共享稀疏模式（非零位置相同）
        # 权重：w_k(i,j) = beta[i,k] * mu[k,j] / denom(i,j)
        # denom(i,j) = sum_{t} beta[i,t]*mu[t,j]
        mu_T = mu.T  # G x K，便于按列取 j
        beta_rows = beta  # S x K

        # 可复用的缓冲数组，避免反复分配
        denom_buf = np.empty(1, dtype=np.float64)  # 占位，仅为接口一致
        for k, ct in enumerate(celltype_list):
            # 为该类型准备 data_k
            data_k = np.empty_like(data, dtype=np.float64)
            write_pos = 0
            for i in range(S):
                start, end = indptr[i], indptr[i + 1]
                if start == end:
                    continue
                # 对该行的每个非零 (i, j)
                beta_i = beta_rows[i, :]  # (K,)
                # 依次处理该行的列 j
                js = indices[start:end]
                ys = data[start:end]
                # 计算 denom(i,j) = beta_i @ mu[:, j]
                # 逐个 j 做点积（K 通常不大）
                for local, j in enumerate(js):
                    denom = float(beta_i @ mu_T[j])
                    if denom <= 0.0:
                        # 若 denom==0，说明 beta_i 和 mu[:,j] 至少有一个全0；该基因对该spot无法分配
                        # 则该类型在该 (i,j) 的分量记为 0
                        w = 0.0
                    else:
                        w = (beta_i[k] * mu[k, j]) / denom
                    data_k[write_pos + local] = ys[local] * w
                write_pos = end

            # 行归一 + 1e-3，*1e4，log1p（与原逻辑一致）
            ct_sp = csr_matrix((data_k, indices.copy(), indptr.copy()), shape=(S, G))
            row_sum = np.asarray(ct_sp.sum(axis=1)).ravel() + 1e-3
            inv_row = 1.0 / row_sum
            ct_sp = diags(inv_row) @ ct_sp
            ct_sp = ct_sp.multiply(1e4)
            # 稀疏 log1p：只作用在非零 data；零保持零
            ct_sp.data = np.log1p(ct_sp.data)
            self.ct_spot_expr[ct] = ct_sp.tocsr()

    # ------------------------------
    # 3') 合并多个 AnnData（稀疏）
    # ------------------------------
    @staticmethod
    def concat(ada_list: List[ad.AnnData]) -> ad.AnnData:
        """Concatenate a list of AnnData objects produced by the predictor.

        Parameters
        ----------
        ada_list
            Sequence of AnnData objects with consistent ``var_names`` and
            ``spatial`` embeddings.

        Returns
        -------
        AnnData
            New object that stacks the expression matrices, observation names
            and spatial coordinates while preserving the predicted cell types.
        """
        if len(ada_list) == 0:
            return ad.AnnData(X=sp.csr_matrix((0, 0)))
        X = vstack([a.X for a in ada_list], format="csr")
        out = ad.AnnData(X=X)
        out.var_names = ada_list[0].var_names
        out.obs_names = sum((list(a.obs_names) for a in ada_list), [])
        out.obsm["spatial"] = np.vstack([a.obsm["spatial"] for a in ada_list])
        out.obs["pred_cell_type"] = np.concatenate([a.obs["pred_cell_type"].to_numpy() for a in ada_list], axis=0)
        return out

    # ------------------------------
    # 4') 主推断：Delaunay + inverse；传播公式保持不变；全程稀疏
    # ------------------------------
    def do_geneinfer(
        self,
        gamma: float = 0.1,
        iterations: int = 10,
        early_stop: bool = True,
        tol: float = 1e-4,
        patience: int = 5,
        return_w: bool = False,
        return_ada: bool = False,
    ):
        """Diffuse spot-level signals to nuclei of the corresponding type.

        Parameters
        ----------
        gamma
            Weight of the soft constraint that keeps the inferred expression of
            labelled nuclei close to the direct aggregation from the spatial
            spots.
        iterations
            Maximum number of diffusion iterations performed per cell type.
        early_stop
            Whether to stop early when the change of the unlabelled predictions
            falls below ``tol`` for ``patience`` consecutive iterations.
        tol
            Minimum relative improvement of the mean squared change required to
            reset the patience counter during early stopping.
        patience
            Number of iterations that are tolerated without improvement when
            ``early_stop`` is enabled.
        return_w
            If ``True``, also return the final affinity matrix used for the
            last processed cell type.
        return_ada
            If ``True``, return a list of individual AnnData objects per cell
            type instead of concatenating them.

        Returns
        -------
        AnnData or tuple
            The concatenated AnnData object that stores the inferred expression
            for all processed cell types.  When ``return_w`` or ``return_ada``
            are requested, additional values are returned as described above.

        Notes
        -----
        The method follows the propagation scheme of the dense implementation:
        labelled nuclei are initialised from neighbouring spots and the
        remaining nuclei are iteratively updated using a random-walk smoothing
        matrix.  All matrix multiplications stay within sparse space for
        efficiency.
        """
        if not self.ct_spot_expr:
            raise RuntimeError("请先调用 ctspecific_spot_gene_exp(...) 计算类型特异的 spot 表达。")

        ada_list: List[ad.AnnData] = []
        cell_types = list(self.ct_spot_expr.keys())

        spot_xy = np.asarray(self.spot_adata.obsm["spatial"])
        if "radius" not in self.spot_adata.uns:
            raise ValueError("spot_adata.uns['radius'] 未设置。")
        base_r = float(self.spot_adata.uns["radius"])

        # 传播参数
        gamma_param = 1.0 / (1.0 + gamma)

        last_W = None  # 若需要返回 W

        for ct in cell_types:
            # 取该类型的细胞
            mask_ct = (self.infered_adata.obs["pred_cell_type"] == ct).to_numpy()
            if not np.any(mask_ct):
                # 没有该类型的推断细胞
                continue
            ct_adata = self.infered_adata[mask_ct, :].copy()
            ct_xy = np.asarray(ct_adata.obsm["spatial"])

            # 邻接：cells(查询) <- spots(基点)，形状 (n_cells x n_spots)
            affi = radius_membership_sparse(
                base_points=spot_xy,
                query_points=ct_xy,
                r=base_r,
                metric="euclidean",
                dtype=np.int8,
                sort_results=False,
            )
            # 以是否与任一spot相邻划分 labeled / unlabeled
            ind = np.asarray(affi.sum(axis=1)).ravel()
            non_zero_index = np.where(ind != 0)[0]  # labeled cells
            zero_index = np.where(ind == 0)[0]      # unlabeled cells

            # 将 labeled 放前以匹配原算法的块分解
            order = np.concatenate([non_zero_index, zero_index]) if zero_index.size else non_zero_index
            ct_adata_sorted = ct_adata[order].copy()
            ct_xy_sorted = ct_adata_sorted.obsm["spatial"]

            # 构图（Delaunay + inverse + 自环 + 行归一）
            W = construct_graph_delaunay_inverse(ct_xy_sorted.astype(np.float64))
            last_W = W  # 记录最后一次的 W（或可存每个ct的）

            n_l = non_zero_index.size
            n_total = ct_adata_sorted.n_obs
            # 重排后的 affi 也需要同步重排
            affi_sorted = affi[order, :]

            # F_l：用 labeled cells 与 spot 的从属矩阵乘以该类型的 spot 特异表达
            ct_sp = self.ct_spot_expr[ct]  # (S x G) CSR
            F_l = affi_sorted[:n_l, :] @ ct_sp  # (n_l x G) CSR

            # Y_u 初始化：用更大半径（4x）并行归一作为平均
            if n_l < n_total:
                affi_u = radius_membership_sparse(
                    base_points=spot_xy,
                    query_points=ct_xy_sorted[n_l:, :],
                    r=base_r * 4.0,
                    metric="euclidean",
                    dtype=np.int8,
                    sort_results=False,
                ).tocsr()
                # 行归一（norm=True 等价）
                rowsum_u = np.asarray(affi_u.sum(axis=1)).ravel().astype(np.float64)
                rowsum_u[rowsum_u == 0.0] = 1.0
                affi_u = diags(1.0 / rowsum_u) @ affi_u
                Y_u = affi_u @ ct_sp  # (n_u x G) CSR
            else:
                Y_u = sp.csr_matrix((0, ct_sp.shape[1]))

            Y_l = F_l.copy()

            # 拆分 W 的块
            W_ll = W[:n_l, :n_l]
            W_lu = W[:n_l, n_l:]
            W_ul = W[n_l:, :n_l]
            W_uu = W[n_l:, n_l:]

            # 迭代传播（保持原公式不变）
            best_diff = np.inf
            wait = 0
            for it in range(iterations):
                if Y_u.shape[0] > 0:
                    Y_u_new = (W_ul @ Y_l) + (W_uu @ Y_u)
                    # diff：均方变化（稀疏）
                    diff = (Y_u_new - Y_u).power(2).sum() / max(1, Y_u.shape[0])
                else:
                    Y_u_new = Y_u
                    diff = 0.0

                # Y_l 软约束回拉 + 平滑
                Y_l_new = gamma_param * F_l + (1.0 - gamma_param) * ((W_ll @ Y_l) + (W_lu @ Y_u))

                if early_stop:
                    if diff < best_diff - tol:
                        best_diff = float(diff)
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            # print(f"[Info] {ct}: Early stopped at iter {it}, diff={diff:.2e}")
                            break

                Y_u = Y_u_new
                Y_l = Y_l_new

            # 组装回该类型的 AnnData（顺序：labeled 在前）
            Y = vstack([Y_l, Y_u], format="csr") if Y_u.shape[0] > 0 else Y_l
            nuclei = ad.AnnData(X=Y)
            nuclei.var_names = self.sc_adata.var_names.copy()
            nuclei.obsm["spatial"] = ct_xy_sorted
            nuclei.obs_names = ct_adata_sorted.obs_names
            nuclei.obs["pred_cell_type"] = ct
            ada_list.append(nuclei)

        if return_ada:
            return ada_list

        adata_all = self.concat(ada_list)
        if return_w:
            return adata_all, last_W
        return adata_all


# ---------------------------------------------------------------------
# 5)（可选）一个独立的 concat 函数（若你需要在类外使用）
# ---------------------------------------------------------------------
def concat(ada_list: List[ad.AnnData]) -> ad.AnnData:
    """Concatenate AnnData objects produced outside the predictor class.

    Parameters
    ----------
    ada_list
        Sequence of AnnData objects that share ``var_names`` and contain spatial
        coordinates in ``.obsm['spatial']``.

    Returns
    -------
    AnnData
        New AnnData instance with stacked expression matrix, concatenated
        observation names and preserved ``pred_cell_type`` annotations.
    """
    if len(ada_list) == 0:
        return ad.AnnData(X=sp.csr_matrix((0, 0)))
    X = vstack([a.X for a in ada_list], format="csr")
    out = ad.AnnData(X=X)
    out.var_names = ada_list[0].var_names
    out.obs_names = sum((list(a.obs_names) for a in ada_list), [])
    out.obsm["spatial"] = np.vstack([a.obsm["spatial"] for a in ada_list])
    out.obs["pred_cell_type"] = np.concatenate([a.obs["pred_cell_type"].to_numpy() for a in ada_list], axis=0)
    return out
