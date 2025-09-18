# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict

import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, diags, vstack

import anndata as ad

from ...._utils.utils import radius_membership_sparse


# ---------------------------------------------------------------------
# 2) Utility: sparse per–cell-type means for scRNA
#     Returns: (K x G) CSR matrix aligned with the order in `celltype_list`
# ---------------------------------------------------------------------
def compute_celltype_means_sparse(
    sc_adata: ad.AnnData,
    celltype_list: List[str],
    celltype_column: str = "celltype_major",
) -> sp.csr_matrix:
    """Compute mean expression profiles for selected cell types using sparse ops.

    A sparse one-hot encoding in the requested order is built for the labels
    and multiplied with the expression matrix to accumulate sums per type.
    Division by per-type counts yields means. Types absent from the data keep
    zero rows after normalisation.

    Parameters
    ----------
    sc_adata
        Annotated single-cell matrix. ``sc_adata.X`` must be CSR or convertible.
    celltype_list
        Ordered list of cell-type names. The output row order follows this list.
    celltype_column
        Column in ``sc_adata.obs`` with the cell-type annotation.

    Returns
    -------
    scipy.sparse.csr_matrix
        Shape ``(len(celltype_list), n_genes)``. Row *k* stores the mean profile
        of ``celltype_list[k]``.

    Notes
    -----
    The computation is: ``means = diag(1/counts) @ (G.T @ X)`` where ``G`` is a
    sparse indicator matrix with one non-zero per cell.
    """
    # Sparse group mean: (onehot^T @ X) / counts
    labels = sc_adata.obs[celltype_column].astype("category")
    labels = labels.cat.set_categories(celltype_list)  # preserve requested order
    valid = labels.notna().to_numpy()
    X = sc_adata.X.tocsr() if not sp.isspmatrix_csr(sc_adata.X) else sc_adata.X
    X = X[valid]

    codes = labels[valid].cat.codes.to_numpy()  # 0..K-1
    K = len(celltype_list)
    rows = np.arange(X.shape[0], dtype=np.int64)
    G = sp.csr_matrix((np.ones_like(rows), (rows, codes)), shape=(X.shape[0], K))

    sum_by_type = G.T @ X  # (K x G)
    counts = np.asarray(G.sum(axis=0)).ravel().astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for missing types
    inv_counts = diags(1.0 / counts)
    means = inv_counts @ sum_by_type
    return means.tocsr()


# ---------------------------------------------------------------------
# 3) Graph construction: Delaunay + inverse distance; add self-loops; row-normalize
# ---------------------------------------------------------------------
def construct_graph_delaunay_inverse(coords: np.ndarray) -> sp.csr_matrix:
    """Build a row-stochastic affinity from 2-D coordinates via Delaunay graph.

    Undirected edges come from the triangulation. Each edge weight is the
    inverse Euclidean distance of the incident points. A self-loop is added
    to each node with weight equal to the mean of its incident weights (or 1
    if the node is isolated). The matrix is then row-normalized to obtain a
    random-walk transition matrix.

    Parameters
    ----------
    coords
        Array of shape ``(n, d)``. Only the first two dimensions are used.

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic sparse matrix with self-loops.

    Edge cases
    ----------
    - ``n == 0`` → empty ``0x0`` matrix.
    - ``n == 1`` → ``1x1`` identity.
    - A small constant (``1e-6``) is added to distances to ensure stability for
      coincident points.
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

    # Self-loops: mean incident weight per node (or 1 for isolated nodes)
    deg = np.asarray(W.sum(axis=1)).ravel()
    nnz_deg = np.maximum((W != 0).sum(axis=1).A.ravel(), 1)
    self_w = np.where(deg > 0, deg / nnz_deg, 1.0)
    W = W + diags(self_w.astype(np.float32))

    # Row-normalize → random-walk matrix D^{-1} W
    rowsum = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)
    rowsum[rowsum <= 1e-12] = 1.0
    W = diags(1.0 / rowsum) @ W
    return W.tocsr()


# ---------------------------------------------------------------------
# 4) Main class: sparse implementation; propagation formula preserved
# ---------------------------------------------------------------------
class GeneExpPredictor(object):
    """Predict gene expression for nuclei by diffusing spot-level signals.

    This class computes cell-type mean profiles from a single-cell reference,
    decomposes spot expression into type-specific contributions, and diffuses
    those signals to nuclei of the matching type. All heavy steps use sparse
    matrices to reduce memory and improve scalability.
    """
    def __init__(self, sc_adata: ad.AnnData, spot_adata: ad.AnnData, infered_adata: ad.AnnData):
        """Initialize with reference (scRNA), spots (ST), and nuclei to infer.

        Parameters
        ----------
        sc_adata
            Single-cell reference used to derive per–cell-type mean profiles.
        spot_adata
            Spatial transcriptomics data with expression in ``.X``, coordinates
            in ``.obsm['spatial']``, and a characteristic length scale
            ``spot_adata.uns['radius']`` for neighbourhood queries.
        infered_adata
            Nuclei to be inferred. Must include coordinates in
            ``.obsm['spatial']`` and a predicted type in ``.obs['pred_cell_type']``.
        """
        self.sc_adata = sc_adata
        self.spot_adata = spot_adata
        self.infered_adata = infered_adata

        # Align genes across objects (keep order consistent)
        self.spot_adata, self.sc_adata = self._find_common_genes(self.spot_adata, self.sc_adata)
        self.infered_adata = self.infered_adata[:, self.sc_adata.var_names].copy()

        # Placeholders
        self.cell_type_means = None  # (K x G) CSR
        self.ct_spot_expr: Dict[str, sp.csr_matrix] = {}  # per-type spot expression (S x G)

    @staticmethod
    def _find_common_genes(adata1: ad.AnnData, adata2: ad.AnnData):
        """Restrict two AnnData objects to their shared genes and return copies."""
        common_genes = adata1.var_names.intersection(adata2.var_names)
        return adata1[:, common_genes].copy(), adata2[:, common_genes].copy()

    # ------------------------------
    # 2') Type-specific spot expression without forming dense 3-D tensors
    #     For each non-zero (spot i, gene j), split Y[i,j] across types
    #     proportionally to beta_i(k) * mu_k(j) / sum_t beta_i(t)*mu_t(j)
    # ------------------------------
    def ctspecific_spot_gene_exp(self, celltype_list: List[str], celltype_column: str = "celltype_major"):
        """Compute per–cell-type contributions to each spot's gene expression.

        For each requested type, the method redistributes the non-zero entries
        of the spot expression matrix according to the product of the spot's
        mixture proportions and the type mean profile. The result for each type
        is a sparse matrix with the same sparsity pattern as the original spot
        matrix. Rows are then library-normalized (add 1e-3), scaled (1e4), and
        transformed with ``log1p`` on non-zeros.

        Parameters
        ----------
        celltype_list
            Ordered list of cell types. The same order is used to read mixture
            proportions from ``spot_adata.obs``.
        celltype_column
            Column in ``sc_adata.obs`` used for computing type means.
        """
        # Per-type means (K x G) in sparse; convert to small dense for K, G access
        self.cell_type_means = compute_celltype_means_sparse(self.sc_adata, celltype_list, celltype_column)
        mu = self.cell_type_means.toarray()  # K x G (K is usually small)

        # beta: S x K, read from spot_adata.obs with columns named by types
        beta = np.vstack([self.spot_adata.obs[ct].to_numpy() for ct in celltype_list]).T  # S x K
        beta = beta.astype(np.float64, copy=False)

        # Y: S x G, keep CSR
        Y = self.spot_adata.X.tocsr() if not sp.isspmatrix_csr(self.spot_adata.X) else self.spot_adata.X
        S, G = Y.shape
        K = len(celltype_list)

        # Pre-fetch CSR structure
        indptr, indices, data = Y.indptr, Y.indices, Y.data

        # For each cell type, build a sparse matrix sharing Y's sparsity pattern
        # Weight: w_k(i,j) = beta[i,k] * mu[k,j] / denom(i,j)
        # denom(i,j) = sum_t beta[i,t] * mu[t,j]
        mu_T = mu.T  # G x K
        beta_rows = beta  # S x K

        for k, ct in enumerate(celltype_list):
            data_k = np.empty_like(data, dtype=np.float64)
            write_pos = 0
            for i in range(S):
                start, end = indptr[i], indptr[i + 1]
                if start == end:
                    continue
                beta_i = beta_rows[i, :]  # (K,)
                js = indices[start:end]
                ys = data[start:end]
                # denom(i,j) = beta_i @ mu[:, j]
                for local, j in enumerate(js):
                    denom = float(beta_i @ mu_T[j])
                    if denom <= 0.0:
                        w = 0.0
                    else:
                        w = (beta_i[k] * mu[k, j]) / denom
                    data_k[write_pos + local] = ys[local] * w
                write_pos = end

            # Row normalisation (+1e-3), scale, log1p on non-zeros
            ct_sp = csr_matrix((data_k, indices.copy(), indptr.copy()), shape=(S, G))
            row_sum = np.asarray(ct_sp.sum(axis=1)).ravel() + 1e-3
            inv_row = 1.0 / row_sum
            ct_sp = diags(inv_row) @ ct_sp
            ct_sp = ct_sp.multiply(1e4)
            ct_sp.data = np.log1p(ct_sp.data)
            self.ct_spot_expr[ct] = ct_sp.tocsr()

    # ------------------------------
    # 3') Concatenate multiple AnnData objects (sparse)
    # ------------------------------
    @staticmethod
    def concat(ada_list: List[ad.AnnData]) -> ad.AnnData:
        """Stack a list of AnnData objects along observations.

        Assumes all inputs share the same ``var_names`` and contain coordinates
        in ``.obsm['spatial']``. The resulting object concatenates rows,
        preserves gene names, merges coordinates, and keeps ``pred_cell_type``.
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
    # 4') Inference: Delaunay + inverse distance; sparse diffusion
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
        """Diffuse type-specific spot signals to nuclei of the same type.

        Workflow per cell type:
        1) Mark nuclei as *labeled* if they have at least one spot neighbour
           within ``radius`` (via a sparse membership matrix), otherwise
           *unlabeled*.
        2) Initialize labeled nuclei by aggregating type-specific spot signals.
           Initialize unlabeled nuclei by averaging over a larger radius (4×).
        3) Build a random-walk matrix ``W`` on nuclei coordinates using
           Delaunay + inverse-distance weights + self-loops + row-normalization.
        4) Iterate:
             Y_u <- W_ul @ Y_l + W_uu @ Y_u
             Y_l <- alpha * F_l + (1 - alpha) * (W_ll @ Y_l + W_lu @ Y_u)
           where ``alpha = 1 / (1 + gamma)`` applies a soft constraint toward
           the labeled initialization.
        5) Stop early if the mean squared change on ``Y_u`` plateaus.

        Parameters
        ----------
        gamma
            Strength of the soft constraint toward the labeled initialization.
        iterations
            Maximum number of diffusion iterations.
        early_stop
            Enable early stopping based on the change of ``Y_u``.
        tol
            Required improvement to reset the patience counter.
        patience
            Allowed number of non-improving iterations when early stopping.
        return_w
            Also return the final ``W`` from the last processed cell type.
        return_ada
            Return a list of per-type AnnData instead of a single concatenation.

        Returns
        -------
        AnnData or tuple
            Concatenated predictions across processed cell types. If
            ``return_w`` or ``return_ada`` is ``True``, those are appended.

        Raises
        ------
        RuntimeError
            If ``ctspecific_spot_gene_exp`` has not been called.
        ValueError
            If ``spot_adata.uns['radius']`` is missing.

        Notes
        -----
        All large matrix operations are kept in sparse form.
        """
        if not self.ct_spot_expr:
            raise RuntimeError("Call ctspecific_spot_gene_exp(...) before inference.")

        ada_list: List[ad.AnnData] = []
        cell_types = list(self.ct_spot_expr.keys())

        spot_xy = np.asarray(self.spot_adata.obsm["spatial"])
        if "radius" not in self.spot_adata.uns:
            raise ValueError("spot_adata.uns['radius'] is not set.")
        base_r = float(self.spot_adata.uns["radius"])

        alpha = 1.0 / (1.0 + gamma)
        last_W = None

        for ct in cell_types:
            # Select nuclei of this type
            mask_ct = (self.infered_adata.obs["pred_cell_type"] == ct).to_numpy()
            if not np.any(mask_ct):
                continue
            ct_adata = self.infered_adata[mask_ct, :].copy()
            ct_xy = np.asarray(ct_adata.obsm["spatial"])

            # Neighbourhood: cells (queries) <- spots (bases), (n_cells x n_spots)
            affi = radius_membership_sparse(
                base_points=spot_xy,
                query_points=ct_xy,
                r=base_r,
                metric="euclidean",
                dtype=np.int8,
                sort_results=False,
            )
            # Labeled vs unlabeled split
            ind = np.asarray(affi.sum(axis=1)).ravel()
            non_zero_index = np.where(ind != 0)[0]  # labeled
            zero_index = np.where(ind == 0)[0]      # unlabeled

            # Place labeled first to match the block structure
            order = np.concatenate([non_zero_index, zero_index]) if zero_index.size else non_zero_index
            ct_adata_sorted = ct_adata[order].copy()
            ct_xy_sorted = ct_adata_sorted.obsm["spatial"]

            # Graph on nuclei (Delaunay + inverse distance + self-loops + row-norm)
            W = construct_graph_delaunay_inverse(ct_xy_sorted.astype(np.float64))
            last_W = W

            n_l = non_zero_index.size
            n_total = ct_adata_sorted.n_obs
            affi_sorted = affi[order, :]

            # F_l: aggregate labeled nuclei from spot memberships and per-type spots
            ct_sp = self.ct_spot_expr[ct]  # (S x G)
            F_l = affi_sorted[:n_l, :] @ ct_sp  # (n_l x G)

            # Y_u initialization: average over a larger radius (4x)
            if n_l < n_total:
                affi_u = radius_membership_sparse(
                    base_points=spot_xy,
                    query_points=ct_xy_sorted[n_l:, :],
                    r=base_r * 4.0,
                    metric="euclidean",
                    dtype=np.int8,
                    sort_results=False,
                ).tocsr()
                rowsum_u = np.asarray(affi_u.sum(axis=1)).ravel().astype(np.float64)
                rowsum_u[rowsum_u == 0.0] = 1.0
                affi_u = diags(1.0 / rowsum_u) @ affi_u
                Y_u = affi_u @ ct_sp  # (n_u x G)
            else:
                Y_u = sp.csr_matrix((0, ct_sp.shape[1]))

            Y_l = F_l.copy()

            # Block partitions of W
            W_ll = W[:n_l, :n_l]
            W_lu = W[:n_l, n_l:]
            W_ul = W[n_l:, :n_l]
            W_uu = W[n_l:, n_l:]

            # Iterative diffusion
            best_diff = np.inf
            wait = 0
            for it in range(iterations):
                if Y_u.shape[0] > 0:
                    Y_u_new = (W_ul @ Y_l) + (W_uu @ Y_u)
                    diff = (Y_u_new - Y_u).power(2).sum() / max(1, Y_u.shape[0])
                else:
                    Y_u_new = Y_u
                    diff = 0.0

                Y_l_new = alpha * F_l + (1.0 - alpha) * ((W_ll @ Y_l) + (W_lu @ Y_u))

                if early_stop:
                    if diff < best_diff - tol:
                        best_diff = float(diff)
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break

                Y_u = Y_u_new
                Y_l = Y_l_new

            # Assemble AnnData for this type (labeled first)
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
# 5) Optional: standalone concat function
# ---------------------------------------------------------------------
def concat(ada_list: List[ad.AnnData]) -> ad.AnnData:
    """Concatenate AnnData objects produced outside the predictor.

    Assumes shared ``var_names`` and presence of ``.obsm['spatial']`` in inputs.
    Stacks ``.X``, concatenates observation names, merges coordinates, and
    preserves ``pred_cell_type``.
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
