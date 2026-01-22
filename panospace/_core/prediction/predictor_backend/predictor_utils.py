# -*- coding: utf-8 -*-
import numpy as np
from typing import List, Dict, Optional, Tuple

import pandas as pd
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, csc_matrix, diags, vstack

import anndata as ad

from ...._utils.utils import radius_membership_sparse


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def align_genes_copy(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """
    Subset two AnnData objects to their shared genes and return copies.
    """
    common = adata1.var_names.intersection(adata2.var_names)
    if len(common) == 0:
        raise ValueError("No overlapping genes between AnnData objects.")
    return adata1[:, common].copy(), adata2[:, common].copy()


def compute_celltype_means_sparse(
    sc_adata: ad.AnnData,
    celltype_list: List[str],
    celltype_column: str,
) -> csr_matrix:
    """
    Compute per-cell-type mean expression using sparse operations.

    Returns
    -------
    (K x G) CSR matrix, where K = number of cell types, G = number of genes.
    """
    labels = sc_adata.obs[celltype_column].astype("category")
    labels = labels.cat.set_categories(celltype_list)
    valid = labels.notna().to_numpy()

    X = sc_adata.X
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    X = X[valid]

    codes = labels[valid].cat.codes.to_numpy()
    K = len(celltype_list)

    rows = np.arange(X.shape[0], dtype=np.int64)
    Gmat = csr_matrix(
        (np.ones_like(rows), (rows, codes)),
        shape=(X.shape[0], K),
    )

    sums = Gmat.T @ X
    counts = np.asarray(Gmat.sum(axis=0)).ravel().astype(np.float64)
    counts[counts == 0.0] = 1.0

    means = diags(1.0 / counts) @ sums
    return means.tocsr()


def build_delaunay_graph(coords: np.ndarray) -> csc_matrix:
    """
    Construct a Delaunay-based adjacency matrix with inverse-distance weights
    and explicit self-loops.

    Returns a CSC matrix (not normalized).
    """
    n = coords.shape[0]
    if n == 0:
        return csc_matrix((0, 0), dtype=np.float32)
    if n == 1:
        return sp.eye(1, format="csc", dtype=np.float32)

    tri = Delaunay(coords)

    rows, cols, data = [], [], []
    incident_weights: Dict[int, List[float]] = {i: [] for i in range(n)}

    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = int(simplex[i]), int(simplex[j])
                if u == v:
                    continue
                d = float(np.linalg.norm(coords[u] - coords[v]))
                edges.add((u, v, d))

    for u, v, d in edges:
        w = 1.0 / (d + 1e-6)
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([w, w])
        incident_weights[u].append(w)
        incident_weights[v].append(w)

    for i in range(n):
        w = float(np.mean(incident_weights[i])) if incident_weights[i] else 1.0
        rows.append(i)
        cols.append(i)
        data.append(w)

    return csc_matrix(
        (np.asarray(data, dtype=np.float32), (rows, cols)),
        shape=(n, n),
    )


def concat_anndata_sparse(adata_list: List[ad.AnnData]) -> ad.AnnData:
    """
    Concatenate AnnData objects assuming identical variables.
    """
    X = vstack([a.X for a in adata_list], format="csr")
    out = ad.AnnData(X=X)
    out.var_names = adata_list[0].var_names
    out.obs_names = sum((list(a.obs_names) for a in adata_list), [])
    out.obsm["spatial"] = np.vstack([a.obsm["spatial"] for a in adata_list])
    out.obs["pred_cell_type"] = np.concatenate(
        [a.obs["pred_cell_type"].to_numpy() for a in adata_list]
    )
    return out


# ---------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------

class GeneExpPredictor:
    """
    Graph-based diffusion of spot-derived gene expression to nuclei.
    """

    def __init__(
        self,
        sc_adata: ad.AnnData,
        spot_adata: ad.AnnData,
        nucleus_adata: ad.AnnData,
    ):
        self.sc_adata = sc_adata
        self.spot_adata = spot_adata
        self.nucleus_adata = nucleus_adata

        self.spot_adata, self.sc_adata = align_genes_copy(
            self.spot_adata, self.sc_adata
        )

        self.celltype_means: Optional[pd.DataFrame] = None
        self.celltype_specific_spot_exp: Optional[np.ndarray] = None  # (K, S, G)

    # -----------------------------------------------------------------

    def compute_celltype_specific_spot_expression(
        self,
        celltype_list: List[str],
        celltype_column: str = "celltype_major",
    ):
        """
        Decompose spot expression into cell-type-specific components.

        Output tensor shape: (K, S, G).
        """
        Y = self.spot_adata.X  # (S, G)
        beta = self.spot_adata.obs[celltype_list].to_numpy().T  # (K, S)

        mu_sparse = compute_celltype_means_sparse(
            self.sc_adata, celltype_list, celltype_column
        )
        mu = mu_sparse.toarray()  # (K, G)

        self.celltype_means = pd.DataFrame(
            mu, index=celltype_list, columns=self.sc_adata.var_names
        )

        denom = beta.T @ mu  # (S, G)

        K, S = beta.shape
        G = mu.shape[1]
        out = np.zeros((K, S, G), dtype=np.float64)

        for k in range(K):
            num = (beta[k][:, None] * mu[k][None, :]) * Y
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k] = np.nan_to_num(num / denom)

        libsize = out.sum(axis=2, keepdims=True) + 1e-3
        out = np.log1p(out / libsize * 1e4)

        self.celltype_specific_spot_exp = out

    # -----------------------------------------------------------------

    def infer_expression(
        self,
        gamma: float = 0.1,
        iterations: int = 10,
        early_stop: bool = True,
        tol: float = 1e-4,
        patience: int = 5,
    ) -> ad.AnnData:
        """
        Diffuse cell-type-specific spot expression to nuclei coordinates.
        """
        if self.celltype_specific_spot_exp is None:
            raise RuntimeError("Run compute_celltype_specific_spot_expression first.")

        radius = float(self.spot_adata.uns["radius"])
        spot_xy = np.asarray(self.spot_adata.obsm["spatial"])
        celltypes = list(self.celltype_means.index)

        outputs: List[ad.AnnData] = []
        gamma_param = 1.0 / (1.0 + gamma)

        for k, ct in enumerate(celltypes):
            mask = self.nucleus_adata.obs["pred_cell_type"] == ct
            nuc = self.nucleus_adata[mask].copy()
            if nuc.n_obs == 0:
                continue

            xy = np.asarray(nuc.obsm["spatial"])
            affi = radius_membership_sparse(
                spot_xy, xy, r=radius, dtype=np.int8
            ).tocsr()

            hit = affi.getnnz(axis=1) > 0
            labeled = np.where(hit)[0]
            unlabeled = np.where(~hit)[0]

            order = np.concatenate([labeled, unlabeled])
            xy = xy[order]
            obs_names = nuc.obs_names[order]

            W = build_delaunay_graph(xy)
            d = np.asarray(W.sum(axis=1)).ravel()
            d[d == 0.0] = 1.0
            W = diags(1.0 / d) @ W

            ct_spot = self.celltype_specific_spot_exp[k]

            # Each labeled nucleus corresponds to exactly one spot
            spot_ids = affi[labeled].indices
            Y_l = csc_matrix(ct_spot[spot_ids])

            if unlabeled.size > 0:
                af = radius_membership_sparse(
                    spot_xy,
                    xy[len(labeled):],
                    r=radius * 4.0,
                    dtype=np.float64,
                ).tocsr()
                rs = np.asarray(af.sum(axis=1)).ravel()
                rs[rs == 0.0] = 1.0
                af = diags(1.0 / rs) @ af
                Y_u = af @ ct_spot
            else:
                Y_u = np.zeros((0, ct_spot.shape[1]))

            best = np.inf
            wait = 0

            for _ in range(iterations):
                Y_u_new = W[len(labeled):, :len(labeled)] @ Y_l + \
                          W[len(labeled):, len(labeled):] @ Y_u
                Y_l_new = gamma_param * Y_l + \
                          (1.0 - gamma_param) * (
                              W[:len(labeled), :len(labeled)] @ Y_l +
                              W[:len(labeled), len(labeled):] @ Y_u
                          )

                diff = np.mean((np.asarray(Y_u_new) - np.asarray(Y_u)) ** 2) \
                    if Y_u.shape[0] > 0 else 0.0

                if early_stop:
                    if diff < best - tol:
                        best = diff
                        wait = 0
                    else:
                        wait += 1
                        if wait >= patience:
                            break

                Y_l, Y_u = Y_l_new, np.asarray(Y_u_new)

            Y = vstack([Y_l, csc_matrix(Y_u)], format="csc")
            out = ad.AnnData(Y)
            out.var_names = self.sc_adata.var_names
            out.obs_names = obs_names
            out.obsm["spatial"] = xy
            out.obs["pred_cell_type"] = ct

            outputs.append(out)

        return concat_anndata_sparse(outputs)
