import numpy as np
import scipy.sparse as sp
import scipy
import ot  # POT for OT (EMD / Sinkhorn)
from qpsolvers import solve_qp  # unified QP interface (supports osqp/cvxopt/...)
try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except Exception:
    _GUROBI_AVAILABLE = False

import numpy as np

from ...._utils.utils import radius_membership_sparse

class CellTypeAnnotator:
    """
    CellTypeAnnotator

    Assigns a cell type to each segmented unit (segment/nucleus) by combining:
      • deconvolved cell-type proportions at spot level and super-resolved spot level,
      • spatial memberships between spots/SR-spots and segments,
      • optional morphology priors from image-derived categories, and
      • a global assignment solved via Optimal Transport (OT) followed by either
        a convex relaxation (QP + rounding) or an exact MILP when available.

    The result is a globally consistent mapping from segments to cell types that
    respects local proportions and optional morphology preferences.
    """

    def __init__(
        self,
        spot_adata,
        sr_spot_adata,
        seg_adata,
        priori_type_affinities=None,
        alpha=0.3,
        ot_mode="emd",          # "sinkhorn" or "emd"
        sinkhorn_reg=0.01,      # Sinkhorn entropy regularization
        qp_solver: str = "osqp", # qpsolvers backend: "osqp"|"cvxopt"|...
        use_mip: bool = False   # If True and Gurobi is available, use MILP for 0/1 assignment
    ):
        """
        Parameters
        ----------
        spot_adata : AnnData
            Spot-level deconvolution. Requirements:
            - .uns['celltype'] : list of cell type names
            - .uns['radius']   : spot radius
            - .obsm['spatial'] : spot coordinates (S x 2/3)
            - .obs[ctype]      : a column for each cell type with its proportion
        sr_spot_adata : AnnData
            Super-resolved spot-level deconvolution. Same columns as above; .obsm['spatial'] required.
        seg_adata : AnnData
            Segmented units (segments or nuclei). Requires .obsm['spatial'].
            Optionally .obs['img_type'] with integer morphology labels enables the morphology branch ("mor" mode).
        priori_type_affinities : dict[str, list[str]] or None
            Optional prior: morphology-category name → preferred cell types.
        alpha : float
            Fusion weight. When morphology is active, blends (SR spatial propagation) with (morphology prior via OT).
        ot_mode : str
            "sinkhorn" for fast/stable entropic OT or "emd" for exact discrete OT.
        sinkhorn_reg : float
            Entropic regularization for Sinkhorn.
        qp_solver : str
            Backend name for qpsolvers ("osqp", "cvxopt", "quadprog", ...).
        use_mip : bool
            If True and Gurobi is available, solve the final 0/1 assignment via MILP; otherwise use QP+rounding.
        """
        self.spot_adata = spot_adata
        self.sr_spot_adata = sr_spot_adata
        self.seg_adata = seg_adata
        self.alpha = float(alpha)
        self.ot_mode = ot_mode
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.qp_solver = qp_solver
        self.use_mip = bool(use_mip)

        # Mode: enable morphology branch if seg_adata has img_type labels
        self.mode = 'mor' if 'img_type' in self.seg_adata.obs.columns else None

        # Cell types and column names
        self.cell_types = list(self.spot_adata.uns['celltype'])
        self.ct_cols = self.cell_types

        # Super-resolved spot proportions (S_sr x K), row-normalized
        self.sr_ct_ratios = self._safe_row_normalize(
            self.sr_spot_adata.obs[self.ct_cols].to_numpy(copy=True)
        )

        # Spot proportions (S x K), clipped to nonnegative then row-normalized
        self.spot_ct_ratios = self._safe_row_normalize(
            np.clip(self.spot_adata.obs[self.ct_cols].to_numpy(copy=True), 0, None)
        )

        # Spatial radius
        self.radius = float(self.spot_adata.uns['radius'])

        # Prior morphology→cell-type affinities
        self.priori_type_affinities = priori_type_affinities or {}
        self.img_type_names = list(self.priori_type_affinities.keys()) if self.priori_type_affinities else []

        # Cache spatial coordinates (NumPy)
        self._sr_spatial = np.asarray(self.sr_spot_adata.obsm['spatial'])
        self._spot_spatial = np.asarray(self.spot_adata.obsm['spatial'])
        self._seg_spatial = np.asarray(self.seg_adata.obsm['spatial'])

        # Sparse matrix caches (filled later)
        self._affil_sr2seg_csr = None        # (S_sr x Nseg)
        self._affil_spot2seg_csr = None      # (S x Nseg)
        self._affil_sr2seg_norm_csr = None   # row-normalized SR-spot→segment
        self._affil_sr2seg_csc = None        # optional transpose/format cache
        self._affil_spot2seg_csr_T = None    # transpose cache if needed

        # Morphology one-hot and aggregates
        self._imgtype_keep_labels = None
        # self._seg_imgtype_labels = None    # shape (Nseg,)
        self._seg_imgtype_onehot_csr = None  # (Nseg x G_kept)
        self._mortype_in_spot = None         # (G_kept x S_nonzero)
        self._imgtype_ratio = None           # (G_kept, ) global normalized vector

        # Counts and quotas
        self._cell_counts_per_spot = None    # (S, )
        self._int_ct_ratios_per_spot = None  # (S_nonzero x K) integerized per-spot quotas
        self._global_ct_quota = None         # (K, ) global integer quota across all segments
        self._nonzero_spot_mask = None       # (S, ) bool

    # ---------- Utilities ----------

    @staticmethod
    def _safe_row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x[x < 0] = 0.0
        s = x.sum(axis=1, keepdims=True)
        s[s < eps] = 1.0
        return x / s

    @staticmethod
    def _integerize_proportions(p: np.ndarray, total: int) -> np.ndarray:
        """Discretize a proportion vector p into integers that sum exactly to `total` (auto-normalizes and handles zeros)."""
        p = np.maximum(np.asarray(p, dtype=float), 0.0)
        if p.sum() <= 0:
            p = np.full_like(p, 1.0 / len(p))
        else:
            p = p / p.sum()
        raw = p * total
        flo = np.floor(raw).astype(int)
        deficit = int(total - flo.sum())
        if deficit > 0:
            resid = raw - flo
            idx = np.argpartition(-resid, deficit - 1)[:deficit]
            flo[idx] += 1
        return flo

    # ---------- Preprocessing: filtering & affiliation matrices ----------

    def filter_and_build_affiliations(self):
        """
        Steps:
        1) Use SR-spot coverage to filter segments (batch inclusion).
        2) Build SR-spot→segment and spot→segment sparse membership matrices (CSR/CSC).
        3) Row-normalize SR-spot→segment for propagating SR proportions.
        """
        # Initial SR-spot coverage for filtering
        affil_sr2seg = radius_membership_sparse(
            base_points = self._sr_spatial,
            query_points = self._seg_spatial,
            r=self.radius,
            metric = 'chebyshev'
        )
        # Keep segments covered by at least one SR-spot
        keep_mask_seg = (affil_sr2seg.getnnz(axis=1) != 0)
        self.seg_adata = self.seg_adata[keep_mask_seg].copy()
        self._seg_spatial = np.asarray(self.seg_adata.obsm['spatial'])

        # Recompute memberships after filtering; adopt the (S_* x Nseg) shape convention
        affil_sr2seg = radius_membership_sparse(
            self._sr_spatial,
            self._seg_spatial,
            r=self.radius,
            metric = 'chebyshev'
        ).transpose()  # (S_sr x Nseg)
        affil_spot2seg = radius_membership_sparse(
            self._spot_spatial,
            self._seg_spatial,
            r=self.radius,
            metric = 'euclidean'
        ).transpose()  # (S x Nseg)

        # Cache CSR
        self._affil_sr2seg_csr = affil_sr2seg.tocsr()      # (S_sr x Nseg)
        self._affil_spot2seg_csr = affil_spot2seg.tocsr()  # (S x Nseg)
        # self._affil_spot2seg_csr_T = self._affil_spot2seg_csr.transpose().tocsr()  # (Nseg x S)

        # Row-normalize SR-spot→segment
        A = self._affil_sr2seg_csr.copy()
        row_sums = A.sum(axis=1).A1
        inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums!=0)
        A.data *= inv.repeat(np.diff(A.indptr))
        self._affil_sr2seg_norm_csr = A  # (S_sr x Nseg), row-normalized

        # Build morphology one-hot and aggregates when available
        if self.mode == 'mor':
            self._build_imgtype_onehot_and_spot_aggregates()

    # ---------- Morphology one-hot & spot aggregates ----------

    def _build_imgtype_onehot_and_spot_aggregates(self):
        """
        Build segment-level morphology one-hot (keeping valid classes) and aggregate counts per spot.
        """
        # Mapping of integer labels to names; drop unlabeled and dead cells
        label_dict = {
            0: 'nolabel',
            1: 'Neoplastic cells',
            2: 'Inflammatory',
            3: 'Connective/Soft tissue cells',
            4: 'Dead Cells',
            5: 'Epithelial'
        }
        remove = {'nolabel', 'Dead Cells'}
        keep = [v for v in label_dict.values() if v not in remove]
        # Name → compact column index
        keep_name2col = {name: i for i, name in enumerate(keep)}
        # Original integer labels
        raw = np.asarray(self.seg_adata.obs['img_type']).astype(int)
        # Map to names
        mapped_names = np.vectorize(label_dict.get)(raw)
        valid_mask = np.array([name in keep_name2col for name in mapped_names], dtype=bool)
        # One-hot (Nseg x G_kept)
        rows = np.nonzero(valid_mask)[0]
        cols = np.fromiter((keep_name2col[n] for n in mapped_names[valid_mask]), count=rows.size, dtype=int)
        data = np.ones_like(rows, dtype=np.int32)
        nseg = self.seg_adata.n_obs
        gk = len(keep_name2col)
        onehot = sp.csr_matrix((data, (rows, cols)), shape=(nseg, gk))
        self._seg_imgtype_onehot_csr = onehot
        self._imgtype_keep_labels = keep
        # self._seg_imgtype_labels = raw  # optional cache

        # Aggregate to spots: (#S x #Nseg) @ (#Nseg x #G_kept) = (#S x #G_kept)
        spot_morph_counts = (self._affil_spot2seg_csr @ onehot).astype(np.float64)  # (S x G_kept)
        # Consider only spots containing at least one segment
        self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)
        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No spots contain any segments after filtering.")

        spot_morph_counts = spot_morph_counts[self._nonzero_spot_mask, :]  # (S_nz x G_kept)
        # Transpose to (G_kept x S_nz) to align with OT code
        self._mortype_in_spot = spot_morph_counts.T.toarray()  # dense for OT
        # Global morphology proportions (G_kept,)
        total = np.sum(self._mortype_in_spot)
        if total <= 0:
            # Degenerate case: no valid morphology; fall back to uniform
            self._imgtype_ratio = np.full(self._mortype_in_spot.shape[0], 1.0 / self._mortype_in_spot.shape[0])
        else:
            self._imgtype_ratio = self._mortype_in_spot.sum(axis=1) / total

    # ---------- Counts & integer quotas ----------

    def compute_counts_and_integerize(self):
        """
        Compute number of segments per spot; discretize spot-level cell-type proportions
        into integer per-spot quotas; and compute a global quota across all segments.
        """
        if self._affil_spot2seg_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() first.")

        # Number of segments per spot
        if self._cell_counts_per_spot is None:
            self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)

        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No non-empty spots found.")

        # Proportions and counts for non-empty spots
        spot_ratios_nz = self.spot_ct_ratios[self._nonzero_spot_mask, :]  # (S_nz x K)
        counts_nz = self._cell_counts_per_spot[self._nonzero_spot_mask]   # (S_nz, )

        # Integerize each spot's cell-type quota V (row sums are exact)
        int_ratios_list = [
            self._integerize_proportions(spot_ratios_nz[s, :], int(counts_nz[s]))
            for s in range(spot_ratios_nz.shape[0])
        ]
        self._int_ct_ratios_per_spot = np.vstack(int_ratios_list)  # (S_nz x K)

        # Global quotas: integerize based on aggregated per-spot counts
        total_segments = self.seg_adata.n_obs
        global_ratio = self._int_ct_ratios_per_spot.sum(axis=0).astype(float)
        if global_ratio.sum() <= 0:
            global_ratio = np.full_like(global_ratio, 1.0 / len(global_ratio))
        else:
            global_ratio = global_ratio / global_ratio.sum()
        self._global_ct_quota = self._integerize_proportions(global_ratio, total_segments)  # (K,)

    # ---------- OT: align cell types ↔ morphology categories ----------

    def build_type_transfer(self, factor: float = 2.0):
        """
        Align cell types and morphology categories via OT using their distributions across spots.

        Cost:
          • cosine distance between (K x S_nz) and (G x S_nz) signatures.
        Priors:
          • if priori_type_affinities is provided, down-weight preferred pairs by dividing cost by `factor`.
        Output:
          • self.type_transfer_prop: (K x G) column-normalized transport plan, interpreted as
            P(cell type | morphology category).
        """
        if self._mortype_in_spot is None and self.mode == 'mor':
            raise RuntimeError("Morphology aggregates not built. Call filter_and_build_affiliations() first.")

        # Signatures across spot space
        ct_signature = self._int_ct_ratios_per_spot.T.astype(float)  # (K x S_nz)
        morph_signature = self._mortype_in_spot.astype(float)        # (G x S_nz)

        # Cosine distance cost (K x G)
        cost = ot.dist(ct_signature, morph_signature, metric='cosine')

        # Apply prior affinities by reducing cost for preferred pairs
        if self.priori_type_affinities:
            adjusted = cost.copy()
            for g_idx, g_name in enumerate(self._imgtype_keep_labels):
                prefer_cts = self.priori_type_affinities.get(g_name, [])
                for ct_name in prefer_cts:
                    if ct_name in self.cell_types:
                        k = self.cell_types.index(ct_name)
                        adjusted[k, g_idx] /= float(factor)
            cost = adjusted

        # Source and target distributions
        ct_ratio_global = self._global_ct_quota.astype(float)
        if ct_ratio_global.sum() <= 0:
            ct_ratio_global = np.full_like(ct_ratio_global, 1.0 / len(ct_ratio_global))
        else:
            ct_ratio_global = ct_ratio_global / ct_ratio_global.sum()

        morph_ratio_global = self._imgtype_ratio

        # Solve OT
        if self.ot_mode == "sinkhorn":
            gamma = ot.sinkhorn(ct_ratio_global, morph_ratio_global, cost, reg=self.sinkhorn_reg, numItermax=2000)
        else:
            gamma = ot.emd(ct_ratio_global, morph_ratio_global, cost, numItermax=100000)

        # Column-normalize to obtain P(cell type | morphology)
        col_sums = gamma.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        self.type_transfer_prop = gamma / col_sums  # (K x G)

    # ---------- Final assignment: QP relaxation (optional exact MILP) ----------

    def infer_cell_types(self):
        """
        Produce the segment→cell-type assignment.

        Scoring:
          • SR-spot proportions propagated to segments: (_affil_sr2seg_norm_csr @ sr_ct_ratios).
          • If morphology is active: blend with (segment one-hot × OT transfer) using weight `alpha`.

        Optimization:
          • If `use_mip` and Gurobi available: solve MILP for exact 0/1 assignment with row=1, col=quota.
          • Else: solve a convex relaxation via QP, then round and repair to match global quotas.

        Returns
        -------
        seg_adata_pred : AnnData
            A copy of seg_adata with:
              - obs['pred_cell_type'] : predicted type
              - one-hot columns for each cell type
        """
        if self._affil_sr2seg_norm_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() and compute_counts_and_integerize() first.")

        # (1) Propagate SR-spot proportions to segments (Nseg x K)
        # Use the transpose of (S_sr x Nseg) to multiply (Nseg x S_sr) @ (S_sr x K)
        sr2seg_norm_T = self._affil_sr2seg_norm_csr.transpose().tocsr()  # (Nseg x S_sr)
        sr_scores = (sr2seg_norm_T @ sp.csr_matrix(self.sr_ct_ratios)).toarray()  # dense (Nseg x K)

        # (2) Morphology branch (if active)
        if self.mode == 'mor':
            # segment one-hot (Nseg x G) × (G x K); use (K x G)^T
            morph_scores = (self._seg_imgtype_onehot_csr @ sp.csr_matrix(self.type_transfer_prop.T)).toarray()
            scores = (1.0 - self.alpha) * sr_scores + self.alpha * morph_scores
        else:
            scores = sr_scores

        nseg, ntypes = scores.shape
        quotas = self._global_ct_quota.copy().astype(int)  # (K,)

        # ---------- Solve ----------
        if self.use_mip and _GUROBI_AVAILABLE:
            # Exact MILP
            X = self._solve_mip(scores, quotas)
        else:
            # QP relaxation (as an LP with small diagonal), then rounding & repair
            X = self._solve_qp_relaxation_and_round(scores, quotas)

        # ---------- Write back to AnnData ----------
        max_idx = np.argmax(X, axis=1)
        seg_cp = self.seg_adata.copy()
        seg_cp.obs['pred_cell_type'] = [self.cell_types[i] for i in max_idx]
        # one-hot columns
        for k, ct in enumerate(self.cell_types):
            seg_cp.obs[ct] = (max_idx == k).astype(int)

        return seg_cp

    # ---------- QP relaxation + rounding ----------

    def _solve_qp_relaxation_and_round(self, scores: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        """
        Solve relaxed assignment with qpsolvers, then round to strictly satisfy quotas.

        Maximize sum(scores * x) subject to:
          • row sums = 1
          • column sums = quotas
          • 0 <= x <= 1

        Implemented as: minimize q^T x with tiny quadratic term for numerical stability.
        """
        nseg, ntypes = scores.shape
        x_size = nseg * ntypes

        # Maximize sum scores * x  => minimize q^T x with q = -vec(scores)
        q = -scores.reshape(-1, order='C')  # row-major (i,k) -> i*ntypes + k

        # Small diagonal regularization to avoid degeneracy in some solvers
        eps = 1e-8
        P = sp.eye(x_size, format='csc') * eps

        # Equality constraints: row sums = 1, column sums = quotas
        # A_row: (nseg x x_size), each row selects the K vars of a segment
        A_row = sp.kron(sp.eye(nseg), np.ones((1, ntypes)))
        b_row = np.ones(nseg)

        # A_col: (ntypes x x_size), each row aggregates one cell type across segments
        A_col = sp.kron(np.ones((1, nseg)), sp.eye(ntypes))
        b_col = quotas.astype(float)

        A = sp.vstack([A_row, A_col], format='csc')
        b = np.hstack([b_row, b_col])

        # Bounds 0 <= x <= 1  =>  Gx <= h with G = [ I ; -I ], h = [1; 0]
        G = sp.vstack([sp.eye(x_size), -sp.eye(x_size)], format='csc')
        h = np.hstack([np.ones(x_size), np.zeros(x_size)])

        # Solve
        x_sol = solve_qp(P, q, G, h, A, b, solver=self.qp_solver, verbose=False)
        if x_sol is None:
            # Fallback: ignore column quotas; pick per-row via softmax,
            # then repair to satisfy quotas.
            probs = scipy.special.softmax(scores, axis=1)
            assign = np.argmax(probs, axis=1)
            X = np.zeros_like(scores, dtype=int)
            for i, k in enumerate(assign):
                X[i, k] = 1
            return self._repair_quotas(X, quotas, scores)

        X = x_sol.reshape(nseg, ntypes, order='C')

        # Round to one-hot
        hard = np.zeros_like(X, dtype=int)
        top1 = np.argmax(X, axis=1)
        hard[np.arange(nseg), top1] = 1

        # Enforce column quotas exactly
        hard = self._repair_quotas(hard, quotas, scores)
        return hard

    @staticmethod
    def _repair_quotas(hard_X: np.ndarray, quotas: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Adjust a one-hot assignment matrix so that column sums equal `quotas`.

        Strategy:
          • For any overfilled column, release the lowest marginal-loss items.
          • Reassign released items to underfilled columns using their best available scores.
        """
        nseg, ntypes = hard_X.shape
        col_sums = hard_X.sum(axis=0).astype(int)
        quotas = quotas.astype(int)

        surplus = col_sums - quotas
        deficit = quotas - col_sums

        # Release phase: remove items from overfilled columns with smallest loss
        release_pool = []  # entries are (loss, i, old_k)
        for k in np.where(surplus > 0)[0]:
            idx = np.where(hard_X[:, k] == 1)[0]
            if idx.size == 0:
                continue
            # For each i, find best alternative score (excluding k)
            alt_scores = scores[idx, :].copy()
            alt_scores[:, k] = -np.inf
            best_alt = np.max(alt_scores, axis=1)
            loss = scores[idx, k] - best_alt
            # Take surplus[k] items with smallest loss
            take = surplus[k]
            if take > 0 and idx.size > 0:
                pick = np.argpartition(loss, take - 1)[:take]
                for ii in pick:
                    i = idx[ii]
                    release_pool.append((loss[ii], i, k))
                hard_X[idx[pick], k] = 0  # release
        # Update deficits after release
        col_sums = hard_X.sum(axis=0).astype(int)
        deficit = quotas - col_sums

        # Reassignment phase: assign released items to underfilled columns
        release_pool.sort(key=lambda t: t[0])
        for _, i, _oldk in release_pool:
            need_ks = np.where(deficit > 0)[0]
            if need_ks.size == 0:
                # No remaining deficit: choose best column for this row
                new_k = np.argmax(scores[i, :])
            else:
                # Among deficit columns, choose the best-scoring one
                k_best = need_ks[np.argmax(scores[i, need_ks])]
                new_k = int(k_best)
            hard_X[i, new_k] = 1
            deficit[new_k] -= 1

        # If some deficit remains (rare), assign free rows as a fallback
        if np.any(deficit > 0):
            free_rows = np.where(hard_X.sum(axis=1) == 0)[0]
            for k in np.where(deficit > 0)[0]:
                need = deficit[k]
                if need <= 0:
                    continue
                if free_rows.size > 0:
                    take = min(len(free_rows), need)
                    take_rows = free_rows[:take]
                    hard_X[take_rows, k] = 1
                    free_rows = free_rows[take:]
                    deficit[k] -= take

        return hard_X

    # ---------- Gurobi MILP (exact 0/1 assignment with row=1, col=quota) ----------

    def _solve_mip(self, scores: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        if not _GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi not available, cannot run MILP.")
        nseg, ntypes = scores.shape
        quotas = quotas.astype(int)

        model = gp.Model("CellTypeAssign")
        model.setParam("OutputFlag", 0)

        X = model.addVars(nseg, ntypes, vtype=GRB.BINARY, name="X")
        # Objective: maximize sum score * X
        model.setObjective(
            gp.quicksum(scores[i, k] * X[i, k] for i in range(nseg) for k in range(ntypes)),
            GRB.MAXIMIZE
        )
        # Row sums = 1
        for i in range(nseg):
            model.addConstr(gp.quicksum(X[i, k] for k in range(ntypes)) == 1)

        # Column sums = quotas
        for k in range(ntypes):
            model.addConstr(gp.quicksum(X[i, k] for i in range(nseg)) == int(quotas[k]))

        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("MILP not optimal.")

        X_sol = np.array([[int(X[i, k].X) for k in range(ntypes)] for i in range(nseg)], dtype=int)
        return X_sol
