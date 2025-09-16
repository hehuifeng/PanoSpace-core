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
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree
from typing import Literal, Optional, Union

def radius_membership_sparse(
    base_points: np.ndarray,
    query_points: np.ndarray,
    r: Union[float, np.ndarray],
    metric: Literal["euclidean", "chebyshev"] = "euclidean",
    leaf_size: int = 40,
    chunk_size: Optional[int] = None,
    dtype=np.int8,
    sort_results: bool = False,
) -> csr_matrix:
    """
    用KDTree在半径 r 内计算查询点对基点的从属关系（邻接），返回稀疏CSR矩阵 M (n_query x n_base)。
    若 query i 到 base j 的距离 <= r，则 M[i, j] = 1（或指定 dtype 的单位值），否则为 0。

    参数
    ----
    base_points : (n_base, d) 的 ndarray
        作为KDTree构建的数据点（基点集合）。
    query_points : (n_query, d) 的 ndarray
        需要查询的点集合。
    r : float 或 (n_query,) 的 ndarray
        半径阈值。可为标量或每个查询点独立半径。
    metric : {"euclidean", "chebyshev"}
        构树与查询的距离度量。欧氏距离/曼哈顿距离。
    leaf_size : int
        KDTree叶子大小，影响查询/内存权衡。
    chunk_size : Optional[int]
        分块大小；为 None 表示一次性处理全部查询点。
        对非常大的数据，建议设一个合适的值（例如 100_000）。
    dtype : np.dtype
        稀疏矩阵的数据类型。
    sort_results : bool
        是否对每个查询点的邻居索引按距离排序（可能略慢）。

    返回
    ----
    M : scipy.sparse.csr_matrix, shape = (n_query, n_base)
        从属稀疏矩阵。
    """
    if metric not in ("euclidean", "chebyshev"):
        raise ValueError("metric 只能是 'euclidean' 或 'chebyshev'")

    base_points = np.ascontiguousarray(base_points, dtype=np.float64)
    query_points = np.ascontiguousarray(query_points, dtype=np.float64)

    n_query = query_points.shape[0]
    n_base = base_points.shape[0]

    tree = KDTree(base_points, leaf_size=leaf_size, metric=metric)

    # 将 r 规范为数组（便于分块）
    if np.isscalar(r):
        r_arr = None  # 用标量 r 直接传入 KDTree.query_radius
        r_scalar = float(r)
    else:
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (n_query,):
            raise ValueError("当 r 为数组时，形状必须是 (n_query,)")
        r_arr = r
        r_scalar = None

    # ---------- 第1遍：只计数，预分配索引数组大小 ----------
    if chunk_size is None:
        counts = tree.query_radius(
            query_points,
            r=r_scalar if r_arr is None else r_arr,
            count_only=True,
        )
        total_nnz = int(np.sum(counts))
        indptr = np.empty(n_query + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(total_nnz, dtype=np.int32)
        # data 全部为 1（或 dtype 对应的“1”）
        data = np.ones(total_nnz, dtype=dtype)

        # ---------- 第2遍：取索引并填充 ----------
        # 为避免再次全量查询距离（浪费），我们再次 query_radius 但取 indices
        neighbor_lists = tree.query_radius(
            query_points,
            r=r_scalar if r_arr is None else r_arr,
            return_distance=False,
            sort_results=sort_results,
        )
        # neighbor_lists 是长度为 n_query 的 object 数组，每个元素是 1D 索引数组
        pos = 0
        for i in range(n_query):
            inds = neighbor_lists[i].astype(np.int32, copy=False)
            end = pos + inds.size
            indices[pos:end] = inds
            pos = end

    else:
        # 分块：两遍流程（计数 -> 分配 -> 填充）
        # 先统计每块的 counts，拼成全局 indptr
        indptr = np.zeros(n_query + 1, dtype=np.int64)
        # 暂存每块的局部 counts 以避免重复查询
        block_counts = []
        starts = list(range(0, n_query, chunk_size))
        for start in starts:
            end = min(start + chunk_size, n_query)
            qp = query_points[start:end]
            if r_arr is None:
                counts_blk = tree.query_radius(qp, r=r_scalar, count_only=True)
            else:
                counts_blk = tree.query_radius(qp, r=r_arr[start:end], count_only=True)
            block_counts.append(counts_blk)
            indptr[start + 1 : end + 1] = counts_blk

        # 前缀和得到全局 indptr
        np.cumsum(indptr, out=indptr)
        total_nnz = int(indptr[-1])
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.ones(total_nnz, dtype=dtype)

        # 再次分块填充 indices
        for blk_id, start in enumerate(starts):
            end = min(start + chunk_size, n_query)
            qp = query_points[start:end]
            if r_arr is None:
                lists_blk = tree.query_radius(
                    qp, r=r_scalar, return_distance=False, sort_results=sort_results
                )
            else:
                lists_blk = tree.query_radius(
                    qp, r=r_arr[start:end], return_distance=False, sort_results=sort_results
                )
            # 将该块写入全局 indices
            write_pos = indptr[start]
            for local_i, inds in enumerate(lists_blk):
                inds = inds.astype(np.int32, copy=False)
                next_pos = write_pos + inds.size
                indices[write_pos:next_pos] = inds
                write_pos = next_pos
            # 安全检查（可去掉以提速）
            # assert write_pos == indptr[end]

    # 构造 CSR 矩阵
    M = csr_matrix((data, indices, indptr), shape=(n_query, n_base), dtype=dtype)
    return M


class CellTypeAnnotator:
    """
    高效版 CellTypeAnnotator：
    - 以解卷积的 spot 级（及其超分辨版本）的细胞类型比例为依据，
      结合分割得到的单元（segment）空间位置与可选的图像形态先验，
      通过 OT 对齐 + （QP 连续松弛 或 Gurobi MILP）完成 segment→celltype 的全局一致指派。
    """

    def __init__(
        self,
        spot_adata,
        sr_spot_adata,
        seg_adata,
        priori_type_affinities=None,
        alpha=0.3,
        ot_mode="emd",          # "sinkhorn" or "emd"
        sinkhorn_reg=0.01,           # Sinkhorn 正则
        qp_solver: str = "osqp",     # qpsolvers 的后端选择："osqp"|"cvxopt"|...
        use_mip: bool = False        # True 则用 Gurobi MILP 做最终 0/1 指派（若可用）
    ):
        """
        Parameters
        ----------
        spot_adata : AnnData
            含 spot 级解卷积比例的 AnnData，要求：
            - .uns['celltype'] 为细胞类型名列表
            - .uns['radius']   为spot半径
            - .obsm['spatial'] 为 spot 坐标 (S x 2/3)
            - .obs[ctype] 列为每个 cell type 的比例
        sr_spot_adata : AnnData
            含“超分辨spot级”解卷积比例，.obs 同上，列名与 celltype 一致；.obsm['spatial'] 坐标。
        seg_adata : AnnData
            分割得到的 segment（或 nuclei），.obsm['spatial'] 为坐标；
            可选 .obs['img_type'] 为整数编码的形态类别（若存在将启用 'mor' 模式）。
        priori_type_affinities : dict[str, list[str]] or None
            形态类别名（键）到细胞类型名（值列表）的先验亲和度。
        alpha : float
            融合权重，启用 'mor' 时在（空间→超分比例）与（形态先验→OT对齐）之间加权。
        ot_mode : str
            "sinkhorn"（默认）更快更稳；或 "emd"（精确离散 OT）。
        sinkhorn_reg : float
            Sinkhorn 熵正则强度。
        qp_solver : str
            qpsolvers 后端名（如 "osqp"、"cvxopt"、"quadprog"...）。
        use_mip : bool
            True：若可用，则用 Gurobi MILP 做最终 0/1 精确指派；False：用 QP+舍入。
        """
        self.spot_adata = spot_adata
        self.sr_spot_adata = sr_spot_adata
        self.seg_adata = seg_adata
        self.alpha = float(alpha)
        self.ot_mode = ot_mode
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.qp_solver = qp_solver
        self.use_mip = bool(use_mip)

        # 模式：若 seg_adata.obs 中存在 img_type，则启用形态对齐分支
        self.mode = 'mor' if 'img_type' in self.seg_adata.obs.columns else None

        # 细胞类型及比例（列名同 celltype 列表）
        self.cell_types = list(self.spot_adata.uns['celltype'])
        self.ct_cols = self.cell_types

        # SR-spot 比例矩阵（S_sr x K）
        self.sr_ct_ratios = self._safe_row_normalize(
            self.sr_spot_adata.obs[self.ct_cols].to_numpy(copy=True)
        )

        # spot 比例矩阵（S x K），负值截断后行归一
        self.spot_ct_ratios = self._safe_row_normalize(
            np.clip(self.spot_adata.obs[self.ct_cols].to_numpy(copy=True), 0, None)
        )

        # 半径阈值
        self.radius = float(self.spot_adata.uns['radius'])

        # 先验亲和度
        self.priori_type_affinities = priori_type_affinities or {}
        self.img_type_names = list(self.priori_type_affinities.keys()) if self.priori_type_affinities else []

        # 提前缓存 spatial 坐标（NumPy）
        self._sr_spatial = np.asarray(self.sr_spot_adata.obsm['spatial'])
        self._spot_spatial = np.asarray(self.spot_adata.obsm['spatial'])
        self._seg_spatial = np.asarray(self.seg_adata.obsm['spatial'])

        # 各类稀疏矩阵缓存（后续填充）
        self._affil_sr2seg_csr = None        # (S_sr x Nseg) 或 (Nseg x S_sr) 视 utils 返回定义而定
        self._affil_spot2seg_csr = None      # (S x Nseg)   与上统一，见 _build_affiliations
        self._affil_sr2seg_norm_csr = None   # 行归一版本
        self._affil_sr2seg_csc = None        # 转置/格式缓存
        self._affil_spot2seg_csr_T = None    # 转置缓存

        # 形态 one-hot 与相关统计缓存
        self._imgtype_keep_labels = None
        # self._seg_imgtype_labels = None      # shape (Nseg,)
        self._seg_imgtype_onehot_csr = None  # (Nseg x G_kept)
        self._mortype_in_spot = None         # (G_kept x S_nonzero)
        self._imgtype_ratio = None           # (G_kept, ) 归一向量

        # 计数与配额
        self._cell_counts_per_spot = None    # (S, )
        self._int_ct_ratios_per_spot = None  # (S_nonzero x K) 整数化后的每 spot 配额
        self._global_ct_quota = None         # N (K, ) 全局配额（整数）
        self._nonzero_spot_mask = None       # (S, ) bool

    # ---------- 基础工具 ----------

    @staticmethod
    def _safe_row_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x[x < 0] = 0.0
        s = x.sum(axis=1, keepdims=True)
        s[s < eps] = 1.0
        return x / s

    @staticmethod
    def _integerize_proportions(p: np.ndarray, total: int) -> np.ndarray:
        """把比例向量 p（可能含 0/负，自动归一）离散为整数并精确满足总和 total。"""
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

    # ---------- 预处理：筛选 & 隶属矩阵构建 ----------

    def filter_and_build_affiliations(self):
        """
        1) 用 SR-spot 视野筛选 segment（一次性批量包含判断）
        2) 构建并缓存 SR-spot→segment 与 spot→segment 的 CSR/CSC 矩阵
        3) 对 SR-spot→segment 做行归一（用于传播 SR 比例）
        """
        # SR-spot 视野筛选（批量）
        affil_sr2seg = radius_membership_sparse(
            base_points = self._sr_spatial,
            query_points = self._seg_spatial,
            r=self.radius,
            metric = 'chebyshev'
        )
        # 过滤 seg：只保留被至少一个 SR-spot 覆盖的 segment
        keep_mask_seg = (affil_sr2seg.getnnz(axis=1) != 0)
        self.seg_adata = self.seg_adata[keep_mask_seg].copy()
        self._seg_spatial = np.asarray(self.seg_adata.obsm['spatial'])

        # 重新计算 SR-spot→segment 和 spot→segment（注意 utils 的返回维度约定）
        affil_sr2seg = radius_membership_sparse(
            self._sr_spatial,
            self._seg_spatial,
            r=self.radius,
            metric = 'chebyshev'
        ).transpose()  # 约定返回形状：(S_sr x Nseg)
        affil_spot2seg = radius_membership_sparse(
            self._spot_spatial,
            self._seg_spatial,
            r=self.radius,
            metric = 'euclidean'
        ).transpose()  # 约定返回形状：(S x Nseg)

        # CSR 缓存
        self._affil_sr2seg_csr = affil_sr2seg.tocsr()             # (S_sr x Nseg)
        self._affil_spot2seg_csr = affil_spot2seg.tocsr()         # (S x Nseg)
        # self._affil_spot2seg_csr_T = self._affil_spot2seg_csr.transpose().tocsr()  # (Nseg x S)

        # SR-spot→segment 行归一
        A = self._affil_sr2seg_csr.copy() 
        row_sums = A.sum(axis=1).A1
        inv = np.divide(1.0, row_sums, out=np.zeros_like(row_sums), where=row_sums!=0)

        A.data *= inv.repeat(np.diff(A.indptr))
        self._affil_sr2seg_norm_csr = A  # (S_sr x Nseg) 行归一

        # 模式判定：形态标签
        if self.mode == 'mor':
            self._build_imgtype_onehot_and_spot_aggregates()

    # ---------- 形态标签：一次性 one-hot + 稀疏乘 ----------

    def _build_imgtype_onehot_and_spot_aggregates(self):
        """
        构造 segment 的 img_type one-hot（仅保留有效类），并按 spot 聚合计数。
        """
        # 标签映射（与原版一致）：0:nolabel, 4:Dead Cells → 移除；其余保留
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
        # 反向：名字->紧凑列索引
        keep_name2col = {name: i for i, name in enumerate(keep)}
        # 原始整数标签
        raw = np.asarray(self.seg_adata.obs['img_type']).astype(int)
        # 映射到名字
        mapped_names = np.vectorize(label_dict.get)(raw)
        valid_mask = np.array([name in keep_name2col for name in mapped_names], dtype=bool)
        # 生成 one-hot（Nseg x G_kept）
        rows = np.nonzero(valid_mask)[0]
        cols = np.fromiter((keep_name2col[n] for n in mapped_names[valid_mask]), count=rows.size, dtype=int)
        data = np.ones_like(rows, dtype=np.int32)
        nseg = self.seg_adata.n_obs
        gk = len(keep_name2col)
        onehot = sp.csr_matrix((data, (rows, cols)), shape=(nseg, gk))
        self._seg_imgtype_onehot_csr = onehot
        self._imgtype_keep_labels = keep
        # self._seg_imgtype_labels = raw  # 仅缓存以备需要

        # spot聚合：(#S x #Nseg) @ (#Nseg x #G_kept) = (#S x #G_kept)
        spot_morph_counts = (self._affil_spot2seg_csr @ onehot).astype(np.float64)  # (S x G_kept)
        # 只考虑 cell_count>0 的 spot
        self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)
        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No spots contain any segments after filtering.")

        spot_morph_counts = spot_morph_counts[self._nonzero_spot_mask, :]  # (S_nz x G_kept)
        # 转置为 (G_kept x S_nz) 以与后续 OT 代码一致
        self._mortype_in_spot = spot_morph_counts.T.toarray()  # dense for OT
        # 全局形态占比（G_kept, )
        total = np.sum(self._mortype_in_spot)
        if total <= 0:
            # 极端情况：无有效形态，回退均匀
            self._imgtype_ratio = np.full(self._mortype_in_spot.shape[0], 1.0 / self._mortype_in_spot.shape[0])
        else:
            self._imgtype_ratio = self._mortype_in_spot.sum(axis=1) / total

    # ---------- 计数与整数配额 ----------

    def compute_counts_and_integerize(self):
        """
        计算每个 spot 的 segment 个数，并将 spot 级 celltype 比例离散为整数配额；
        同时计算全局 celltype 配额 N。
        """
        if self._affil_spot2seg_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() first.")

        # 每个 spot 的 segment 数
        if self._cell_counts_per_spot is None:
            self._cell_counts_per_spot = np.asarray(self._affil_spot2seg_csr.sum(axis=1)).ravel().astype(int)

        self._nonzero_spot_mask = self._cell_counts_per_spot != 0
        if not np.any(self._nonzero_spot_mask):
            raise ValueError("No non-empty spots found.")

        # 取非零 spot 的比例
        spot_ratios_nz = self.spot_ct_ratios[self._nonzero_spot_mask, :]  # (S_nz x K)
        counts_nz = self._cell_counts_per_spot[self._nonzero_spot_mask]   # (S_nz, )

        # 整数化每个 spot 的 celltype 配额 V（严格满足每行和）
        int_ratios_list = [
            self._integerize_proportions(spot_ratios_nz[s, :], int(counts_nz[s]))
            for s in range(spot_ratios_nz.shape[0])
        ]
        self._int_ct_ratios_per_spot = np.vstack(int_ratios_list)  # (S_nz x K)

        # 全局 N：按全局比例 + 总 segment 数离散
        total_segments = self.seg_adata.n_obs
        global_ratio = self._int_ct_ratios_per_spot.sum(axis=0).astype(float)
        if global_ratio.sum() <= 0:
            global_ratio = np.full_like(global_ratio, 1.0 / len(global_ratio))
        else:
            global_ratio = global_ratio / global_ratio.sum()
        self._global_ct_quota = self._integerize_proportions(global_ratio, total_segments)  # (K,)

    # ---------- OT：细胞类型 ↔ 形态类别 对齐 ----------

    def build_type_transfer(self, factor: float = 2.0):
        """
        以“在 spot 空间中的分布向量”作余弦距离，做细胞类型 ↔ 形态类别的 OT 对齐；
        若给定先验亲和度，则对成本降权。
        """
        if self._mortype_in_spot is None and self.mode == 'mor':
            raise RuntimeError("Morphology aggregates not built. Call filter_and_build_affiliations() first.")

        # 类型在 spot 空间的“签名”（K x S_nz）
        ct_signature = self._int_ct_ratios_per_spot.T.astype(float)  # (K x S_nz)
        # 形态在 spot 空间的“签名”（G_kept x S_nz）
        morph_signature = self._mortype_in_spot.astype(float)        # (G x S_nz)

        # 余弦距离成本 (K x G)
        cost = ot.dist(ct_signature, morph_signature, metric='cosine')

        # 先验亲和度降权
        if self.priori_type_affinities:
            adjusted = cost.copy()
            for g_idx, g_name in enumerate(self._imgtype_keep_labels):
                prefer_cts = self.priori_type_affinities.get(g_name, [])
                for ct_name in prefer_cts:
                    if ct_name in self.cell_types:
                        k = self.cell_types.index(ct_name)
                        adjusted[k, g_idx] /= float(factor)
            cost = adjusted

        # 源分布：全局 celltype 占比（K,）
        ct_ratio_global = self._global_ct_quota.astype(float)
        if ct_ratio_global.sum() <= 0:
            ct_ratio_global = np.full_like(ct_ratio_global, 1.0 / len(ct_ratio_global))
        else:
            ct_ratio_global = ct_ratio_global / ct_ratio_global.sum()

        # 目标分布：全局形态占比（G,）
        morph_ratio_global = self._imgtype_ratio

        # OT
        if self.ot_mode == "sinkhorn":
            gamma = ot.sinkhorn(ct_ratio_global, morph_ratio_global, cost, reg=self.sinkhorn_reg, numItermax=2000)
        else:
            gamma = ot.emd(ct_ratio_global, morph_ratio_global, cost, numItermax=100000)

        # 列归一得到“形态→类型”的传输比例
        col_sums = gamma.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1.0
        self.type_transfer_prop = gamma / col_sums  # (K x G)

    # ---------- 最终指派：QP 连续松弛（可选 MIP 精确） ----------

    def infer_cell_types(self):
        """
        生成 segment→celltype 的指派：
        - 得分矩阵：SR-spot 传播到 segment 的比例（_affil_sr2seg_norm_csr @ sr_ct_ratios）
          与（形态 one-hot × OT 传输）的线性融合（若启用 'mor'）
        - 若 use_mip 且可用 Gurobi：求解 MILP 得到精确 0/1 指派
          否则：QP 连续松弛 + 舍入，并做全局配额修正
        Returns
        -------
        seg_adata_pred : AnnData
            带有 'pred_cell_type' 与 one-hot 列的 seg_adata 拷贝
        """
        if self._affil_sr2seg_norm_csr is None:
            raise RuntimeError("Call filter_and_build_affiliations() and compute_counts_and_integerize() first.")

        # （1）SR-spot 比例传播到 segment（Nseg x K）
        #    (Nseg x S_sr) @ (S_sr x K) —— 我们已有 (S_sr x Nseg) 的 CSR，使用其转置以减少临时对象
        sr2seg_norm_T = self._affil_sr2seg_norm_csr.transpose().tocsr()  # (Nseg x S_sr)
        sr_scores = (sr2seg_norm_T @ sp.csr_matrix(self.sr_ct_ratios)).toarray()  # dense (Nseg x K)

        # （2）形态先验分支（若启用）
        if self.mode == 'mor':
            # segment 的 onehot（Nseg x G） × (G x K) 需要 (K x G)^T
            # type_transfer_prop: (K x G) → (G x K)
            morph_scores = (self._seg_imgtype_onehot_csr @ sp.csr_matrix(self.type_transfer_prop.T)).toarray()
            scores = (1.0 - self.alpha) * sr_scores + self.alpha * morph_scores
        else:
            scores = sr_scores

        nseg, ntypes = scores.shape
        quotas = self._global_ct_quota.copy().astype(int)  # (K,)

        # ---------- 求解 ----------
        if self.use_mip and _GUROBI_AVAILABLE:
            # 精确 MILP
            X = self._solve_mip(scores, quotas)
        else:
            # 连续松弛 QP（作为 LP 使用），再舍入修正
            X = self._solve_qp_relaxation_and_round(scores, quotas)

        # ---------- 写回 AnnData ----------
        max_idx = np.argmax(X, axis=1)
        seg_cp = self.seg_adata.copy()
        seg_cp.obs['pred_cell_type'] = [self.cell_types[i] for i in max_idx]
        # one-hot 列
        for k, ct in enumerate(self.cell_types):
            seg_cp.obs[ct] = (max_idx == k).astype(int)

        return seg_cp

    # ---------- QP 连续松弛 + 舍入 ----------

    def _solve_qp_relaxation_and_round(self, scores: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        """
        Solve relaxed assignment with qpsolvers, then round to satisfy quotas strictly.
        Minimize: 0.5 * x^T P x + q^T x  with P = eps*I (stabilize), q = -vec(scores)
        s.t.  (row sums) = 1, (column sums) = quotas, 0 <= x <= 1
        """
        nseg, ntypes = scores.shape
        x_size = nseg * ntypes

        # 目标：maximize sum scores * x  => minimize q^T x with q = -vec(scores)
        q = -scores.reshape(-1, order='C')  # 行主序 (i,k) -> i*ntypes + k

        # 小的对角正则，避免部分 solver 对 P=0 的退化
        eps = 1e-8
        P = sp.eye(x_size, format='csc') * eps

        # 等式约束：行和=1、列和=quota
        # A_row: (nseg x x_size) —— 每行对应一组变量 x[i, :]
        A_row = sp.kron(sp.eye(nseg), np.ones((1, ntypes)))  # 每个 segment 的1xK 全1
        b_row = np.ones(nseg)

        # A_col: (ntypes x x_size) —— 每列对应所有 segment 的该类型
        A_col = sp.kron(np.ones((1, nseg)), sp.eye(ntypes))  # 1xN  kron  I_K
        b_col = quotas.astype(float)

        A = sp.vstack([A_row, A_col], format='csc')
        b = np.hstack([b_row, b_col])

        # 0 <= x <= 1  =>  Gx <= h with G = [ I ; -I ], h = [1; 0]
        G = sp.vstack([sp.eye(x_size), -sp.eye(x_size)], format='csc')
        h = np.hstack([np.ones(x_size), np.zeros(x_size)])

        # 求解
        x_sol = solve_qp(P, q, G, h, A, b, solver=self.qp_solver, verbose=False)
        if x_sol is None:
            # 回退：忽略列配额，只 enforce 每行=1 的 softmax 选择
            # （在部分求解器不可用/失败时的鲁棒回退）
            probs = scipy.special.softmax(scores, axis=1)
            assign = np.argmax(probs, axis=1)
            X = np.zeros_like(scores, dtype=int)
            for i, k in enumerate(assign):
                X[i, k] = 1
            # 配额修正
            return self._repair_quotas(X, quotas, scores)

        X = x_sol.reshape(nseg, ntypes, order='C')

        # 舍入为 one-hot
        hard = np.zeros_like(X, dtype=int)
        top1 = np.argmax(X, axis=1)
        hard[np.arange(nseg), top1] = 1

        # 严格修正列配额为 quotas
        hard = self._repair_quotas(hard, quotas, scores)
        return hard

    @staticmethod
    def _repair_quotas(hard_X: np.ndarray, quotas: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        把 one-hot 指派矩阵 hard_X 的列和修正为 quotas（严格）。
        策略：若某列超额，从该列中“边际损失最小”的样本转移到“欠额列”的最佳候选。
        """
        nseg, ntypes = hard_X.shape
        col_sums = hard_X.sum(axis=0).astype(int)
        quotas = quotas.astype(int)

        # 先处理超额列：把多余的样本释放出来，记录其重新安置的优先顺序
        # gain_ik = scores[i, cur_k] - scores[i, alt_k] 用于挑最小损失
        surplus = col_sums - quotas
        deficit = quotas - col_sums

        # 释放阶段
        release_pool = []  # (loss, i) 由超额列释放的样本
        for k in np.where(surplus > 0)[0]:
            idx = np.where(hard_X[:, k] == 1)[0]
            if idx.size == 0:
                continue
            # 对每个 i，找到其次优候选得分（不等于 k 的最高分）
            alt_scores = scores[idx, :].copy()
            alt_scores[:, k] = -np.inf
            best_alt = np.max(alt_scores, axis=1)
            loss = scores[idx, k] - best_alt  # 转移损失
            # 取 surplus[k] 个损失最小的释放
            take = surplus[k]
            if take > 0 and idx.size > 0:
                pick = np.argpartition(loss, take - 1)[:take]
                for ii in pick:
                    i = idx[ii]
                    release_pool.append((loss[ii], i, k))
                hard_X[idx[pick], k] = 0  # 释放
        # 更新目前列和与缺口
        col_sums = hard_X.sum(axis=0).astype(int)
        deficit = quotas - col_sums

        # 重新安置阶段：把释放出来的样本按“损失从小到大”放到欠额列的最佳选择
        release_pool.sort(key=lambda t: t[0])
        for _, i, _oldk in release_pool:
            need_ks = np.where(deficit > 0)[0]
            if need_ks.size == 0:
                # 若无缺口，塞回其本行的最佳列
                new_k = np.argmax(scores[i, :])
            else:
                # 从欠额列中选分数最高的
                k_best = need_ks[np.argmax(scores[i, need_ks])]
                new_k = int(k_best)
            hard_X[i, new_k] = 1
            deficit[new_k] -= 1

        # 若仍有 deficit（极少发生），对未指派行为做兜底
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

    # ---------- Gurobi MILP（精确 0/1 指派，行和=1、列和=quota） ----------

    def _solve_mip(self, scores: np.ndarray, quotas: np.ndarray) -> np.ndarray:
        if not _GUROBI_AVAILABLE:
            raise RuntimeError("Gurobi not available, cannot run MILP.")
        nseg, ntypes = scores.shape
        quotas = quotas.astype(int)

        model = gp.Model("CellTypeAssign")
        model.setParam("OutputFlag", 0)

        X = model.addVars(nseg, ntypes, vtype=GRB.BINARY, name="X")
        # maximize sum score * X
        model.setObjective(gp.quicksum(scores[i, k] * X[i, k] for i in range(nseg) for k in range(ntypes)),
                           GRB.MAXIMIZE)
        # 每行=1
        for i in range(nseg):
            model.addConstr(gp.quicksum(X[i, k] for k in range(ntypes)) == 1)

        # 每列=quota
        for k in range(ntypes):
            model.addConstr(gp.quicksum(X[i, k] for i in range(nseg)) == int(quotas[k]))

        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("MILP not optimal.")

        X_sol = np.array([[int(X[i, k].X) for k in range(ntypes)] for i in range(nseg)], dtype=int)
        return X_sol

# if __name__ == "__main__":
#     cta = CellTypeAnnotator(
#         spot_adata=deconv_adata,
#         sr_spot_adata=sr_deconv_adata,
#         seg_adata=segment_adata,
#         priori_type_affinities=None,  # 可选
#         alpha=0.3,
#         ot_mode="emd",      # "emd" or "sinkhorn"
#         # sinkhorn_reg=0.01,
#         # qp_solver="osqp",        # qpsolvers 开源后端
#         use_mip=True            # 如需精确 0/1 指派，设 True（需 Gurobi）
#     )

#     cta.filter_and_build_affiliations()
#     cta.compute_counts_and_integerize()
#     if cta.mode == 'mor':
#         cta.build_type_transfer(factor=2.0)
#     seg_adata_pred = cta.infer_cell_types()