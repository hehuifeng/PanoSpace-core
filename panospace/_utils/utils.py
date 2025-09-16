"""
PanoSpace Common Utilities
==========================

"""
import numpy as np
from typing import Optional, Union, Literal
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree



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
    if metric not in ("euclidean", "chebyshev"):
        raise ValueError("metric 只能是 'euclidean' 或 'chebyshev'")

    base_points = np.ascontiguousarray(base_points, dtype=np.float64)
    query_points = np.ascontiguousarray(query_points, dtype=np.float64)

    n_query = query_points.shape[0]
    n_base = base_points.shape[0]

    tree = KDTree(base_points, leaf_size=leaf_size, metric=metric)

    # 将 r 规范为数组（便于分块）
    if np.isscalar(r):
        r_arr = None
        r_scalar = float(r)
    else:
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (n_query,):
            raise ValueError("当 r 为数组时，形状必须是 (n_query,)")
        r_arr = r
        r_scalar = None

    if chunk_size is None:
        counts = tree.query_radius(
            query_points, r=r_scalar if r_arr is None else r_arr, count_only=True
        )
        total_nnz = int(np.sum(counts))
        indptr = np.empty(n_query + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.ones(total_nnz, dtype=dtype)

        neighbor_lists = tree.query_radius(
            query_points,
            r=r_scalar if r_arr is None else r_arr,
            return_distance=False,
            sort_results=sort_results,
        )
        pos = 0
        for i in range(n_query):
            inds = neighbor_lists[i].astype(np.int32, copy=False)
            end = pos + inds.size
            indices[pos:end] = inds
            pos = end
    else:
        indptr = np.zeros(n_query + 1, dtype=np.int64)
        starts = list(range(0, n_query, chunk_size))
        for start in starts:
            end = min(start + chunk_size, n_query)
            qp = query_points[start:end]
            if r_arr is None:
                counts_blk = tree.query_radius(qp, r=r_scalar, count_only=True)
            else:
                counts_blk = tree.query_radius(qp, r=r_arr[start:end], count_only=True)
            indptr[start + 1 : end + 1] = counts_blk

        np.cumsum(indptr, out=indptr)
        total_nnz = int(indptr[-1])
        indices = np.empty(total_nnz, dtype=np.int32)
        data = np.ones(total_nnz, dtype=dtype)

        for start in starts:
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
            write_pos = indptr[start]
            for inds in lists_blk:
                inds = inds.astype(np.int32, copy=False)
                next_pos = write_pos + inds.size
                indices[write_pos:next_pos] = inds
                write_pos = next_pos

    M = csr_matrix((data, indices, indptr), shape=(n_query, n_base), dtype=dtype)
    return M