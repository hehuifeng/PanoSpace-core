import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from panospace._core.prediction.predictor_backend.predictor_utils import (
    compute_celltype_means_sparse,
    construct_graph_delaunay_inverse,
)


def _build_reference_adata():
    data = sp.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    adata = ad.AnnData(X=data)
    adata.obs["celltype_major"] = pd.Categorical(
        ["T", "T", "B", "B"], categories=["T", "B", "Myeloid"]
    )
    adata.var_names = ["g1", "g2", "g3"]
    return adata


def test_compute_celltype_means_sparse_returns_ordered_means():
    sc_adata = _build_reference_adata()
    result = compute_celltype_means_sparse(sc_adata, ["T", "B"], "celltype_major")

    assert sp.isspmatrix_csr(result)
    np.testing.assert_allclose(
        result.toarray(),
        np.array(
            [
                [0.5, 0.5, 1.0],  # mean of T cells
                [1.0, 1.0, 0.5],  # mean of B cells
            ]
        ),
    )


def test_compute_celltype_means_sparse_adds_empty_rows_for_missing_types():
    sc_adata = _build_reference_adata()
    result = compute_celltype_means_sparse(
        sc_adata, ["T", "B", "Myeloid"], "celltype_major"
    )

    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.toarray()[-1], np.zeros(3))


def test_construct_graph_delaunay_inverse_triangle_row_stochastic():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph = construct_graph_delaunay_inverse(coords)

    assert graph.shape == (3, 3)
    row_sums = np.asarray(graph.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(3))
    assert np.all(graph.diagonal() > 0)
    # Each node should have non-zero affinity to the others
    assert np.count_nonzero(graph.toarray() > 0) >= 9


def test_construct_graph_delaunay_inverse_degenerate_cases():
    empty_graph = construct_graph_delaunay_inverse(np.empty((0, 2)))
    assert empty_graph.shape == (0, 0)

    single_graph = construct_graph_delaunay_inverse(np.array([[0.0, 0.0]]))
    assert single_graph.shape == (1, 1)
    np.testing.assert_allclose(single_graph.toarray(), np.array([[1.0]]))
