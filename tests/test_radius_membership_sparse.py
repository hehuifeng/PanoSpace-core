"""Unit tests for :func:`panospace._utils.utils.radius_membership_sparse`."""

import numpy as np

from panospace._utils.utils import radius_membership_sparse


def test_radius_membership_sparse_basic_usage():
    """The function should identify neighbors using a shared scalar radius."""

    base = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    query = np.array([[0.0, 0.0], [1.1, 1.1], [3.0, 3.0]])

    matrix = radius_membership_sparse(base, query, r=0.5)

    expected = np.array(
        [
            [1, 0, 0],  # first query matches first base point
            [0, 1, 0],  # second query matches second base point
            [0, 0, 0],  # third query has no neighbors within radius
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(matrix.toarray(), expected)
    assert matrix.dtype == np.int8


def test_radius_membership_sparse_chunked_with_radius_array():
    """Chunked execution should respect per-query radii and sorting."""

    base = np.array([[0.0, 0.0], [0.0, 0.2], [0.0, 0.4]])
    query = np.array([[0.0, 0.1], [0.0, 0.35]])
    radii = np.array([0.15, 0.15])

    matrix = radius_membership_sparse(
        base,
        query,
        r=radii,
        metric="chebyshev",
        chunk_size=1,
        dtype=np.int16,
        sort_results=True,
    )

    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=np.int16,
    )

    np.testing.assert_array_equal(matrix.toarray(), expected)
    assert matrix.dtype == np.int16

    # Ensure sorted results deliver monotonically increasing indices per row.
    first_row = matrix.indices[matrix.indptr[0] : matrix.indptr[1]]
    second_row = matrix.indices[matrix.indptr[1] : matrix.indptr[2]]
    np.testing.assert_array_equal(first_row, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(second_row, np.array([1, 2], dtype=np.int32))


def test_radius_membership_sparse_handles_empty_queries():
    """An empty query array should produce an empty sparse matrix."""

    base = np.array([[1.0, 1.0], [2.0, 2.0]])
    query = np.empty((0, 2))

    matrix = radius_membership_sparse(base, query, r=1.0)

    assert matrix.shape == (0, base.shape[0])
    assert matrix.nnz == 0
    assert matrix.dtype == np.int8
