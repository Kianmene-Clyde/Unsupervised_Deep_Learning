import numpy as np

from kmeas.utils.kmeas import (
    assign_clusters,
    compute_squared_distances,
    fit_kmeans,
    has_converged,
    update_centroids,
)


def test_squared_distances_match_broadcast_reference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(13, 7)).astype(np.float32)
    C = rng.normal(size=(4, 7)).astype(np.float32)
    expected = np.square(X[:, None, :] - C[None, :, :]).sum(axis=2)
    actual = np.asarray(compute_squared_distances(X, C))
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_fit_is_repeatable_and_returns_fresh_labels():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 5)).astype(np.float32)
    first = fit_kmeans(X, 5, max_iter=2, tolerance=0.0, seed=9, verbose=False)
    second = fit_kmeans(X, 5, max_iter=2, tolerance=0.0, seed=9, verbose=False)
    C1, labels1 = map(np.asarray, first)
    C2, labels2 = map(np.asarray, second)
    np.testing.assert_array_equal(C1, C2)
    np.testing.assert_array_equal(labels1, labels2)
    expected = np.asarray(assign_clusters(compute_squared_distances(X, C1)))
    np.testing.assert_array_equal(labels1, expected)


def test_convergence_tolerance_is_an_l2_displacement():
    old = np.zeros((1, 2), dtype=np.float32)
    moved = np.array([[0.005, 0.0]], dtype=np.float32)
    assert not bool(has_converged(old, moved, 1e-4))
    assert bool(has_converged(old, moved, 0.005))


def test_empty_cluster_keeps_its_previous_centroid():
    X = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int32)
    old = np.array([[1.0, 1.0], [9.0, 9.0]], dtype=np.float32)
    updated = np.asarray(update_centroids(X, labels, 2, old))
    np.testing.assert_allclose(updated, [[1.0, 1.0], [9.0, 9.0]])
