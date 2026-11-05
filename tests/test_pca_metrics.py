import numpy as np
import pytest

from utils.metrics_common import (
    fit_pca_features,
    frechet_distance,
    generation_diagnostics,
    mse,
    psnr,
    purity,
    silhouette,
    serialized_storage,
    ssim,
)
from utils.pca.dim_reduction import components_for_variance, fit_pca


def test_constant_pca_is_finite_and_round_trips():
    X = np.full((5, 4), 3.0)
    pca = fit_pca(X)
    assert np.isfinite(pca.explained_variance_ratio_).all()
    np.testing.assert_array_equal(pca.explained_variance_ratio_, np.zeros(4))
    assert components_for_variance(pca, 0.9) == 0
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X)


@pytest.mark.parametrize("bad", [np.empty((0, 3)), np.ones((1, 3)), [[1.0, np.nan]]])
def test_pca_rejects_invalid_training_data(bad):
    with pytest.raises(ValueError):
        fit_pca(bad)


def test_pca_rejects_invalid_component_count():
    X = np.eye(4)
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(X, n_components=5)


def test_reconstruction_metrics_identity_and_validation():
    X = np.zeros((2, 784), dtype=np.float32)
    assert mse(X, X) == 0.0
    assert psnr(X, X) == float("inf")
    assert ssim(X, X) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="même forme"):
        mse(X, X[:1])
    with pytest.raises(ValueError, match="non vide"):
        mse(np.empty((0, 784)), np.empty((0, 784)))


def test_clustering_metrics_cover_labels_and_singletons():
    assert purity([0, 0, 1, 1], [0, 0, 0, 1]) == pytest.approx(0.75)
    score = silhouette([[0.0], [0.1], [10.0]], [0, 0, 1], sample_size=3)
    assert 0.0 < score < 1.0
    assert np.isnan(silhouette([[0.0], [1.0]], [0, 1], sample_size=2))


def test_frechet_is_zero_for_identical_rank_deficient_clouds():
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert frechet_distance(points, points) == pytest.approx(0.0, abs=1e-10)
    assert frechet_distance(points, points + 1.0) > 0.0


def test_fd_pca_rejects_components_beyond_training_rank_bound():
    with pytest.raises(ValueError, match="n_components"):
        fit_pca_features(np.ones((3, 10)), n_components=3)


def test_storage_includes_model_and_container_costs():
    codes = np.zeros((10, 2), dtype=np.float32)
    storage = serialized_storage(codes, {"dictionary": np.eye(2)}, {"algorithm": "example"})
    assert storage["code_npz_bytes"] > 0
    assert storage["model_npz_bytes"] > 0
    assert storage["amortized_npz_bits_per_image"] == pytest.approx(
        8 * (storage["code_npz_bytes"] + storage["model_npz_bytes"]) / 10)
    # A tiny batch can cost more than its raw codes because headers are included.
    assert storage["code_npz_bytes"] > codes.nbytes


def test_generation_diagnostics_exposes_exact_collapse_and_training_copy():
    train = np.array([[0., 0.], [1., 1.], [2., 2.]])
    reference = train + 10.
    generated = np.repeat(train[:1], 4, axis=0)
    diagnostics = generation_diagnostics(train, reference, generated)
    assert diagnostics["gen_unique_fraction"] == .25
    assert diagnostics["gen_pixel_variance"] == 0.
    assert diagnostics["gen_nearest_train_mse"] == 0.
    assert diagnostics["gen_nearest_reference_mse"] > 0.
    assert diagnostics["diagnostic_n_generated"] == 4
