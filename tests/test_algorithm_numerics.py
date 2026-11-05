"""Independent numerical references for the algorithms implemented by hand."""

import numpy as np
import pytest

from gan.utils.model import Adam, _bce_with_logits, _sigmoid, mlp_backward, mlp_forward
from kohonen.utils.generation import _neighbors, sample_synthetic
from kohonen.utils.kohonen import fit_kohonen, grid_coords
from utils.neural_config import validate_training_config
from utils.pca.dim_reduction import components_for_variance, fit_pca
from utils.pca.generation import fit_latent_gaussian, sample_latent


@pytest.mark.parametrize("activation", ["linear", "sigmoid"])
def test_handwritten_mlp_gradients_match_central_differences(activation):
    # Float64 forward reference avoids perturbation round-off; the production
    # backward intentionally returns float32 gradients.
    rng = np.random.default_rng(12)
    X = rng.normal(size=(3, 2))
    layers = [[rng.normal(size=(2, 3)), np.full(3, 0.4)],
              [rng.normal(size=(3, 1)), np.full(1, 0.2)]]
    target = np.array([[0.0], [1.0], [0.0]])

    def objective():
        output, _ = mlp_forward(X, layers, activation)
        return (_bce_with_logits(output, target) if activation == "linear"
                else float(np.mean((output - target) ** 2)))

    output, caches = mlp_forward(X, layers, activation)
    delta = ((_sigmoid(output) - target) / len(X) if activation == "linear"
             else 2 * (output - target) / len(X))
    gradients, dx = mlp_backward(delta, layers, caches, activation, d_input_needed=True)
    epsilon = 1e-5
    for params, grads in zip(layers, gradients):
        for param, grad in zip(params, grads):
            numeric = np.zeros_like(param)
            for index in np.ndindex(param.shape):
                value = param[index]
                param[index] = value + epsilon
                above = objective()
                param[index] = value - epsilon
                below = objective()
                param[index] = value
                numeric[index] = (above - below) / (2 * epsilon)
            np.testing.assert_allclose(grad, numeric, atol=2e-5, rtol=2e-4)
    numeric_dx = np.empty_like(X)
    for index in np.ndindex(X.shape):
        value = X[index]
        X[index] = value + epsilon
        above = objective()
        X[index] = value - epsilon
        below = objective()
        X[index] = value
        numeric_dx[index] = (above - below) / (2 * epsilon)
    np.testing.assert_allclose(dx, numeric_dx, atol=2e-5, rtol=2e-4)


def test_adam_bias_correction_matches_first_step_reference():
    layers = [[np.array([[0.5]], dtype=np.float32), np.array([0.2], dtype=np.float32)]]
    Adam(layers, lr=0.01).step(layers, [[np.array([[2.0]]), np.array([-3.0])]])
    np.testing.assert_allclose(layers[0][0], [[0.49]], atol=1e-7)
    np.testing.assert_allclose(layers[0][1], [0.21], atol=1e-7)


def test_bce_with_extreme_logits_is_finite():
    logits = np.array([-1000.0, 1000.0])
    assert _bce_with_logits(logits, np.array([0.0, 1.0])) == 0.0
    assert _bce_with_logits(logits, np.array([1.0, 0.0])) == 1000.0


def test_som_generation_neighbors_match_training_hexagon_geometry():
    rows, cols = 5, 5
    coords = grid_coords(rows, cols)
    for code in range(rows * cols):
        geometric = set(np.flatnonzero(np.isclose(np.linalg.norm(coords - coords[code], axis=1), 1)))
        assert set(_neighbors(code, rows, cols)) == geometric
    assert len(_neighbors(12, rows, cols)) == 6


def test_single_neuron_som_can_generate_without_a_neighbor():
    prototype = np.full((1, 784), 0.4, dtype=np.float32)
    images, starts = sample_synthetic(prototype, 1, 1, np.ones(1), 3, noise=0)
    np.testing.assert_allclose(images, np.repeat(prototype, 3, axis=0), atol=1e-7)
    np.testing.assert_array_equal(starts, np.zeros(3))


@pytest.mark.parametrize("changes", [{"alpha": np.nan}, {"gamma": np.inf},
                                     {"rows": 0}, {"n_epochs": 0}])
def test_som_rejects_invalid_training_parameters(changes):
    params = dict(rows=1, cols=2, n_epochs=1, alpha=0.1, gamma=1.0, seed=42, verbose=False)
    params.update(changes)
    with pytest.raises(ValueError):
        fit_kohonen(np.zeros((3, 4)), **params)


def test_pca_singleton_class_gaussian_is_finite_and_has_correct_dimension():
    X = np.array([[0., 0., 0.], [1., 2., 0.], [2., 1., 1.]])
    pca = fit_pca(X, 2)
    groups = fit_latent_gaussian(pca, X, y=np.array([0, 1, 1]))
    mean, cov = groups[0]
    np.testing.assert_allclose(cov, 1e-6 * np.eye(2))
    assert sample_latent(mean, cov, 4).shape == (4, 2)


def test_pca_full_variance_fraction_never_exceeds_feature_count():
    X = np.random.default_rng(2).normal(size=(15, 7))
    full = fit_pca(X, n_components=1.0)
    assert full.n_components_ <= X.shape[1]
    truncated = fit_pca(X, n_components=1)
    with pytest.raises(ValueError, match="spectre"):
        components_for_variance(truncated, 0.99)


@pytest.mark.parametrize("changes", [{"epochs": 0}, {"batch_size": -1},
                                     {"learning_rate": np.nan}, {"latent_dim": 1.5}])
def test_neural_training_budget_is_validated(changes):
    cfg = dict(latent_dim=2, epochs=1, batch_size=8, learning_rate=0.001, seed=42)
    cfg.update(changes)
    with pytest.raises(ValueError):
        validate_training_config(cfg)
