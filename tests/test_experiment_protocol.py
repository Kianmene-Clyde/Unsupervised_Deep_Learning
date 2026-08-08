import numpy as np

from experiments.compare_all import balanced_subset, generation_fid
from utils.metrics_common import fit_pca_features, frechet_distance, pca_features


def test_balanced_subset_is_repeatable_and_balanced():
    X = np.arange(90).reshape(30, 3)
    y = np.repeat([0, 1, 2], 10)
    first_X, first_y = balanced_subset(X, y, 14, seed=42)
    second_X, second_y = balanced_subset(X, y, 14, seed=42)
    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    counts = np.bincount(first_y, minlength=3)
    assert counts.max() - counts.min() <= 1


def test_shared_fid_context_matches_direct_feature_comparison():
    real = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    generated = real + np.array([0.5, -0.25])
    featurizer = fit_pca_features(real, n_components=2)
    context = (featurizer, pca_features(real, featurizer))
    expected = frechet_distance(context[1], pca_features(generated, featurizer))
    assert generation_fid(real, generated, context) == expected
