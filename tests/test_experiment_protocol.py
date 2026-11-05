import numpy as np

from experiments.compare_all import (
    balanced_subset, build_fd_context, generation_fid, main, summarize_runs,
)
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


def test_fd_pca_basis_is_fitted_on_train_not_reference():
    # Training varies only horizontally; evaluation varies only vertically.
    # Fitting on evaluation would report a large vertical displacement instead.
    train = np.array([[-2., 0.], [0., 0.], [2., 0.]])
    reference = np.array([[0., -5.], [0., 0.], [0., 5.]])
    context = build_fd_context(train, reference, n_components=1)
    assert generation_fid(reference, reference + [0., 100.], context) == 0.
    assert generation_fid(reference, reference + [2., 0.], context) == 4.
    np.testing.assert_array_equal(context[0][0], train.mean(axis=0))


def test_balanced_subset_rejects_fractional_sizes():
    import pytest

    with pytest.raises(ValueError, match="entier"):
        balanced_subset(np.ones((5, 2)), np.zeros(5), 2.5, seed=42)


def test_summary_reports_sample_std_and_no_single_seed_precision():
    repeated = summarize_runs([
        {"algo": "PCA", "seed": 42, "mse": 1.},
        {"algo": "PCA", "seed": 43, "mse": 3.},
    ])
    mse_row = next(row for row in repeated if row["metric"] == "mse")
    assert mse_row["mean"] == 2.
    assert mse_row["std"] == np.sqrt(2.)
    single = summarize_runs([{"algo": "PCA", "seed": 42, "mse": 1.}])
    assert np.isnan(single[0]["std"])


def test_existing_scientific_results_are_not_overwritten(tmp_path):
    import pytest

    result = tmp_path / "results_mnist.csv"
    result.write_text("historical results", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path)])
    assert result.read_text(encoding="utf-8") == "historical results"
