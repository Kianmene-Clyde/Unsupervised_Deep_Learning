import numpy as np

from data_import import Dataset
from experiments import run_experiments as sweep


def test_sweep_partitions_do_not_overlap_or_use_official_test_for_validation(monkeypatch):
    X = np.repeat(np.arange(60, dtype=np.float32)[:, None], 784, axis=1)
    y = np.repeat(np.arange(3), 20)
    Xtest = np.full((9, 784), 1000, dtype=np.float32)
    ds = Dataset(X, y, Xtest, np.repeat(np.arange(3), 3), ["a", "b", "c"], "fixture")
    monkeypatch.setattr(sweep, "load_dataset", lambda *args, **kwargs: ds)
    loaded = sweep.load_data("mnist", 24, 6, 42, 2, n_validation=9)
    _, Xtr, _, Xval, _, Xte, _ = loaded
    assert Xtr.shape == (24, 784) and Xval.shape == (9, 784)
    assert not set(Xtr[:, 0]) & set(Xval[:, 0])
    assert max(Xval[:, 0]) < 1000
    assert np.all(Xte == 1000)


def test_only_validation_winner_is_evaluated_on_test(monkeypatch):
    class FakeAlgo:
        def check(self, params, n):
            pass
        def train(self, X, params):
            return params["offset"], {"k": 1, "inertia": len(X) * 784 * 0.1}
    evaluated = []
    def reconstruct(key, algo, weights, meta, X):
        evaluated.append((weights, float(X[0, 0])))
        return X + weights
    monkeypatch.setattr(sweep, "reconstruct_for", reconstruct)
    algo, candidates = FakeAlgo(), {}
    Xtr = np.zeros((4, 784))
    Xval = np.ones((2, 784))
    for offset in (0.2, 0.1, 0.3):
        record, _ = sweep.run_one("pca", algo, {"offset": offset}, Xtr, Xval,
                                  "fixture", 4, False, set(), candidates)
        assert "test_mse" not in record
    assert evaluated == [(0.2, 1.0), (0.1, 1.0), (0.3, 1.0)]
    results = sweep.evaluate_selected(candidates, {"pca": algo}, np.full((2, 784), 2.0))
    assert evaluated[-1] == (0.1, 2.0)
    assert len(results) == 1
    np.testing.assert_allclose(results[0]["test_mse"], 0.01)


def test_json_export_is_strict_and_marks_undefined_metrics(tmp_path):
    import json
    path = tmp_path / "details.json"
    sweep.write_json(path, {"metric": float("nan"), "value": np.int64(3)})
    assert json.loads(path.read_text()) == {"metric": None, "value": 3}


def test_selected_vae_rebuilds_after_its_model_cache_is_evicted():
    import pytest
    pytest.importorskip("tensorflow")
    from vae.algo import VAE
    from vae.utils import model
    rng = np.random.default_rng(12)
    Xtr = rng.random((20, 784), dtype=np.float32)
    Xval = rng.random((8, 784), dtype=np.float32)
    algo = VAE()
    params = {p.name: p.default for p in algo.params}
    params.update(latent_dim=2, hidden_dims=(8,), batch_size=16, epochs=1)
    candidates = {}
    record, _ = sweep.run_one("vae", algo, params, Xtr, Xval, "fixture", 20,
                              False, set(), candidates)
    assert record["status"] == "ok", record
    model._KNOWN.clear()
    model._BUILT.clear()
    results = sweep.evaluate_selected(candidates, {"vae": algo}, Xval)
    assert np.isfinite(results[0]["test_mse"])
    assert results[0]["test_mse"] == pytest.approx(record["validation_mse"], rel=1e-5)
