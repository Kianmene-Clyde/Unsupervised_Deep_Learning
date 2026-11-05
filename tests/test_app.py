"""Small offline UI/registry/export checks, without downloading any dataset."""

import importlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gradio")
pytest.importorskip("tensorflow")

import app
from data_import import Dataset


@pytest.fixture
def offline_app(monkeypatch, tmp_path):
    rng = np.random.default_rng(71)
    X = rng.random((30, 784), dtype=np.float32)
    ds = Dataset(X[:24], np.arange(24) % 2, X[24:], np.arange(6) % 2,
                 ["a", "b"], "synthetic-ui-validation")
    monkeypatch.setattr(app, "_LOADED", {"mnist": ds})
    app._META_CACHE.clear()
    for algo in app.ALGOS.values():
        module = importlib.import_module(type(algo).__module__.rsplit(".", 1)[0] + ".utils.registry")
        monkeypatch.setattr(module, "MODELS_DIR", tmp_path / algo.key)
    return ds


def save_kmeans(name="small", dataset="mnist", algo="kmeans"):
    weights = np.zeros((2, 784), dtype=np.float32)
    metadata = {"algo": algo, "dataset": dataset, "n_samples": 24, "k": 2,
                "inertia": 1.0, "inertia_history": [2.0, 1.0], "max_iter": 1,
                "tolerance": 1e-4, "seed": 42}
    app.ALGOS["kmeans"].save(name, weights, metadata)


def test_app_import_does_not_load_data():
    assert callable(app.build_app)
    # UI construction is explicit; there is no implicit global demo/server.
    assert not hasattr(app, "demo")


def test_model_cache_tracks_external_updates_and_excludes_wrong_provenance(offline_app):
    save_kmeans()
    assert app.models_for("kmeans", "mnist") == ["small"]
    save_kmeans(dataset="quickdraw")
    assert app.models_for("kmeans", "mnist") == []
    assert app.models_for("kmeans", "quickdraw") == ["small"]
    save_kmeans(dataset="mnist", algo="vae")
    assert app.models_for("kmeans", "mnist") == []


def test_build_and_export_include_portable_files_and_provenance(offline_app):
    save_kmeans()
    demo = app.build_app(initial_dataset=offline_app)
    assert len(demo.config["dependencies"]) > 10
    gallery, archive_path, note = app.export_all("kmeans", "mnist", "small", 8, 2, 5,
                                               progress=lambda *args, **kwargs: None)
    assert gallery and all(Path(path).is_file() for path in gallery)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["visualization"]["seed"] == 5
        assert manifest["model_metadata"]["dataset"] == "mnist"
        assert len(manifest["figures"]) == len(gallery)
        assert all(Path(name).name == name for name in archive.namelist())
    assert "manifest.json" in note


def test_export_figure_name_cannot_create_directories():
    name = app.figure_filename(10, 'Génération / Test: a? <b> \\ relative')
    assert name == "10_generation_test_a_b_relative.png"


def test_training_preserves_existing_model(offline_app):
    save_kmeans("same")
    path = app.ALGOS["kmeans"].model_path("same")
    before = path.read_bytes()
    values = [next(p for p in app.ALGOS[key].params if p.name == name).default
              for key, name in app.PARAM_SLOTS]
    for i, (key, name) in enumerate(app.PARAM_SLOTS):
        if key == "kmeans" and name == "k":
            values[i] = 2
        if key == "kmeans" and name == "max_iter":
            values[i] = 1
    update, _, status = app.train("kmeans", "mnist", 24, "same", *values)
    assert update["value"] == "same_v2"
    assert path.read_bytes() == before
    assert "conservé" in status
