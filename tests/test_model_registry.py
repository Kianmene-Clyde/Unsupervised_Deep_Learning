import json

import numpy as np
import pytest

from utils.model_registry import (
    delete_model,
    list_models,
    load_model,
    load_metadata,
    model_path,
    save_model,
    validate_model_name,
)


@pytest.mark.parametrize(
    "name",
    ["", " ", ".", "..", "../escape", r"..\escape", "a/b", "a:b", "model.npz", "CON"],
)
def test_invalid_model_names_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        model_path(tmp_path, name)


def test_registry_round_trip_and_delete(tmp_path):
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = save_model(tmp_path, "modèle sûr", "weights", values, {"k": 3})

    assert path.parent == tmp_path.resolve()
    assert list_models(tmp_path) == ["modèle sûr"]
    loaded, metadata = load_model(tmp_path, "modèle sûr", "weights")
    np.testing.assert_array_equal(loaded, values)
    assert metadata == {"k": 3}
    assert delete_model(tmp_path, "modèle sûr") is True
    assert delete_model(tmp_path, "modèle sûr") is False


def test_registry_rejects_nonfinite_arrays_and_metadata(tmp_path):
    with pytest.raises(ValueError, match="NaN"):
        save_model(tmp_path, "bad", "weights", [np.nan], {})
    with pytest.raises(ValueError, match="JSON"):
        save_model(tmp_path, "bad", "weights", [1.0], {"bad": object()})


def test_corrupt_model_has_clear_error(tmp_path):
    path = model_path(tmp_path, "corrupt")
    path.write_text(json.dumps({"not": "npz"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Impossible de charger"):
        load_model(tmp_path, "corrupt", "weights")


def test_valid_name_is_preserved():
    assert validate_model_name("mnist_k10-v2") == "mnist_k10-v2"


def test_listing_ignores_unsafe_legacy_filename(tmp_path):
    (tmp_path / "legacy.npz.npz").write_bytes(b"not relevant")
    assert list_models(tmp_path) == []


def test_metadata_read_does_not_deserialize_weight_member(tmp_path):
    path = model_path(tmp_path, "metadata_only")
    # A pickle-requiring weight member must not be touched by a metadata read.
    np.savez(path, weights=np.array([object()], dtype=object),
             metadata=json.dumps({"dataset": "mnist"}))
    assert load_metadata(tmp_path, "metadata_only") == {"dataset": "mnist"}
    with pytest.raises(ValueError, match="Impossible de charger"):
        load_model(tmp_path, "metadata_only", "weights")


@pytest.mark.parametrize("raw", ['{"loss": NaN}', '{"loss": Infinity}', '[]'])
def test_loading_rejects_invalid_or_nonfinite_metadata(tmp_path, raw):
    np.savez(model_path(tmp_path, "invalid_meta"), weights=[1.0], metadata=raw)
    with pytest.raises(ValueError):
        load_metadata(tmp_path, "invalid_meta")
    with pytest.raises(ValueError):
        load_model(tmp_path, "invalid_meta", "weights")


def test_npy_disguised_as_npz_has_clear_error(tmp_path):
    with model_path(tmp_path, "wrong_format").open("wb") as stream:
        np.save(stream, np.arange(3))
    with pytest.raises(ValueError, match="Archive NPZ attendue"):
        load_model(tmp_path, "wrong_format", "weights")
    with pytest.raises(ValueError, match="Archive NPZ attendue"):
        load_metadata(tmp_path, "wrong_format")
