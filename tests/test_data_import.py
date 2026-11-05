import numpy as np
import pytest

import data_import
from data_import import (download_bitmap, load_quickdraw, split_train_validation,
                         stratified_split_indices)


@pytest.mark.parametrize("name", ["", "../escape", r"..\escape", "a/b", "a:b"])
def test_quickdraw_cache_rejects_unsafe_class_names(name):
    with pytest.raises(ValueError, match="invalide"):
        download_bitmap(name)


def test_quickdraw_parameters_are_validated_before_io():
    with pytest.raises(ValueError, match="uniques"):
        load_quickdraw(classes=["cat", "cat"])
    with pytest.raises(ValueError, match="max_per_class"):
        load_quickdraw(classes=["cat"], max_per_class=1)
    with pytest.raises(ValueError, match="test_ratio"):
        load_quickdraw(classes=["cat"], test_ratio=1.0)


def test_stratified_split_is_disjoint_repeatable_and_preserves_every_class():
    y = np.repeat([0, 1, 2], [2, 7, 11])
    train, validation = stratified_split_indices(y, 0.2, seed=19)
    again = stratified_split_indices(y, 0.2, seed=19)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(validation, again[1])
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(len(y)))
    assert set(y[train]) == set(y[validation]) == {0, 1, 2}
    X = np.arange(len(y)).reshape(-1, 1)
    Xtr, ytr, Xval, yval = split_train_validation(X, y, seed=19)
    np.testing.assert_array_equal(Xtr[:, 0], train)
    np.testing.assert_array_equal(yval, y[validation])


def test_stratified_split_rejects_singleton_classes_and_invalid_shapes():
    with pytest.raises(ValueError, match="au moins deux"):
        stratified_split_indices(np.array([0, 0, 1]))
    with pytest.raises(ValueError, match="même longueur"):
        split_train_validation(np.ones((3, 2)), np.array([0, 0]))
    with pytest.raises(ValueError, match="labels entiers"):
        stratified_split_indices(np.array([0.0, np.nan]))


def test_quickdraw_tiny_dataset_has_each_class_in_train_and_test(tmp_path, monkeypatch):
    for label, name in enumerate(["cat", "car"]):
        images = np.full((2, 784), label * 50, dtype=np.uint8)
        images[:, 0] = np.arange(2) + label * 10
        np.save(tmp_path / f"{name}.npy", images)
    monkeypatch.setattr(data_import, "BITMAP_DIR", tmp_path)
    ds = load_quickdraw(classes=["cat", "car"], max_per_class=2)
    assert set(ds.y_train) == set(ds.y_test) == {0, 1}
    assert not set(ds.X_train[:, 0]) & set(ds.X_test[:, 0])


def test_corrupted_download_is_never_promoted_to_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_import, "BITMAP_DIR", tmp_path)
    def invalid_download(url, target, reporthook):
        target.write_bytes(b"not an npy file")
    monkeypatch.setattr(data_import.urllib.request, "urlretrieve", invalid_download)
    with pytest.raises(ValueError, match="invalide"):
        download_bitmap("cat")
    assert list(tmp_path.iterdir()) == []


def test_existing_invalid_cache_fails_with_actionable_message(tmp_path, monkeypatch):
    monkeypatch.setattr(data_import, "BITMAP_DIR", tmp_path)
    (tmp_path / "cat.npy").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="Retéléchargez"):
        download_bitmap("cat")
