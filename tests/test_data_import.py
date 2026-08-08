import pytest

from data_import import download_bitmap, load_quickdraw


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
