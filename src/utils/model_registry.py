"""Safe, atomic storage shared by every algorithm model registry.

The public registries under ``src/<algo>/utils/registry.py`` deliberately stay
small and keep their algorithm-specific array key (``centroids`` or
``weights``).  This module centralises the guarantees that must not diverge:

* a model name can never escape its ``models`` directory;
* a save is atomic, so an interrupted process cannot leave a half-written NPZ;
* arrays and JSON metadata are validated when written and when loaded;
* ``allow_pickle=False`` remains enforced.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


MAX_MODEL_NAME_LENGTH = 128
_INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_model_name(name: str) -> str:
    """Return a safe model name or raise ``ValueError`` with a UI-ready message."""
    if not isinstance(name, str):
        raise ValueError("Le nom du modèle doit être une chaîne de caractères.")
    if not name or name != name.strip():
        raise ValueError("Le nom du modèle ne peut pas être vide ni entouré d'espaces.")
    if len(name) > MAX_MODEL_NAME_LENGTH:
        raise ValueError(
            f"Le nom du modèle est trop long ({len(name)} caractères, "
            f"maximum {MAX_MODEL_NAME_LENGTH})."
        )
    if name in {".", ".."} or _INVALID_WINDOWS_CHARS.search(name):
        raise ValueError(
            "Le nom du modèle contient un caractère interdit "
            "(< > : \" / \\ | ? * ou caractère de contrôle)."
        )
    if name.endswith((".", " ")):
        raise ValueError("Le nom du modèle ne peut pas finir par un point ou un espace.")
    if name.lower().endswith(".npz"):
        raise ValueError("Saisis le nom sans l'extension .npz.")
    if name.split(".", 1)[0].upper() in _RESERVED_WINDOWS_NAMES:
        raise ValueError(f"{name!r} est un nom réservé sous Windows.")
    return name


def model_path(models_dir: Path, name: str) -> Path:
    """Resolve a validated model name strictly inside ``models_dir``."""
    safe_name = validate_model_name(name)
    base = Path(models_dir).resolve()
    path = (base / f"{safe_name}.npz").resolve()
    if path.parent != base:  # defensive invariant if validation changes later
        raise ValueError("Le chemin du modèle doit rester dans son registre.")
    return path


def list_models(models_dir: Path) -> list[str]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []
    names = []
    for path in models_dir.glob("*.npz"):
        if not path.is_file():
            continue
        try:
            if model_path(models_dir, path.stem) == path.resolve():
                names.append(path.stem)
        except ValueError:
            # Ignore legacy or externally-created files with an unsafe name.
            continue
    return sorted(names)


def _validated_array(values, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"{label} ne peut pas être vide.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contient des valeurs NaN ou infinies.")
    return np.ascontiguousarray(array)


def _metadata_json(metadata: Mapping) -> str:
    if not isinstance(metadata, Mapping):
        raise ValueError("Les métadonnées du modèle doivent être un dictionnaire.")
    try:
        return json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Métadonnées non sérialisables en JSON : {exc}") from exc


def _read_metadata(data, path):
    raw_metadata = data["metadata"]
    if raw_metadata.size != 1:
        raise ValueError("Les métadonnées doivent être une chaîne JSON scalaire.")
    metadata = json.loads(str(raw_metadata.item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"Métadonnées invalides dans {path.name} : dictionnaire attendu.")
    # json.loads accepts NaN/Infinity by default; the writer deliberately does not.
    _metadata_json(metadata)
    return metadata


def _open_archive(path):
    archive = np.load(path, allow_pickle=False)
    if not hasattr(archive, "files"):
        raise ValueError("Archive NPZ attendue, et non un tableau NPY.")
    return archive


def load_metadata(models_dir: Path, name: str):
    """Read only the NPZ metadata member; large weights stay compressed on disk."""
    path = model_path(models_dir, name)
    try:
        with _open_archive(path) as data:
            return _read_metadata(data, path)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Impossible de lire les métadonnées de {path.name} : {exc}") from exc


def save_model(
    models_dir: Path,
    name: str,
    array_key: str,
    values,
    metadata: Mapping,
) -> Path:
    """Atomically save one float32 array and its JSON metadata."""
    path = model_path(models_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = _validated_array(values, array_key)
    metadata_text = _metadata_json(metadata)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix=f".{path.stem}-",
            dir=path.parent, delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            np.savez_compressed(stream, **{array_key: array, "metadata": metadata_text})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return path


def load_model(models_dir: Path, name: str, array_key: str):
    """Load and validate one model without ever enabling pickle."""
    path = model_path(models_dir, name)
    if not path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {path}")
    try:
        with _open_archive(path) as data:
            missing = {array_key, "metadata"} - set(data.files)
            if missing:
                raise ValueError(
                    f"Modèle corrompu {path.name} : clés manquantes {sorted(missing)}."
                )
            values = _validated_array(data[array_key], array_key).copy()
            metadata = _read_metadata(data, path)
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Impossible de charger le modèle {path.name} : {exc}") from exc
    return values, metadata


def delete_model(models_dir: Path, name: str) -> bool:
    path = model_path(models_dir, name)
    if not path.exists():
        return False
    path.unlink()
    return True
