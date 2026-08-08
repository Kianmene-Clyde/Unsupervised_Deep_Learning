"""Point d'entrée unique pour charger les datasets du projet.

Tous les algos (K-means, autoencodeurs…) passent par `load_dataset(key)` et
récupèrent le même format, quel que soit le dataset :

    X_train (n, 784) float32 dans [0, 1]   ·   y_train (n,) int
    X_test  (m, 784) float32 dans [0, 1]   ·   y_test  (m,) int
    class_names : le nom lisible de chaque label

Trois datasets sont disponibles :
    - "mnist"       : chiffres manuscrits 0-9, via Keras.
    - "quickdraw"   : les classes de CLASSES (cat, apple, car), via les bitmaps
                      28x28 officiels de Google.
    - "quickdraw10" : même source, mais les 10 classes de CLASSES_10 — un
                      Quick, Draw! aussi « large » que MNIST (10 classes).

--- Deux formats Quick, Draw!, à ne pas confondre ---

Google publie ses dessins sous deux formes, et ce module expose les deux :

    .bin  (vectoriel) -> load_group() / drawing_to_record()
        Des tracés (listes de points) + métadonnées, via le paquet `quickdraw`.
        C'est ce qu'explore `exploration.ipynb`. Inutilisable tel quel pour le
        clustering : ce ne sont pas des images.

    .npy  (bitmap)    -> load_quickdraw()
        Des images 28x28 déjà rendues, de forme (N, 784) uint8 — exactement le
        format de MNIST. C'est ce dont le K-means a besoin.

Rendre soi-même les tracés en 28x28 donnerait des images légèrement différentes
de celles de Google, pour un coût de calcul inutile : on télécharge les .npy.
"""

import urllib.parse
import urllib.request
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Les trois classes retenues pour le projet.
CLASSES = ["cat", "apple", "car"]

# La variante à 10 classes : les 3 d'origine plus 7 formes bien distinctes
# les unes des autres, pour comparer les algos à « largeur » égale avec MNIST
# (10 classes aussi). Chaque nom correspond à un .npy officiel de Google.
CLASSES_10 = CLASSES + ["fish", "house", "tree", "clock", "star", "umbrella", "airplane"]

CACHE_DIR = ROOT / "data" / "binary"          # .bin vectoriels (paquet quickdraw)
BITMAP_DIR = ROOT / "data" / "numpy_bitmap"   # .npy bitmaps 28x28 (clustering)
BITMAP_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"

IMAGE_DIM = 784   # 28 * 28
HEAD = 3          # nombre de dessins chargés/affichés par classe dans le CLI


class Dataset(NamedTuple):
    """Un dataset prêt à l'emploi, au même format quelle que soit la source."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_names: list
    name: str

    def describe(self) -> str:
        return (
            f"{self.name} — {len(self.X_train)} train / {len(self.X_test)} test · "
            f"{len(self.class_names)} classes : {', '.join(self.class_names)}"
        )


# Catalogue des datasets, pour peupler un sélecteur d'UI sans le coder en dur.
DATASETS = {
    "mnist": "MNIST — chiffres manuscrits 0-9",
    "quickdraw": f"Quick, Draw! — {', '.join(CLASSES)}",
    "quickdraw10": f"Quick, Draw! 10 — {len(CLASSES_10)} classes ({CLASSES_10[0]}, {CLASSES_10[1]}…)",
}


# ------------------------------------------------------- Quick, Draw! bitmaps


def download_bitmap(name: str, progress=None) -> Path:
    """Télécharge (une fois) le .npy 28x28 d'une classe et renvoie son chemin.

    progress: callable(fraction, desc) optionnel — branché sur gr.Progress côté app.
    """
    if (not isinstance(name, str) or not name or name != name.strip()
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name in {".", ".."}):
        raise ValueError(f"Nom de classe Quick Draw invalide : {name!r}.")
    BITMAP_DIR.mkdir(parents=True, exist_ok=True)
    base = BITMAP_DIR.resolve()
    path = (base / f"{name}.npy").resolve()
    if path.parent != base:
        raise ValueError("Le fichier Quick Draw doit rester dans le cache prévu.")
    if path.exists():
        return path

    # Les noms à plusieurs mots ("power outlet") doivent être encodés dans l'URL.
    url = BITMAP_URL.format(urllib.parse.quote(name, safe=""))

    def hook(blocks, block_size, total):
        if progress is not None and total > 0:
            done = min(blocks * block_size / total, 1.0)
            progress(done, desc=f"Téléchargement {name}.npy ({total / 1024**2:.0f} Mo)…")

    # On écrit dans un .part renommé à la fin : une coupure réseau laisserait
    # sinon un .npy tronqué en cache, que les runs suivants croiraient valide.
    tmp = path.with_suffix(".npy.part")
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(path)
    return path


def load_quickdraw(
    classes=None,
    max_per_class: int = 10000,
    test_ratio: float = 1 / 7,
    seed: int = 42,
    progress=None,
    name: str = "quickdraw",
) -> Dataset:
    """Charge les classes Quick, Draw! en bitmaps 28x28, prêtes pour le clustering.

    Args:
        classes:       noms des catégories ; CLASSES par défaut.
        max_per_class: dessins tirés par classe (les fichiers en contiennent
                       100 000+ : tout charger ferait plusieurs Go).
        test_ratio:    part réservée au test ; 1/7 reproduit le ratio de MNIST
                       (60 000 train / 10 000 test).
        seed:          rend le tirage et le découpage reproductibles.
        name:          clé du dataset ("quickdraw" ou "quickdraw10") — elle part
                       dans les métadonnées des modèles, qui filtrent dessus.
    """
    classes = list(CLASSES if classes is None else classes)
    if (not classes or not all(isinstance(name, str) and name for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("classes doit contenir des noms uniques et au moins une classe.")
    if not isinstance(max_per_class, (int, np.integer)) or max_per_class < 2:
        raise ValueError("max_per_class doit être un entier supérieur ou égal à 2.")
    if not np.isfinite(test_ratio) or not 0.0 < float(test_ratio) < 1.0:
        raise ValueError("test_ratio doit être strictement compris entre 0 et 1.")
    rng = np.random.default_rng(seed)

    X_parts, y_parts = [], []
    # « cls » et non « name » : la variable de boucle écrasait le paramètre
    # name de la fonction — tous les datasets Quick, Draw! se retrouvaient
    # nommés d'après leur DERNIÈRE classe (« car ») dans les métadonnées.
    for label, cls in enumerate(classes):
        path = download_bitmap(cls, progress)

        # mmap : le fichier fait ~100 Mo, on ne veut que max_per_class lignes.
        # Sans lui, np.load chargerait les 100 Mo pour en jeter 90 %.
        data = np.load(path, mmap_mode="r")
        if data.ndim != 2 or data.shape[1] != IMAGE_DIM or len(data) < 2:
            raise ValueError(
                f"Fichier Quick Draw invalide {path.name} : forme {data.shape}, "
                f"attendu (n, {IMAGE_DIM}) avec n >= 2."
            )
        n = min(max_per_class, len(data))

        # Tirage aléatoire plutôt que les n premiers : l'ordre du fichier n'est
        # pas garanti neutre (regroupements par pays ou par date possibles).
        # Indices triés -> lecture séquentielle du disque, bien plus rapide.
        idx = np.sort(rng.choice(len(data), size=n, replace=False))

        X_parts.append(np.asarray(data[idx], dtype=np.float32) / 255.0)
        y_parts.append(np.full(n, label, dtype=np.int64))

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)

    # Sans ce mélange, X serait trié par classe et un découpage par tranche
    # mettrait une classe entière dans le test.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    n_test = int(round(len(X) * test_ratio))
    n_test = min(max(n_test, 1), len(X) - 1)
    return Dataset(
        X_train=X[n_test:],
        y_train=y[n_test:],
        X_test=X[:n_test],
        y_test=y[:n_test],
        class_names=classes,
        name=name,
    )


# -------------------------------------------------------------------- MNIST


def load_mnist(progress=None) -> Dataset:
    """Charge MNIST au format commun (aplati, normalisé dans [0, 1])."""
    # Import local : TensorFlow met ~10 s à charger, inutile de le payer quand
    # on n'importe ce module que pour les helpers Quick, Draw!.
    from tensorflow.keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    def prepare(images):
        return (images.astype(np.float32) / 255.0).reshape(-1, IMAGE_DIM)

    return Dataset(
        X_train=prepare(x_train),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=prepare(x_test),
        y_test=np.asarray(y_test, dtype=np.int64),
        class_names=[str(d) for d in range(10)],
        name="mnist",
    )


# ---------------------------------------------------------------- Dispatch


def load_dataset(key: str = "mnist", **kwargs) -> Dataset:
    """Charge un dataset par sa clé — l'entrée unique pour tous les algos.

    kwargs (max_per_class, seed…) ne concernent que "quickdraw" et sont ignorés
    pour MNIST, dont le découpage train/test est fixé par Keras.
    """
    if key == "mnist":
        return load_mnist(progress=kwargs.get("progress"))
    if key == "quickdraw":
        return load_quickdraw(**kwargs)
    if key == "quickdraw10":
        # 6 000 dessins par classe et non 10 000 : 10 classes × 6 000 = 60 000
        # images, la taille exacte de MNIST — les comparaisons restent à volume
        # égal, et la RAM ne double pas au passage à 10 classes.
        kwargs.setdefault("max_per_class", 6000)
        return load_quickdraw(classes=CLASSES_10, name="quickdraw10", **kwargs)
    raise ValueError(f"Dataset inconnu : {key!r}. Attendu : {list(DATASETS)}")


# ------------------------------- Quick, Draw! vectoriel (exploration.ipynb)


def load_group(name: str, max_drawings: int = HEAD):
    """Télécharge (au premier appel) et charge une classe Quick, Draw! vectorielle.

    Renvoie des tracés + métadonnées, PAS des images : voir load_quickdraw()
    pour les bitmaps 28x28 utilisés par le clustering.
    """
    from quickdraw import QuickDrawDataGroup

    return QuickDrawDataGroup(name, max_drawings=max_drawings, cache_dir=str(CACHE_DIR))


def drawing_to_record(drawing) -> dict:
    """Aplatit les métadonnées d'un QuickDrawing en dict (pratique pour un DataFrame)."""
    return {
        "key_id": drawing.key_id,
        "country": drawing.countrycode,
        "recognized": drawing.recognized,
        "timestamp": drawing.timestamp,
        "strokes": drawing.no_of_strokes,
    }


def main() -> None:
    for name in CLASSES:
        group = load_group(name)
        print(f"=== {name} (head of {HEAD}) ===")
        for d in group.drawings:
            r = drawing_to_record(d)
            print(
                f"key_id={r['key_id']} country={r['country']} "
                f"recognized={r['recognized']} timestamp={r['timestamp']} "
                f"strokes={r['strokes']}"
            )
        print()


if __name__ == "__main__":
    main()
