"""Quick, Draw! loader — metadata inspection *and* a flat design matrix.

Two jobs, sharing one download/cache path:

1. **Metadata inspection** (used by `exploration.ipynb`) via `load_group` /
   `drawing_to_record` — thin wrappers over the `quickdraw` package, which
   downloads a category's `<name>.bin` from Google Cloud Storage on first use
   and caches it under `data/binary/`.

2. **A flat, scaled design matrix** (used by the PCA notebooks) via
   `load_sketches`, which renders each vector drawing to a 28x28 grayscale
   bitmap and flattens it. This mirrors the MNIST contract in
   `utils.mnist_data` *exactly*, so the same PCA helpers run on either dataset:

       X, y = load_sketches()
         X : np.ndarray, shape (n_samples, 784), float32 in [0, 1]  -- one flat sketch per row
         y : np.ndarray, shape (n_samples,),     int                -- class index per row

   Sketches are rendered to match MNIST's convention: **0 = background, 1 = ink**
   (the raw `quickdraw` image is white-on-black, so we invert it).
"""

from pathlib import Path

import numpy as np
from quickdraw import QuickDrawDataGroup

ROOT = Path(__file__).resolve().parents[2]  # src/utils/ -> src/ -> repo root

# Three distinct classes (categories) shared across exploration + PCA notebooks.
CLASSES = ["cat", "apple", "car"]
CACHE_DIR = ROOT / "data" / "binary"          # raw .bin files (managed by quickdraw)
SKETCH_CACHE_DIR = ROOT / "data" / "sketches"  # rendered (X, y) arrays (managed here)

IMG_SIZE = 28
N_FEATURES = IMG_SIZE * IMG_SIZE  # 784 -- same feature count as MNIST

HEAD = 3  # number of drawings to load & print per class in the CLI run


# --------------------------------------------------------------------------- #
# Metadata inspection (exploration.ipynb)
# --------------------------------------------------------------------------- #
def load_group(name: str, max_drawings: int | None = HEAD) -> QuickDrawDataGroup:
    """Download (on first use) and load one Quick, Draw! class.

    `max_drawings=None` loads the entire category; an int caps it.
    """
    return QuickDrawDataGroup(
        name, max_drawings=max_drawings, cache_dir=str(CACHE_DIR)
    )


def drawing_to_record(drawing) -> dict:
    """Flatten a QuickDrawing's metadata into a plain dict (nice for DataFrames)."""
    return {
        "key_id": drawing.key_id,
        "country": drawing.countrycode,
        "recognized": drawing.recognized,
        "timestamp": drawing.timestamp,
        "strokes": drawing.no_of_strokes,
    }


# --------------------------------------------------------------------------- #
# Flat design matrix (PCA notebooks)
# --------------------------------------------------------------------------- #
def render_drawing(drawing, img_size: int = IMG_SIZE) -> np.ndarray:
    """Render one vector `QuickDrawing` to a flat, [0, 1]-scaled ink vector.

    The raw `quickdraw` image is black strokes on a white background; we convert
    to grayscale, downscale to `img_size`x`img_size`, then **invert** so that
    0 = background and 1 = ink -- matching the MNIST contract.

    Returns
    -------
    np.ndarray, shape (img_size ** 2,), float32 in [0, 1]
    """
    from PIL import Image

    img = drawing.image.convert("L").resize((img_size, img_size), Image.LANCZOS)
    ink = 255.0 - np.asarray(img, dtype=np.float32)  # invert: ink now high
    return (ink / 255.0).reshape(-1)


def load_sketches(
    classes=CLASSES,
    per_class: int | None = None,
    img_size: int = IMG_SIZE,
    recognized: bool | None = None,
    seed: int = 0,
    cache: bool = True,
):
    """Load the selected Quick, Draw! classes as a flat, scaled matrix.

    Mirrors `utils.mnist_data.load_digits`: same shapes, same [0, 1] scaling,
    same 0 = background / 1 = ink convention, so the PCA helpers are reusable
    across both datasets unchanged.

    Rendering the full categories is expensive, so the resulting `(X, y)` is
    cached to `data/sketches/` as an `.npz`; later calls with the same
    parameters load instantly.

    Parameters
    ----------
    classes : iterable of str
        Quick, Draw! category names (default: cat, apple, car).
    per_class : int or None
        Keep at most this many (shuffled) drawings per class. None loads the
        entire category -- the full dataset.
    img_size : int
        Side length to render each sketch to (default 28, matching MNIST).
    recognized : bool or None
        If set, keep only drawings whose `recognized` flag matches.
    seed : int
        RNG seed for the per-class subsampling/shuffle (reproducible).
    cache : bool
        If True, read/write a rendered-array cache under `data/sketches/`.

    Returns
    -------
    X : np.ndarray, shape (n_samples, img_size ** 2), float32 in [0, 1]
    y : np.ndarray, shape (n_samples,), int  -- index into `classes` per row
    """
    classes = list(classes)
    cap = "all" if per_class is None else str(per_class)
    rec = "any" if recognized is None else ("rec" if recognized else "unrec")
    cache_path = SKETCH_CACHE_DIR / f"{'-'.join(classes)}_{cap}_{img_size}_{rec}.npz"

    if cache and cache_path.exists():
        data = np.load(cache_path)
        return data["X"], data["y"]

    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for label, name in enumerate(classes):
        group = QuickDrawDataGroup(
            name,
            max_drawings=per_class,
            recognized=recognized,
            cache_dir=str(CACHE_DIR),
        )
        drawings = list(group.drawings)
        if per_class is not None and len(drawings) > per_class:
            idx = rng.permutation(len(drawings))[:per_class]
            drawings = [drawings[i] for i in idx]
        vecs = np.stack([render_drawing(d, img_size) for d in drawings])
        X_parts.append(vecs)
        y_parts.append(np.full(len(drawings), label, dtype=int))

    X = np.concatenate(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Mix the classes so row order isn't blocked by category.
    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    if cache:
        SKETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def as_images(X, img_size: int = IMG_SIZE):
    """Reshape a flat design matrix (n, img_size**2) back to (n, s, s) for plotting."""
    X = np.asarray(X)
    return X.reshape(-1, img_size, img_size)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
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

    # Small rendered-matrix smoke test (few per class -> fast, no cache).
    X, y = load_sketches(per_class=50, cache=False)
    print(f"load_sketches -> X {X.shape} {X.dtype} range "
          f"[{X.min():.2f}, {X.max():.2f}], classes {sorted(set(y.tolist()))}")


if __name__ == "__main__":
    main()
