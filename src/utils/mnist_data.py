"""Shared MNIST loader for the PCA work.

The whole PCA study (dimensionality reduction, compression, generation) runs on
**three MNIST digit classes: 0, 1 and 2**. MNIST digits are 28x28 grayscale
bitmaps (0 = background, 255 = ink). This module is the single source of truth
for turning them into a flat, [0, 1]-scaled design matrix `X` and label vector
`y`, so every notebook/module shares the exact same data contract.

Contract (used by src/utils/pca/*.py and src/pca/*.ipynb):
    X, y = load_digits()
      X : np.ndarray, shape (n_samples, 784), float32 in [0, 1]  -- one flat image per row
      y : np.ndarray, shape (n_samples,),     int                -- digit label per row
"""

from __future__ import annotations

import numpy as np

DIGITS = (0, 1, 2)  # the three MNIST classes used across the PCA notebooks
IMG_SIZE = 28
N_FEATURES = IMG_SIZE * IMG_SIZE  # 784


def load_digits(
    digits=DIGITS,
    per_class: int | None = None,
    split: str = "train",
    seed: int = 0,
):
    """Load the selected MNIST digit classes as a flat, scaled matrix.

    Parameters
    ----------
    digits : iterable of int
        Which digit classes to keep (default: 0, 1, 2).
    per_class : int or None
        If given, keep at most this many (shuffled) samples per class -- handy
        for keeping PCA notebooks fast. None keeps every matching sample.
    split : {"train", "test", "both"}
        Which MNIST split to draw from.
    seed : int
        RNG seed for the per-class subsampling/shuffle (reproducible).

    Returns
    -------
    X : np.ndarray, shape (n_samples, 784), float32 in [0, 1]
    y : np.ndarray, shape (n_samples,), int
    """
    from keras.datasets import mnist  # keras is bundled with TensorFlow

    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    if split == "train":
        images, labels = x_tr, y_tr
    elif split == "test":
        images, labels = x_te, y_te
    elif split == "both":
        images = np.concatenate([x_tr, x_te])
        labels = np.concatenate([y_tr, y_te])
    else:
        raise ValueError(f"split must be 'train', 'test' or 'both', got {split!r}")

    rng = np.random.default_rng(seed)
    keep_idx = []
    for d in digits:
        idx = np.flatnonzero(labels == d)
        rng.shuffle(idx)
        if per_class is not None:
            idx = idx[:per_class]
        keep_idx.append(idx) 
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)  # mix the classes so row order isn't blocked by digit

    X = images[keep_idx].reshape(len(keep_idx), N_FEATURES).astype(np.float32) / 255.0
    y = labels[keep_idx].astype(int)
    return X, y


def as_images(X):
    """Reshape a flat design matrix (n, 784) back to (n, 28, 28) for plotting."""
    X = np.asarray(X)
    return X.reshape(-1, IMG_SIZE, IMG_SIZE)


if __name__ == "__main__":
    X, y = load_digits(per_class=100)
    print(f"X {X.shape} {X.dtype} range [{X.min():.2f}, {X.max():.2f}]")
    print(f"y {y.shape} classes {sorted(set(y.tolist()))}")
