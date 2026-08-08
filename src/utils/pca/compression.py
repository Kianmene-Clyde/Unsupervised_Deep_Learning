"""PCA image compression -- the quality-vs-size trade-off.

PCA gives a lossy image codec: keep only the top `k` principal components, and
each 784-pixel image is stored as just `k` numbers (its "code"). Reconstruction
maps those `k` codes back into pixel space. Fewer components -> smaller codes
(higher compression ratio) but blurrier reconstructions.

The PCA itself is **not** re-implemented here -- `fit_pca` / `PCA` are imported
from `utils.pca.dim_reduction`, the project's single PCA. This module only adds
the compression-specific pieces on top:

    Encode / decode / measure:
        - compress(pca, X):               images -> low-dim codes (== pca.transform)
        - reconstruct(pca, codes):        codes -> reconstructed images in [0, 1]
        - reconstruction_error(X, Xhat):  per-pixel mean squared error
        - compression_ratio(k):           784 / k, the storage saving factor
        - compression_sweep(X, ks):       MSE + ratio for a list of k
    Plot (used by src/pca/compression.ipynb):
        - plot_reconstruction_grid(...):  originals vs reconstructions at rising k
        - plot_tradeoff(...):             MSE and compression ratio vs k
        - plot_normalized_comparison(...):normalized MSE curves for >=1 dataset
"""

from __future__ import annotations

import numpy as np

from utils.mnist_data import N_FEATURES, as_images
from utils.pca.dim_reduction import PCA, fit_pca


def compress(pca: PCA, X):
    """Project images onto the top principal components (the encode step).

    Thin, intention-revealing alias for `pca.transform` in a compression context.

    Returns
    -------
    np.ndarray, shape (n_samples, n_components)
        The low-dimensional codes -- all that must be stored per image.
    """
    return pca.transform(X)


def reconstruct(pca: PCA, codes):
    """Map low-dim codes back into image space (the decode step).

    The inverse transform can produce values slightly outside [0, 1], so the
    result is clipped back to the valid pixel range.

    Returns
    -------
    np.ndarray, shape (n_samples, 784), float in [0, 1]
    """
    X_hat = pca.inverse_transform(codes)
    return np.clip(X_hat, 0.0, 1.0)


def reconstruction_error(X, X_hat) -> float:
    """Per-pixel mean squared error between originals and reconstructions."""
    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)
    return float(np.mean((X - X_hat) ** 2))


def compression_ratio(n_components: int, n_features: int = N_FEATURES) -> float:
    """Storage saving factor from keeping `n_components` instead of every pixel.

    e.g. 784 / 20 = 39.2x smaller codes.
    """
    return n_features / n_components


def compression_sweep(X, ks, center="feature", verbose: bool = True):
    """Fit PCA at each `k`, reconstruct the whole set, and measure quality/size.

    Parameters
    ----------
    X : (n, 784) design matrix.
    ks : iterable of int -- component counts to try.
    center : {"feature", "global"} -- centering passed to fit_pca. "global"
        wastes a component on the mean image, so its reconstructions are slightly
        worse at every k (see src/pca/dim_reduction.ipynb).
    verbose : print a per-k table if True.

    Returns
    -------
    mses : np.ndarray -- per-pixel reconstruction MSE at each k.
    ratios : np.ndarray -- compression ratio (784 / k) at each k.
    total_variance : float -- mean per-pixel variance (the MSE at k=0, i.e.
        reconstructing with only the mean). Divide MSE by this to compare
        datasets of different ink density on the same scale.
    """
    total_variance = float(np.mean(np.var(X, axis=0)))
    mses, ratios = [], []
    for k in ks:
        pca = fit_pca(X, k, center=center)
        X_hat = reconstruct(pca, compress(pca, X))
        mses.append(reconstruction_error(X, X_hat))
        ratios.append(compression_ratio(k))
    mses, ratios = np.array(mses), np.array(ratios)

    if verbose:
        print(f"total per-pixel variance = {total_variance:.5f}")
        for k, m, r in zip(ks, mses, ratios):
            print(f"k={k:>3}  ratio={r:6.2f}x  MSE={m:.5f}  normalized={m / total_variance:.3f}")
    return mses, ratios, total_variance


# --------------------------------------------------------------------------- #
# Plotting (src/pca/compression.ipynb)
# --------------------------------------------------------------------------- #
def plot_reconstruction_grid(X, y, class_names, ks, n_per_class=2, seed=0,
                             center="feature", suptitle=None) -> None:
    """Show sample images (top row) vs their reconstructions at each k (rows below).

    Picks `n_per_class` random samples of every class, fits one PCA per k on the
    full `X` (with the given `center`), and decodes the samples. Reading
    top-to-bottom shows detail returning as k grows.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    sample_idx = []
    for i in range(len(class_names)):
        pool = np.flatnonzero(y == i)
        sample_idx.extend(rng.choice(pool, size=n_per_class, replace=False))
    sample_idx = np.array(sample_idx)
    X_samp = X[sample_idx]

    recons = {}
    for k in ks:
        pca = fit_pca(X, k, center=center)
        recons[k] = reconstruct(pca, compress(pca, X_samp))

    n_show = len(sample_idx)
    rows = len(ks) + 1
    fig, axes = plt.subplots(rows, n_show, figsize=(1.4 * n_show, 1.4 * rows))
    for j in range(n_show):
        axes[0, j].imshow(as_images(X_samp[j:j + 1])[0], cmap="gray")
        axes[0, j].set_title(class_names[y[sample_idx[j]]], fontsize=9)
        axes[0, j].axis("off")
    for i, k in enumerate(ks, start=1):
        for j in range(n_show):
            axes[i, j].imshow(as_images(recons[k][j:j + 1])[0], cmap="gray")
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(f"k={k}", rotation=0, ha="right", va="center", fontsize=10)
    axes[0, 0].set_ylabel("original", rotation=0, ha="right", va="center", fontsize=10)
    fig.suptitle(suptitle or "Original (top) vs PCA reconstruction at increasing k", y=1.01)
    plt.tight_layout()
    plt.show()


def plot_tradeoff(ks, mses, ratios, suptitle=None) -> None:
    """Twin-axis plot: reconstruction MSE (falling) and compression ratio (falling) vs k."""
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, mses, "o-", color="tab:blue", label="reconstruction MSE")
    ax1.set_xlabel("number of components k")
    ax1.set_ylabel("reconstruction MSE (per pixel)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(ks, ratios, "s--", color="tab:red", label="compression ratio")
    ax2.set_ylabel("compression ratio (784 / k)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_yscale("log")

    ax1.set_title(suptitle or "Quality vs. size: MSE falls, compression ratio drops as k grows")
    fig.tight_layout()
    plt.show()


def plot_normalized_comparison(ks, series, suptitle=None) -> None:
    """Compare datasets on **normalized** MSE = MSE / total variance.

    Normalizing by each dataset's own variance (the fraction of variance left
    unexplained) makes datasets of different ink density comparable -- raw MSE
    is not, since a mostly-blank sketch has little error to make.

    Parameters
    ----------
    series : list of (name, mses, total_variance, color)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, mses, var, color in series:
        ax.plot(ks, np.asarray(mses) / var, "o-", color=color, label=name)
    ax.set_xlabel("number of components k")
    ax.set_ylabel("normalized MSE  (fraction of variance unexplained)")
    ax.set_title(suptitle or "Compression quality across datasets")
    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, y = load_digits(per_class=1000)
    print(f"Loaded X {X.shape}")
    compression_sweep(X, (10, 50, 100))
