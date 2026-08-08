"""Generate synthetic images by sampling in PCA latent space.

A simple *linear-Gaussian* generative model: fit a PCA to get a low-dimensional
latent space, model the latent codes with a (multivariate) Gaussian, draw fresh
latent vectors from that Gaussian, and `inverse_transform` them back into 28x28
pixel space. Fitting one Gaussian per class lets us generate a chosen class on
demand.

The PCA is **not** re-implemented here -- `fit_pca` / `PCA` come from
`utils.pca.dim_reduction`, the project's single PCA. This module adds only the
generation-specific pieces:

    Model / sample:
        - fit_latent_gaussian(pca, X, y):  Gaussian(s) over the latent codes
        - sample_latent(mean, cov, n):     draw latent vectors
        - generate(pca, latents):          decode latents -> images in [0, 1]
        - interpolate(pca, a, b, n_steps):  walk between two latent codes
    Plot (used by src/pca/generation.ipynb):
        - plot_generated_grid(...):        a grid of freshly sampled images
        - plot_real_vs_generated(...):     pixel mean vs decoded latent mean
        - plot_interpolation(...):         a latent-space morph between two classes

This is deliberately a baseline: PCA + a Gaussian can only capture *linear*,
*unimodal* structure, so samples look like smooth "average" images rather than
the crisp, varied strokes a nonlinear model (VAE/GAN) would produce.
"""

from __future__ import annotations

import numpy as np

from utils.pca.dim_reduction import PCA, fit_pca

IMG_SIZE = 28
N_FEATURES = IMG_SIZE * IMG_SIZE  # 784


def fit_latent_gaussian(pca: PCA, X, y=None, ridge: float = 1e-6):
    """Fit a Gaussian to the PCA latent codes of `X`.

    The latent codes are `pca.transform(X)`. A small `ridge` is added to the
    covariance diagonal so it stays positive-definite (safe to sample from). If
    `y` is given, one Gaussian is fitted per class.

    Returns
    -------
    If `y is None`: a tuple ``(mean, cov)`` with shapes (d,) and (d, d).
    Otherwise: a dict ``{label: (mean, cov)}`` mapping each class label to its
    Gaussian parameters.
    """
    Z = pca.transform(X)
    d = Z.shape[1]

    def _gaussian(codes):
        mean = codes.mean(axis=0)
        cov = np.cov(codes, rowvar=False)
        cov = np.atleast_2d(cov) + ridge * np.eye(d)
        return mean, cov

    if y is None:
        return _gaussian(Z)

    y = np.asarray(y)
    return {int(label): _gaussian(Z[y == label]) for label in np.unique(y)}


def sample_latent(mean, cov, n: int, seed: int = 0):
    """Draw `n` latent vectors from the Gaussian ``N(mean, cov)`` -> (n, d)."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(np.asarray(mean), np.asarray(cov), size=n)


def generate(pca: PCA, latents, as_images: bool = False):
    """Decode latent vectors into pixel-space images, clipped to [0, 1].

    Parameters
    ----------
    as_images : bool
        If True, return (n, 28, 28); otherwise flat (n, 784).
    """
    imgs = pca.inverse_transform(np.asarray(latents))
    imgs = np.clip(imgs, 0.0, 1.0)
    if as_images:
        return imgs.reshape(-1, IMG_SIZE, IMG_SIZE)
    return imgs


def interpolate(pca: PCA, code_a, code_b, n_steps: int = 10, as_images: bool = False):
    """Linearly walk between two latent codes and decode each step.

    Pass the `pca.transform` codes of two real images to morph one into the
    other. Returns the decoded images (n_steps, ...) in [0, 1].
    """
    code_a = np.asarray(code_a)
    code_b = np.asarray(code_b)
    ts = np.linspace(0.0, 1.0, n_steps)[:, None]
    path = (1.0 - ts) * code_a[None, :] + ts * code_b[None, :]
    return generate(pca, path, as_images=as_images)


# --------------------------------------------------------------------------- #
# Plotting (src/pca/generation.ipynb)
# --------------------------------------------------------------------------- #
def plot_generated_grid(pca: PCA, per_class, class_names, n_per_class=8,
                        suptitle=None) -> None:
    """Sample each class's Gaussian, decode, and show a grid (one class per row)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(class_names), n_per_class,
                             figsize=(n_per_class * 1.1, len(class_names) * 1.1))
    for row, name in enumerate(class_names):
        mean, cov = per_class[row]
        imgs = generate(pca, sample_latent(mean, cov, n=n_per_class, seed=row), as_images=True)
        for col in range(n_per_class):
            axes[row, col].imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=25, va="center")
    fig.suptitle(suptitle or "Newly generated images (one Gaussian per class)")
    plt.tight_layout()
    plt.show()


def plot_real_vs_generated(pca: PCA, X, y, per_class, class_names) -> None:
    """Per class: the real pixel mean next to the decoded latent mean."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(class_names), 2, figsize=(3, len(class_names) * 1.4))
    for row, name in enumerate(class_names):
        real_mean = X[y == row].mean(axis=0)
        mean, _ = per_class[row]
        latent_mean_img = generate(pca, mean[None, :], as_images=True)[0]
        axes[row, 0].imshow(real_mean.reshape(IMG_SIZE, IMG_SIZE), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(latent_mean_img, cmap="gray", vmin=0, vmax=1)
        for col in range(2):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(name, rotation=0, labelpad=22, va="center")
    axes[0, 0].set_title("pixel mean")
    axes[0, 1].set_title("latent mean")
    plt.tight_layout()
    plt.show()


def plot_interpolation(pca: PCA, X, y, src_class, dst_class, class_names,
                       n_steps=10) -> None:
    """Interpolate in latent space between one real `src_class` and one `dst_class`."""
    import matplotlib.pyplot as plt

    idx_a = np.flatnonzero(y == src_class)[0]
    idx_b = np.flatnonzero(y == dst_class)[0]
    codes = pca.transform(X[[idx_a, idx_b]])
    walk = interpolate(pca, codes[0], codes[1], n_steps=n_steps, as_images=True)

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 1.0, 1.3))
    for col in range(n_steps):
        axes[col].imshow(walk[col], cmap="gray", vmin=0, vmax=1)
        axes[col].set_xticks([]); axes[col].set_yticks([])
    fig.suptitle(f"Latent-space interpolation:  real {class_names[src_class]}"
                 f"  ->  real {class_names[dst_class]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, y = load_digits(per_class=300)
    pca = fit_pca(X, n_components=50)
    print(f"PCA: {pca.n_components_} comps, "
          f"explained var {pca.explained_variance_ratio_.sum():.3f}")

    per_class = fit_latent_gaussian(pca, X, y=y)
    for label, (m, c) in per_class.items():
        imgs = generate(pca, sample_latent(m, c, n=3, seed=label), as_images=True)
        print(f"class {label}: generated {imgs.shape}")

    codes = pca.transform(X[:2])
    walk = interpolate(pca, codes[0], codes[1], n_steps=8, as_images=True)
    print(f"interpolation {walk.shape}")
