"""PCA — the project's single source of truth for Principal Component Analysis.

Principal Component Analysis rotates the 784-dimensional pixel space into an
orthogonal basis ordered by how much variance each axis explains. This module
implements PCA **from scratch** -- mean-centering, a covariance matrix, and an
eigendecomposition -- rather than calling `sklearn.decomposition.PCA`, so every
step of the algorithm stays visible.

It is the canonical PCA for the whole study: `compression.py` and
`generation.py` import `fit_pca` / `PCA` from here rather than re-implementing
it, so there is exactly one PCA in the codebase.

    Fit / analyse:
        - fit_pca(X, n_components):            fit a PCA model on a design matrix
        - cumulative_explained_variance(pca):  running sum of explained-variance ratio
        - n_components_for_variance(X, thr):   #components needed to reach a variance
        - project(pca, X) / project_2d(X):     map images onto principal components
    Plot (used by src/pca/dim_reduction.ipynb):
        - plot_spectrum(pca, ...):             scree + cumulative curves
        - plot_projection(coords, y, ...):     2D + 3D scatter, coloured by class

The `PCA` object mirrors the handful of `sklearn.decomposition.PCA` attributes
the project relies on (`n_components_`, `explained_variance_ratio_`,
`.transform`, `.inverse_transform`). All functions follow the shared data
contract: X is an (n_samples, 784) float32 matrix in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default per-class colours, reused across every scatter in the project.
CLASS_COLORS = ["tab:blue", "tab:orange", "tab:green"]


@dataclass
class PCA:
    """A fitted PCA model: the mean, principal axes, and their variances.

    Exposes the handful of `sklearn.decomposition.PCA`-shaped attributes the
    rest of this project relies on (`n_components_`, `explained_variance_ratio_`,
    `.transform`, `.inverse_transform`), even though the fit never touches
    sklearn.
    """

    mean_: np.ndarray                      # (n_features,) subtracted before projecting
    components_: np.ndarray                # (n_components, n_features) axes, descending variance
    explained_variance_: np.ndarray        # (n_components,) eigenvalue (variance) of each axis
    explained_variance_ratio_: np.ndarray  # (n_components,) fraction of total variance per axis

    @property
    def n_components_(self) -> int:
        return self.components_.shape[0]

    def transform(self, X) -> np.ndarray:
        """Project `X` (n, n_features) onto the fitted axes -> codes (n, n_components)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim not in (1, 2) or X.shape[-1] != self.mean_.size:
            raise ValueError(
                f"X doit finir par {self.mean_.size} caractéristiques, forme reçue : {X.shape}."
            )
        if X.size == 0 or not np.isfinite(X).all():
            raise ValueError("X doit être non vide et ne contenir que des valeurs finies.")
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, codes) -> np.ndarray:
        """Map codes (n, n_components) back to feature space -> (n, n_features).

        The inverse of `transform`: rebuild each image as the class mean plus a
        linear combination of the kept principal axes. With fewer components
        than features this is lossy (that is exactly what `compression.py` and
        `generation.py` exploit).
        """
        codes = np.asarray(codes, dtype=np.float64)
        if codes.ndim not in (1, 2) or codes.shape[-1] != self.n_components_:
            raise ValueError(
                f"codes doit finir par {self.n_components_} valeurs, forme reçue : {codes.shape}."
            )
        if codes.size == 0 or not np.isfinite(codes).all():
            raise ValueError("codes doit être non vide et ne contenir que des valeurs finies.")
        return codes @ self.components_ + self.mean_


def fit_pca(X, n_components=None, center="feature") -> PCA:
    """Fit a PCA model on the design matrix `X` by decomposing it step by step.

    1. Center `X` on a mean (see `center`) -- PCA looks for directions of
       variance *around* the point it is centered on.
    2. Build the covariance matrix of the centered features.
    3. Eigendecompose that (symmetric) covariance matrix: eigenvectors are the
       principal axes, eigenvalues are the variance each axis explains.
    4. Sort the eigenpairs by descending eigenvalue.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Flat, scaled images (the shared data contract).
    n_components : int, float or None
        None keeps every axis (== n_features here, since we have far more
        samples than pixels); an int keeps that many leading axes; a float in
        (0, 1] keeps just enough leading axes to explain that fraction of
        variance.
    center : {"feature", "global"}
        Which mean to subtract in step 1.
        - "feature" (default, standard PCA): the per-feature (per-pixel) mean
          `X.mean(axis=0)`. Removes each pixel's baseline, so the axes describe
          variation *around the mean image*.
        - "global": a single scalar mean `X.mean()` subtracted from every pixel.
          This only shifts the data by a constant, so the mean image is *not*
          removed; PC1 then ends up dominated by that mean image. Included to
          make the effect of the centering choice visible.

    Returns
    -------
    PCA
        The fitted model (call `.transform` / `.inverse_transform` to move
        between pixel and code space).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X doit être une matrice 2D, forme reçue : {X.shape}.")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("La PCA exige au moins 2 observations et 1 caractéristique.")
    if not np.isfinite(X).all():
        raise ValueError("X contient des valeurs NaN ou infinies.")

    # Step 1: center. `mean_` is kept as a length-(n_features) vector in both
    # modes so `.transform` / `.inverse_transform` behave identically.
    if center == "feature":
        mean_ = X.mean(axis=0)
    elif center == "global":
        mean_ = np.full(X.shape[1], X.mean())
    else:
        raise ValueError(f"center must be 'feature' or 'global', got {center!r}")
    Xc = X - mean_

    # Step 2: covariance matrix of the centered features, shape (n_features, n_features).
    cov = (Xc.T @ Xc) / (X.shape[0] - 1)

    # Step 3: eigendecomposition. `cov` is symmetric, so `eigh` is exact and
    # faster than a general eigensolver; it returns eigenvalues ascending.
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Step 4: sort descending -- largest-variance axis first. Clip float noise
    # that can push a near-zero eigenvalue slightly below 0.
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    total_variance = float(eigvals.sum())
    ratio = (
        eigvals / total_variance
        if total_variance > np.finfo(np.float64).eps
        else np.zeros_like(eigvals)
    )

    # Step 5: decide how many leading axes to keep.
    if n_components is None:
        k = len(eigvals)
    elif isinstance(n_components, float):
        if not 0.0 < n_components <= 1.0:
            raise ValueError(f"n_components float must be in (0, 1], got {n_components!r}")
        k = (
            int(np.searchsorted(np.cumsum(ratio), n_components) + 1)
            if total_variance > np.finfo(np.float64).eps
            else 1
        )
    else:
        k = int(n_components)

    if not 1 <= k <= X.shape[1]:
        raise ValueError(
            f"n_components doit être compris entre 1 et {X.shape[1]}, reçu : {k}."
        )

    return PCA(
        mean_=mean_,
        components_=eigvecs[:, :k].T,
        explained_variance_=eigvals[:k],
        explained_variance_ratio_=ratio[:k],
    )


def cumulative_explained_variance(pca: PCA) -> np.ndarray:
    """Running total of the explained-variance ratio over the components.

    Returns
    -------
    np.ndarray, shape (n_components,)
        `cumsum` of `pca.explained_variance_ratio_`; entry i is the fraction of
        total variance captured by the first i + 1 principal components.
    """
    return np.cumsum(pca.explained_variance_ratio_)


def components_for_variance(pca: PCA, threshold: float) -> int:
    """#leading components of a *fitted* pca whose cumulative variance >= threshold."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
    cum = cumulative_explained_variance(pca)
    if cum.size == 0 or cum[-1] <= np.finfo(np.float64).eps:
        return 0
    # +1 converts a 0-based index into a count of components.
    return int(np.searchsorted(cum, threshold) + 1)


def n_components_for_variance(X, threshold: float = 0.90) -> int:
    """Smallest number of principal components (fit on `X`) reaching a variance.

    Convenience wrapper that fits a full PCA on `X` and calls
    `components_for_variance`.
    """
    return components_for_variance(fit_pca(X), threshold)


def project(pca: PCA, X) -> np.ndarray:
    """Project `X` onto the principal components of a fitted `pca`.

    Returns
    -------
    np.ndarray, shape (n_samples, pca.n_components_)
        The coordinates of each row of `X` in the PCA basis.
    """
    return pca.transform(X)


def project_2d(X) -> np.ndarray:
    """Convenience: fit PCA and return the first 2 principal components of `X`."""
    return project(fit_pca(X, n_components=2), X)


# --------------------------------------------------------------------------- #
# Plotting (src/pca/dim_reduction.ipynb)
# --------------------------------------------------------------------------- #
def plot_spectrum(pca: PCA, thresholds=(0.90, 0.95), n_show=50,
                  color="steelblue", suptitle=None) -> dict:
    """Scree plot + cumulative explained-variance curve for a fitted `pca`.

    Marks each variance threshold and the number of components that reaches it.

    Returns
    -------
    dict {threshold: n_components} for the printed/annotated thresholds.
    """
    import matplotlib.pyplot as plt

    cum = cumulative_explained_variance(pca)
    ks = {thr: components_for_variance(pca, thr) for thr in thresholds}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show], color=color)
    ax1.set_title(f"Scree plot (first {n_show} components)")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance ratio")

    ax2.plot(range(1, len(cum) + 1), cum, color=color)
    for (thr, k), c in zip(ks.items(), ["crimson", "darkorange", "purple"]):
        ax2.axhline(thr, color=c, ls="--", lw=1)
        ax2.axvline(k, color=c, ls=":", lw=1)
        ax2.scatter([k], [thr], color=c, zorder=5, label=f"{thr:.0%} -> {k} comps")
    ax2.set_title("Cumulative explained variance")
    ax2.set_xlabel("number of components")
    ax2.set_ylabel("cumulative explained variance")
    ax2.legend(loc="lower right")

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()
    return ks


def plot_spectrum_comparison(pcas, labels, colors=None, n_show=15, suptitle=None):
    """Overlay the variance spectra of several fitted PCAs on one figure.

    Used to compare two centering choices (see `fit_pca(center=...)`) on the same
    data: left panel overlays their cumulative explained-variance curves, right
    panel their first `n_show` per-component ratios.

    Returns
    -------
    list of (label, pc1_ratio, k90) -- PC1's variance share and the #components
    reaching 90% variance, for each pca.
    """
    import matplotlib.pyplot as plt

    colors = colors or ["steelblue", "crimson", "seagreen"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    summary = []
    for pca, label, c in zip(pcas, labels, colors):
        cum = cumulative_explained_variance(pca)
        k90 = components_for_variance(pca, 0.90)
        summary.append((label, float(pca.explained_variance_ratio_[0]), k90))
        ax1.plot(range(1, len(cum) + 1), cum, color=c, label=f"{label} (90% at {k90} comps)")
        ax2.plot(range(1, n_show + 1), pca.explained_variance_ratio_[:n_show],
                 "o-", color=c, label=label)
    ax1.axhline(0.90, color="gray", ls="--", lw=1)
    ax1.set_title("Cumulative explained variance")
    ax1.set_xlabel("number of components")
    ax1.set_ylabel("cumulative explained variance")
    ax1.legend(loc="lower right")
    ax2.set_title(f"Scree (first {n_show} components)")
    ax2.set_xlabel("principal component")
    ax2.set_ylabel("explained variance ratio")
    ax2.legend()

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    plt.show()
    return summary


def plot_projection(coords, y, class_names, colors=None, suptitle=None) -> None:
    """Side-by-side 2D (PC1/PC2) and 3D (PC1/PC2/PC3) scatter, coloured by class.

    Parameters
    ----------
    coords : np.ndarray, shape (n, >=3)
        PCA coordinates (`project(pca, X)`).
    y : np.ndarray, shape (n,)
        Class index per row (0 .. len(class_names) - 1).
    class_names : list of str
        Display label for each class index.
    colors : list of str or None
        One colour per class (defaults to CLASS_COLORS).
    """
    import matplotlib.pyplot as plt

    colors = colors or CLASS_COLORS
    fig, ax2d = plt.subplots(figsize=(7, 6))
    for i, (name, c) in enumerate(zip(class_names, colors)):
        m = y == i
        ax2d.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.4, color=c, label=name)
    ax2d.set_xlabel("PC1"); ax2d.set_ylabel("PC2")
    ax2d.set_title(suptitle or "First 2 principal components")
    ax2d.legend(markerscale=3)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8, 6.5))
    ax3d = fig.add_subplot(111, projection="3d")
    for i, (name, c) in enumerate(zip(class_names, colors)):
        m = y == i
        ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=6, alpha=0.4, color=c, label=name)
    ax3d.set_xlabel("PC1"); ax3d.set_ylabel("PC2"); ax3d.set_zlabel("PC3")
    ax3d.set_title("First 3 principal components")
    ax3d.legend(markerscale=3)
    plt.tight_layout()
    plt.show()


def _image_side(n_features: int) -> int:
    """Side length of the square image a flat feature vector reshapes to."""
    return int(round(np.sqrt(n_features)))


def plot_eigenimages(pca: PCA, n=8, suptitle=None, cmap="gray") -> None:
    """Render the mean image and the top-`n` principal components as images.

    Each principal axis is a length-(n_features) vector; reshaping it back to a
    square shows the pixel pattern that component encodes (the classic
    "eigen-digit" / "eigen-sketch" view). The leading column is the mean image
    `pca.mean_`.
    """
    import matplotlib.pyplot as plt

    side = _image_side(pca.components_.shape[1])
    fig, axes = plt.subplots(1, n + 1, figsize=(1.4 * (n + 1), 1.7))
    axes[0].imshow(pca.mean_.reshape(side, side), cmap=cmap)
    axes[0].set_title("mean", fontsize=9)
    axes[0].axis("off")
    for i in range(n):
        axes[i + 1].imshow(pca.components_[i].reshape(side, side), cmap=cmap)
        axes[i + 1].set_title(f"PC{i + 1}", fontsize=9)
        axes[i + 1].axis("off")
    if suptitle:
        fig.suptitle(suptitle, y=1.18)
    plt.tight_layout()
    plt.show()


def plot_pc1_vs_mean(X, suptitle=None) -> None:
    """Show why global centering "wastes" PC1: PC1(global) looks like the mean image.

    Fits the same `X` with per-feature and global centering and renders, side by
    side, the mean image and each fit's PC1. Under global centering the mean
    image is left in the data, so its PC1 re-encodes that mean image; under
    per-feature centering PC1 is free to describe a genuine contrast pattern.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    side = _image_side(X.shape[1])
    pf = fit_pca(X, center="feature")
    pg = fit_pca(X, center="global")

    # Eigenvector signs are arbitrary; flip global PC1 to correlate positively
    # with the mean image so the visual resemblance is obvious rather than inverted.
    mean_centered = pf.mean_ - pf.mean_.mean()
    pc1_global = pg.components_[0]
    if pc1_global @ mean_centered < 0:
        pc1_global = -pc1_global

    panels = [
        ("mean image\nX.mean(axis=0)", pf.mean_),
        ("PC1 — per-feature\ncentering", pf.components_[0]),
        ("PC1 — global\ncentering", pc1_global),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.6))
    for ax, (title, vec) in zip(axes, panels):
        ax.imshow(vec.reshape(side, side), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle or "PC1 under each centering vs. the mean image", y=1.02)
    plt.tight_layout()
    plt.show()


def plot_component_heatmaps(pca: PCA, n=8, suptitle=None) -> None:
    """Render the top-`n` components as diverging +/- heatmaps.

    A component pushes each pixel up or down as you move along its axis, so a
    diverging colormap makes that structure readable: **red adds ink, blue
    removes it** (each panel is scaled symmetrically about 0).
    """
    import matplotlib.pyplot as plt

    side = _image_side(pca.components_.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(1.5 * n, 1.9))
    for i in range(n):
        comp = pca.components_[i].reshape(side, side)
        vmax = float(np.abs(comp).max())
        axes[i].imshow(comp, cmap="seismic", vmin=-vmax, vmax=vmax)
        axes[i].set_title(f"PC{i + 1}", fontsize=9)
        axes[i].axis("off")
    base = suptitle or "Principal components as +/- heatmaps"
    fig.suptitle(f"{base}   (red = adds ink, blue = removes)", y=1.12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from utils.mnist_data import load_digits

    X, y = load_digits(per_class=1000)
    print(f"X {X.shape} {X.dtype} range [{X.min():.2f}, {X.max():.2f}]")

    pca = fit_pca(X)
    print(f"fitted PCA with {pca.n_components_} components")
    print(f"first 5 explained-variance ratios: {pca.explained_variance_ratio_[:5]}")

    for thr in (0.90, 0.95):
        print(f"components for {thr:.0%} variance: {components_for_variance(pca, thr)}")

    # transform / inverse_transform round-trip sanity check.
    codes = project(pca, X)
    X_hat = pca.inverse_transform(codes)
    print(f"full round-trip max abs error: {np.abs(X - X_hat).max():.2e}")
