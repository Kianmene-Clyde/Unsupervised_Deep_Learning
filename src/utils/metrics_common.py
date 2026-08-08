"""Robust metrics shared by all algorithms in the comparison report."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

IMG_SIZE = 28
IMAGE_DIM = IMG_SIZE * IMG_SIZE


def _matrix(values, name, *, allow_single=True):
    array = np.asarray(values, dtype=np.float64)
    if allow_single and array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} doit être une matrice 2D non vide, forme reçue : {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contient des valeurs NaN ou infinies.")
    return array


def _paired_matrices(left, right, left_name="X", right_name="X_hat"):
    left = _matrix(left, left_name)
    right = _matrix(right, right_name)
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} et {right_name} doivent avoir la même forme, "
            f"reçu : {left.shape} et {right.shape}."
        )
    return left, right


# ---------------------------------------------------- Tâche 2 : compression


def mse(X, X_hat):
    """Mean squared error per feature."""
    X, X_hat = _paired_matrices(X, X_hat)
    return float(np.mean(np.square(X - X_hat), dtype=np.float64))


def psnr(X, X_hat, data_range=1.0):
    """Peak Signal-to-Noise Ratio in dB (``+inf`` for an exact match)."""
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range doit être un nombre fini strictement positif.")
    error = mse(X, X_hat)
    if error == 0:
        return float("inf")
    return float(10.0 * np.log10(float(data_range) ** 2 / error))


def ssim(X, X_hat, data_range=1.0, window=7):
    """Mean structural similarity for batches of flattened 28×28 images."""
    X, X_hat = _paired_matrices(X, X_hat)
    if X.shape[1] != IMAGE_DIM:
        raise ValueError(f"SSIM attend {IMAGE_DIM} pixels par image, reçu : {X.shape[1]}.")
    if not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range doit être un nombre fini strictement positif.")
    if not isinstance(window, (int, np.integer)) or not 1 <= int(window) <= IMG_SIZE:
        raise ValueError(f"window doit être un entier entre 1 et {IMG_SIZE}.")

    X = X.reshape(-1, IMG_SIZE, IMG_SIZE)
    X_hat = X_hat.reshape(-1, IMG_SIZE, IMG_SIZE)
    size = (1, int(window), int(window))  # never blur across the batch axis
    mean_x = ndimage.uniform_filter(X, size=size)
    mean_y = ndimage.uniform_filter(X_hat, size=size)
    var_x = ndimage.uniform_filter(X * X, size=size) - mean_x * mean_x
    var_y = ndimage.uniform_filter(X_hat * X_hat, size=size) - mean_y * mean_y
    covariance = ndimage.uniform_filter(X * X_hat, size=size) - mean_x * mean_y
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (
        var_x + var_y + c2
    )
    return float(np.mean(np.divide(numerator, denominator)))


def compression_ratio(code_bits):
    """Compression ratio relative to a raw 8-bit 28×28 image."""
    if not np.isfinite(code_bits) or code_bits < 0:
        raise ValueError("code_bits doit être un nombre fini positif ou nul.")
    if code_bits == 0:
        return float("inf")
    return float(IMAGE_DIM * 8 / code_bits)


# ---------------------------------------------------- Tâche 1 : clustering


def _labels(values, name):
    raw = np.asarray(values)
    if raw.size == 0:
        raise ValueError(f"{name} ne peut pas être vide.")
    try:
        numeric = raw.astype(np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} doit contenir des entiers.") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} doit contenir uniquement des entiers finis.")
    return numeric.astype(np.int64)


def purity(labels, y_true):
    """Fraction of samples belonging to the majority true class of their cluster."""
    labels = _labels(labels, "labels")
    y_true = _labels(y_true, "y_true")
    if labels.size != y_true.size:
        raise ValueError("labels et y_true doivent avoir la même longueur.")
    total = 0
    for cluster in np.unique(labels):
        _, counts = np.unique(y_true[labels == cluster], return_counts=True)
        total += int(counts.max())
    return float(total / y_true.size)


def silhouette(X, labels, sample_size=2000, seed=42):
    """Mean Euclidean silhouette on a deterministic optional subsample."""
    X = _matrix(X, "X")
    labels = _labels(labels, "labels")
    if len(X) != len(labels):
        raise ValueError("X et labels doivent avoir la même longueur.")
    if not isinstance(sample_size, (int, np.integer)) or sample_size < 2:
        raise ValueError("sample_size doit être un entier supérieur ou égal à 2.")

    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=int(sample_size), replace=False)
        X, labels = X[idx], labels[idx]

    unique = np.unique(labels)
    if unique.size < 2 or unique.size >= len(labels):
        return float("nan")

    squared_norms = np.sum(X * X, axis=1)
    distances = np.sqrt(np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * X @ X.T, 0.0
    ))

    scores = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        same = labels == labels[i]
        same_count = int(same.sum())
        if same_count == 1:  # sklearn convention for singleton clusters
            continue
        intra = distances[i, same].sum() / (same_count - 1)
        nearest = min(
            distances[i, labels == cluster].mean()
            for cluster in unique if cluster != labels[i]
        )
        scale = max(intra, nearest)
        scores[i] = (nearest - intra) / scale if scale > 0 else 0.0
    return float(scores.mean())


# ---------------------------------------------------- Tâche 3 : génération


def fit_pca_features(X_real, n_components=50, seed=42):
    """Fit the deterministic PCA feature space used by the simplified FID."""
    del seed  # Kept for API compatibility; symmetric eigendecomposition is deterministic.
    X_real = _matrix(X_real, "X_real")
    max_components = X_real.shape[1]
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components <= max_components:
        raise ValueError(f"n_components doit être compris entre 1 et {max_components}.")
    if len(X_real) < 2:
        raise ValueError("Il faut au moins deux images réelles pour ajuster les features.")

    mean = X_real.mean(axis=0)
    centered = X_real - mean
    covariance = (centered.T @ centered) / (len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:int(n_components)]
    return mean, eigenvectors[:, order].T


def pca_features(X, featurizer):
    try:
        mean, components = featurizer
    except (TypeError, ValueError) as exc:
        raise ValueError("featurizer doit contenir (moyenne, composantes).") from exc
    mean = np.asarray(mean, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    if mean.ndim != 1 or components.ndim != 2 or components.shape[1] != mean.size:
        raise ValueError("Les dimensions du featureur PCA sont incohérentes.")
    if not np.isfinite(mean).all() or not np.isfinite(components).all():
        raise ValueError("Le featureur PCA contient des valeurs non finies.")
    X = _matrix(X, "X")
    if X.shape[1] != mean.size:
        raise ValueError(f"X doit contenir {mean.size} caractéristiques par ligne.")
    return (X - mean) @ components.T


def frechet_distance(F_real, F_gen):
    """Stable Fréchet distance between two Gaussian feature distributions."""
    F_real = _matrix(F_real, "F_real")
    F_gen = _matrix(F_gen, "F_gen")
    if F_real.shape[1] != F_gen.shape[1]:
        raise ValueError("F_real et F_gen doivent avoir la même dimension de features.")
    if len(F_real) < 2 or len(F_gen) < 2:
        raise ValueError("La distance de Fréchet exige au moins deux points par lot.")

    mean_delta = F_real.mean(axis=0) - F_gen.mean(axis=0)
    covariance_real = np.atleast_2d(np.cov(F_real, rowvar=False))
    covariance_gen = np.atleast_2d(np.cov(F_gen, rowvar=False))

    # Tr((C1 C2)^1/2) is evaluated through the symmetric PSD matrix
    # C1^(1/2) C2 C1^(1/2), avoiding complex values from scipy.sqrtm(C1 @ C2).
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_real)
    sqrt_real = (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))) @ eigenvectors.T
    middle = sqrt_real @ covariance_gen @ sqrt_real
    middle = (middle + middle.T) * 0.5
    trace_covmean = float(np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0.0, None)).sum())

    distance = (
        float(mean_delta @ mean_delta)
        + float(np.trace(covariance_real))
        + float(np.trace(covariance_gen))
        - 2.0 * trace_covmean
    )
    # A tiny negative value is possible for two identical, rank-deficient lots.
    return float(max(distance, 0.0))


def frechet_from_images(X_real, X_gen, n_components=50, seed=42):
    """Compute the simplified FID directly from two image batches."""
    feature_space = fit_pca_features(X_real, n_components=n_components, seed=seed)
    return frechet_distance(
        pca_features(X_real, feature_space), pca_features(X_gen, feature_space)
    )
