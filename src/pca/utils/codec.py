"""Encode / décode une image via la PCA du projet (utils/pca/dim_reduction.py).

--- Le code PCA n'est pas un entier ---

K-means et SOM transmettent UN indice ; la PCA transmet les k coordonnées de
l'image sur les composantes principales. Or l'app suppose des codes hashables
(elle compte les codes distincts avec set()) et affichables (elle titre chaque
reconstruction « code {c} »). PCACode enveloppe donc le vecteur : hashable par
ses octets, affiché par ses deux premières coordonnées.

Aucun calcul PCA ici : transform / inverse_transform viennent de l'objet PCA
d'utils/pca, reconstruit depuis les poids de l'app par pca_from_weights.
"""

import numpy as np

from utils.pca.compression import reconstruct
from utils.pca.dim_reduction import PCA


def pca_from_weights(weights):
    """Reconstruit l'objet PCA depuis les poids de l'app : (k+1, 784).

    Ligne 0 : l'image moyenne ; lignes 1..k : les composantes principales.
    Les variances sont factices (transform / inverse_transform n'en ont pas
    besoin) : les vues qui en dépendent reconstruisent leur PCA depuis les
    métadonnées, où le spectre exact est stocké.
    """
    weights = np.asarray(weights, dtype=np.float64)
    k = len(weights) - 1
    dummy = np.ones(k)
    return PCA(
        mean_=weights[0],
        components_=weights[1:],
        explained_variance_=dummy,
        explained_variance_ratio_=dummy / max(k, 1),
    )


class PCACode:
    """Le code PCA d'une image : k flottants, hashable et affichable pour l'app."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.values.setflags(write=False)  # un code est une valeur, pas un buffer

    def __eq__(self, other):
        return isinstance(other, PCACode) and np.array_equal(self.values, other.values)

    def __hash__(self):
        return hash(self.values.tobytes())

    def __str__(self):
        # Les 2 premières coordonnées suffisent à titrer une vignette ; les k
        # afficher rendrait les libellés illisibles dès k=5.
        head = ", ".join(f"{v:+.1f}" for v in self.values[:2])
        tail = ", …" if len(self.values) > 2 else ""
        return f"z=({head}{tail})"

    __repr__ = __str__


def encode(image, weights):
    """Code d'une image : ses coordonnées sur les composantes du modèle."""
    codes = pca_from_weights(weights).transform(np.asarray(image, dtype=np.float64))
    return PCACode(codes)


def decode(code, weights):
    """Reconstruit l'image d'un code : moyenne + Σ zᵢ·PCᵢ, clippée dans [0, 1]."""
    values = code.values if isinstance(code, PCACode) else np.asarray(code)
    image = reconstruct(pca_from_weights(weights), np.asarray(values, dtype=np.float64))
    return np.asarray(image, dtype=np.float32)
