"""Codec de Kohonen — identique à celui du K-means.

Encoder une image = transmettre l'indice de son neurone gagnant ; la décoder =
lire le feature vector de ce neurone. Le SOM organise ses prototypes sur une
grille, mais cela ne change rien à la compression : le code reste un entier.
"""

import numpy as np

from .kohonen import compute_squared_distances


def encode(X, weights):
    """Indice du neurone gagnant — le code transmis. Retourne un int."""
    X = np.asarray(X, dtype=np.float32)

    # Si X représente une seule image : (784,) -> (1, 784)
    if X.ndim == 1:
        X = X[None, :]

    distances = compute_squared_distances(X, weights)   # (1, k)
    return int(np.argmin(distances, axis=1)[0])         # ex. 4


def decode(code, weights):
    """Feature vector du neurone `code`, shape (784,) — l'image reconstruite."""
    weights = np.asarray(weights, dtype=np.float32)
    return weights[int(code)]
