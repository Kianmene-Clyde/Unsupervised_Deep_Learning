"""Encode / décode une image via l'autoencodeur (utils/model.py).

--- Le code d'un autoencodeur n'est pas un entier ---

K-means et SOM transmettent UN indice ; l'autoencodeur transmet le vecteur
latent z = encoder(image), latent_dim flottants. Or l'app suppose des codes
hashables (elle compte les codes distincts avec set()) et affichables (elle
titre chaque reconstruction « code {c} »). AECode enveloppe donc le vecteur,
exactement comme PCACode côté PCA : hashable par ses octets, affiché par ses
deux premières coordonnées.

Aucun calcul ici : encoder/décoder viennent des modèles Keras que
utils/model.py reconstruit (et met en cache) depuis les poids de l'app.
"""

import numpy as np

from .model import decode_batch, encode_batch


class AECode:
    """Le code latent d'une image : latent_dim flottants, hashable et affichable."""

    __slots__ = ("values",)

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.values.setflags(write=False)  # un code est une valeur, pas un buffer

    def __eq__(self, other):
        return isinstance(other, AECode) and np.array_equal(self.values, other.values)

    def __hash__(self):
        return hash(self.values.tobytes())

    def __str__(self):
        # Les 2 premières coordonnées suffisent à titrer une vignette ; en
        # afficher plus rendrait les libellés illisibles dès latent_dim=5.
        head = ", ".join(f"{v:+.1f}" for v in self.values[:2])
        tail = ", …" if len(self.values) > 2 else ""
        return f"z=({head}{tail})"

    __repr__ = __str__


def encode(image, weights):
    """Code d'une image : son vecteur latent, sorti de l'encodeur."""
    return AECode(encode_batch(np.asarray(image), weights)[0])


def decode(code, weights):
    """Reconstruit l'image d'un code : décodeur(z), clippée dans [0, 1]."""
    values = code.values if isinstance(code, AECode) else np.asarray(code)
    return decode_batch(values, weights)[0]
