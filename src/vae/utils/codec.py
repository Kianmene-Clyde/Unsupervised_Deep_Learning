"""Encode / décode une image via le VAE (utils/model.py).

Le code d'une image est z_mean = encoder(image) — latent_dim flottants, le
point latent déterministe (voir model.encode_batch). C'est le même contrat que
l'autoencodeur simple, on réutilise donc son enveloppe AECode : hashable par ses
octets (l'app compte les codes distincts avec set()), affichable par ses deux
premières coordonnées (elle titre chaque reconstruction « z=(…) »).

Aucun calcul ici : encoder/décoder viennent des modèles Keras que utils/model.py
reconstruit (et met en cache) depuis les poids de l'app.
"""

import numpy as np

from autoencoder.utils.codec import AECode  # même enveloppe : z_mean = latent_dim flottants
from .model import decode_batch, encode_batch


def encode(image, weights):
    """Code d'une image : son z_mean, sorti de l'encodeur."""
    return AECode(encode_batch(np.asarray(image), weights)[0])


def decode(code, weights):
    """Reconstruit l'image d'un code : décodeur(z), clippée dans [0, 1]."""
    values = code.values if isinstance(code, AECode) else np.asarray(code)
    return decode_batch(values, weights)[0]
