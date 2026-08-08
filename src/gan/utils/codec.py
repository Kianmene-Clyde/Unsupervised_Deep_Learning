"""Encode / décode une image via le GAN (utils/model.py).

--- Un GAN n'a pas d'encodeur : le code est une INVERSION ---

K-means transmet un indice, l'autoencodeur un vecteur latent sorti d'un
encodeur appris. Le GAN, lui, ne sait que GÉNÉRER : rien ne va de l'image
vers z. Le « code » d'une image est donc le z de l'image générée la plus
proche dans une banque de candidates (voir model.encode_batch) — une
inversion approximative, assumée comme telle : c'est elle qui rend visible,
dans l'onglet compression, que le GAN est un mauvais codec.

On réutilise l'enveloppe AECode de l'autoencodeur (même besoin : un vecteur
de flottants hashable et affichable), comme le VAE le fait déjà.
"""

import numpy as np

from autoencoder.utils.codec import AECode  # même enveloppe : latent_dim flottants
from .model import decode_batch, encode_batch


def encode(image, weights):
    """Code d'une image : le z de sa plus proche voisine générée (inversion)."""
    return AECode(encode_batch(np.asarray(image), weights)[0])


def decode(code, weights):
    """« Reconstruit » l'image d'un code : G(z), clippée dans [0, 1]."""
    values = code.values if isinstance(code, AECode) else np.asarray(code)
    return decode_batch(values, weights)[0]
