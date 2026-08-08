"""Métriques de reconstruction du codec K-means.

--- MSE et inertie sont la même quantité ---

Le codec reconstruit une image par le centroïde de son cluster :

    decode(encode(x)) = c[a(x)]        a(x) = cluster assigné à x

L'inertie somme les écarts au carré à ce même centroïde :

    inertie = Σ_i ||x_i - c[a(x_i)]||²

et la MSE en prend la moyenne, par image ET par pixel :

    MSE = (1 / (n·d)) · Σ_i Σ_p (x_ip - c[a(x_i)]_p)²
        = inertie / (n · d)

Les deux ne diffèrent donc que d'un facteur constant (n · d) : elles décroissent
exactement de la même façon, et tracer les deux courbes reviendrait à tracer deux
fois la même. La MSE reste utile pour sa lisibilité — elle s'exprime dans l'unité
des pixels (échelle 0-1) et ne dépend ni du nombre d'images ni de leur taille,
contrairement à l'inertie brute.
"""

import numpy as np
import tensorflow as tf

IMAGE_DIM = 784  # 28 * 28


def mean_squared_error(X, X_reconstructed):
    """MSE par pixel entre des images et leurs reconstructions.

    Args:
        X:               (n, d) images d'origine.
        X_reconstructed: (n, d) leurs reconstructions, même ordre.

    Returns:
        float — l'écart quadratique moyen par pixel. Sur des images normalisées
        dans [0, 1], sa racine se lit directement comme un écart de niveau de gris.
    """
    X = tf.cast(tf.convert_to_tensor(X), tf.float32)
    X_reconstructed = tf.cast(tf.convert_to_tensor(X_reconstructed), tf.float32)

    if X.shape != X_reconstructed.shape:
        raise ValueError(
            f"Formes différentes : X {tuple(X.shape)} vs reconstructions "
            f"{tuple(X_reconstructed.shape)}."
        )

    return float(tf.reduce_mean(tf.square(X - X_reconstructed)))


def mse_from_inertia(inertia, n_samples, image_dim=IMAGE_DIM):
    """Convertit une inertie en MSE — voir l'entête du module pour la démonstration.

    Évite de reconstruire les images pour rien : fit_kmeans calcule déjà l'inertie
    à chaque itération, donc la courbe de MSE s'en déduit par une division.
    """
    if n_samples <= 0:
        raise ValueError("n_samples doit être strictement positif.")
    return float(inertia) / (int(n_samples) * int(image_dim))


def reconstruct(X, centroids, cluster_labels):
    """Reconstruction de chaque image par le centroïde de son cluster."""
    centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)
    cluster_labels = tf.cast(tf.convert_to_tensor(cluster_labels), tf.int32)
    return tf.gather(centroids, cluster_labels)


def compute_compression_ratio(k, image_dim=IMAGE_DIM, bits_per_pixel=8):
    """Taux de compression du codec : pixels bruts / bits du code.

    Une image coûte image_dim · bits_per_pixel bits ; son code n'en coûte que
    ceil(log2(k)) — l'indice du centroïde. Le dictionnaire (les k centroïdes)
    n'est pas compté : il est transmis une fois, pas à chaque image.
    """
    k = int(k)
    if k < 1:
        raise ValueError("k doit valoir au moins 1.")

    code_bits = int(np.ceil(np.log2(k))) if k > 1 else 0
    raw_bits = int(image_dim) * int(bits_per_pixel)
    # K=1 : le code ne porte aucune information, la compression est « infinie ».
    return float("inf") if code_bits == 0 else raw_bits / code_bits
