"""Génération de données synthétiques avec un K-means — la 3e tâche du projet.

--- Un K-means ne sait pas générer. Voici le mieux qu'il puisse faire ---

Le « modèle » appris est un dictionnaire de k centroïdes : decode() ne rendra
jamais que k images, toujours les mêmes. Pour produire des images NOUVELLES,
on ajuste sur chaque cluster la distribution la plus simple possible — une
gaussienne diagonale (centroïde = moyenne, écart-type par pixel mesuré sur les
membres du cluster) — et on en tire :

    1. cluster c ~ proportions des clusters (les gros clusters génèrent plus) ;
    2. image = centroïde_c + jitter · sigma_c · epsilon,  epsilon ~ N(0, I).

C'est un vrai générateur probabiliste (un mélange de gaussiennes dont K-means
a placé les moyennes), mais l'indépendance pixel à pixel se paie cash : le
bruit s'ajoute SANS respecter les traits, les images générées sont des
centroïdes neigeux plutôt que de nouveaux dessins. C'est le résultat attendu —
la démonstration, chiffrable et visible, que la tâche de génération demande un
modèle de la STRUCTURE des images (VAE, GAN), pas un dictionnaire de moyennes.
NumPy pur : rien ici ne touche TensorFlow, on part des centroïdes déjà appris.
"""

import numpy as np

IMAGE_DIM = 784


def cluster_stats(X, labels, k):
    """Poids (proportions) et écart-type par pixel de chaque cluster.

    Un cluster vide (jamais assigné) garde un poids nul et un sigma nul : il ne
    sera jamais tiré, et s'il l'était son centroïde sortirait tel quel.
    """
    X = np.asarray(X, dtype=np.float32).reshape(len(X), IMAGE_DIM)
    labels = np.asarray(labels).astype(int).ravel()

    weights = np.zeros(k, dtype=np.float64)
    sigmas = np.zeros((k, IMAGE_DIM), dtype=np.float32)
    for c in range(k):
        members = X[labels == c]
        weights[c] = len(members)
        if len(members) > 1:
            sigmas[c] = members.std(axis=0)
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights, sigmas


def sample_synthetic(centroids, weights, sigmas, n_images, seed=42, jitter=1.0):
    """Tire n_images du mélange de gaussiennes. → (images (n, 784), clusters (n,)).

    jitter dose le bruit : 0 rend les centroïdes purs (la « génération » dégénère
    en dictionnaire), 1 respecte la dispersion mesurée de chaque cluster.
    """
    centroids = np.asarray(centroids, dtype=np.float32).reshape(-1, IMAGE_DIM)
    rng = np.random.default_rng(seed)

    chosen = rng.choice(len(centroids), size=n_images, p=weights)
    eps = rng.standard_normal((n_images, IMAGE_DIM)).astype(np.float32)
    images = centroids[chosen] + jitter * sigmas[chosen] * eps
    return np.clip(images, 0.0, 1.0), chosen
