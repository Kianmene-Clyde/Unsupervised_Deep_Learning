"""Génération de données synthétiques avec un SOM — la 3e tâche du projet.

--- Le SOM a un atout que le K-means n'a pas : sa TOPOLOGIE ---

Comme le K-means, un SOM ne rendra jamais que ses k feature vectors par
decode(). Mais sa grille ordonne les prototypes : deux cases voisines se
ressemblent, par construction. On peut donc générer une image NOUVELLE en
tirant un point ENTRE deux cases voisines et en interpolant leurs feature
vectors :

    1. neurone gagnant c ~ proportions d'activation (mesurées sur les données) ;
    2. voisin v : une case adjacente de la grille, au hasard ;
    3. image = (1 − t) · w_c + t · w_v,  t ~ U(0, ½)   (+ un léger bruit).

L'interpolation entre voisins d'un SOM reste plausible LÀ OÙ celle entre deux
centroïdes K-means quelconques traverserait le vide (la moyenne d'un « 3 » et
d'un « 7 » n'est pas un chiffre). C'est la topologie qui paie. Les images
restent toutefois des mélanges de prototypes existants : pour inventer des
formes neuves, il faut un modèle génératif entraîné pour ça (VAE, GAN) — la
comparaison chiffrée est dans le rapport final. NumPy pur, comme le SOM.
"""

import numpy as np

IMAGE_DIM = 784


def activation_weights(labels, k):
    """Proportion d'images gagnées par chaque neurone (les morts pèsent 0)."""
    labels = np.asarray(labels).astype(int).ravel()
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else np.full(k, 1.0 / k)


def _neighbors(code, rows, cols):
    """Les cases adjacentes (4-connexité rectangulaire) d'un neurone."""
    r, c = divmod(int(code), cols)
    around = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [rr * cols + cc for rr, cc in around if 0 <= rr < rows and 0 <= cc < cols]


def sample_synthetic(weights, rows, cols, act_weights, n_images, seed=42,
                     noise=0.02):
    """Tire n_images par interpolation entre neurones voisins.

    → (images (n, 784), neurones de départ (n,)).
    noise ajoute un léger bruit gaussien uniforme — assez pour que deux tirages
    de la même paire diffèrent, pas assez pour neiger l'image (contrairement au
    K-means, le gros de la nouveauté vient de l'INTERPOLATION, pas du bruit).
    """
    W = np.asarray(weights, dtype=np.float32).reshape(-1, IMAGE_DIM)
    rng = np.random.default_rng(seed)

    starts = rng.choice(len(W), size=n_images, p=act_weights)
    images = np.empty((n_images, IMAGE_DIM), dtype=np.float32)
    for i, c in enumerate(starts):
        vs = _neighbors(c, rows, cols)
        v = vs[rng.integers(len(vs))]
        t = rng.uniform(0.0, 0.5)
        images[i] = (1.0 - t) * W[c] + t * W[v]
    images += noise * rng.standard_normal(images.shape).astype(np.float32)
    return np.clip(images, 0.0, 1.0), starts
