"""Carte auto-organisatrice de Kohonen (SOM).

--- Ce qui change par rapport au K-means ---

Les deux apprennent un dictionnaire de k prototypes et encodent une image par
l'indice du plus proche : le codec est le même. Trois différences seulement :

    1. K-means est *batch* — il recalcule chaque centroïde comme la moyenne de
       tout son cluster. Kohonen est *stochastique* : on tire un exemple, on
       corrige les poids, on recommence.

    2. K-means ne bouge QUE le centroïde gagnant. Kohonen bouge aussi ses
       voisins, atténués par la distance.

    3. Ce voisinage n'existe que parce que les neurones sont posés sur une
       grille 2D — l'espace topologique. K-means n'en a pas : ses centroïdes
       n'ont aucune position les uns par rapport aux autres.

C'est la 3e qui fait tout l'intérêt : deux neurones voisins sur la grille étant
toujours tirés ensemble, ils finissent par apprendre des prototypes proches. Le
dictionnaire devient une carte qu'on peut lire, là où l'ordre des centroïdes
d'un K-means est arbitraire.

--- La règle de mise à jour ---

À chaque tirage d'un exemple Sj, on trouve le neurone gagnant k (celui dont le
feature vector est le plus proche de Sj), puis on corrige TOUS les neurones :

    Wi <- Wi + alpha · exp(-||Ci - Ck||² / (2·gamma)) · (Sj - Wi)

    Ci, Ck : positions sur la grille (PAS dans l'espace des images)
    alpha  : pas d'apprentissage — quelle part du chemin vers Sj on parcourt
    gamma  : largeur du voisinage — au-delà, l'exponentielle écrase la correction

Le facteur exponentiel vaut 1 pour le gagnant (Ci = Ck) et décroît avec
l'éloignement sur la grille : le gagnant prend la correction pleine, ses voisins
une version atténuée, les neurones lointains rien du tout.
"""

import numpy as np


def grid_coords(rows, cols):
    """Positions (k, 2) des neurones — leur espace topologique. Colonnes : (x, y).

    Le neurone i occupe la case (i // cols, i % cols). Ce sont ces coordonnées
    qui entrent dans ||Ci - Ck||², jamais les feature vectors.

    Grille hexagonale : lignes impaires décalées d'une demi-case, lignes espacées
    de √3/2. Les 6 voisins tombent alors tous à distance exactement 1 — c'est
    vérifiable : le voisin de la ligne du dessus est à (±0,5 ; √3/2), soit
    √(0,25 + 0,75) = 1. C'est la grille classique de Kohonen, la seule pour
    laquelle « voisin » a un sens unique.

    Le reste du code ne lit que des distances euclidiennes entre ces coordonnées.
    """
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    r = r.ravel().astype(np.float32)
    c = c.ravel().astype(np.float32)

    return np.stack([c + 0.5 * (r % 2), r * (np.sqrt(3) / 2)], axis=1).astype(np.float32)


def neighborhood_matrix(coords, gamma):
    """H (k, k) : H[i, k] = exp(-||Ci - Ck||² / (2·gamma)).

    gamma étant constant, ce facteur ne dépend que du couple (neurone, gagnant)
    et jamais des données : on le calcule une fois pour toutes plutôt qu'à
    chacune des dizaines de milliers d'itérations. H[:, k] donne alors
    directement la colonne de coefficients à appliquer quand k gagne.
    """
    if gamma <= 0:
        raise ValueError("gamma doit être strictement positif.")

    diff = coords[:, None, :] - coords[None, :, :]        # (k, k, 2)
    squared = np.sum(diff ** 2, axis=2)                   # ||Ci - Ck||²
    return np.exp(-squared / (2.0 * gamma)).astype(np.float32)


def initialize_weights(X, k, seed):
    """Initialise les k feature vectors en tirant k exemples au hasard.

    Comme pour le K-means : partir d'images réelles plutôt que de bruit place
    d'emblée les poids dans la zone où vivent les données.
    """
    X = np.asarray(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def compute_squared_distances(X, weights):
    """Distances au carré entre chaque image et chaque neurone -> (n, k).

    Via ||x - w||² = ||x||² - 2·x·w + ||w||². La forme naïve
    `((X[:, None, :] - weights) ** 2).sum(-1)` matérialise un tenseur
    (n, k, 784) — les 784 différences pixel à pixel de chaque couple — avant de
    les sommer, soit 784x le résultat utile. Ici le produit matriciel somme les
    pixels au passage : rien de plus gros que (n, k) n'est jamais alloué.
    """
    X = np.asarray(X, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]

    x_squared = np.sum(X ** 2, axis=1, keepdims=True)     # (n, 1)
    w_squared = np.sum(weights ** 2, axis=1)              # (k,)
    cross = X @ weights.T                                 # (n, k)

    distances = x_squared - 2.0 * cross + w_squared

    # Soustraire deux grands nombres proches perd des décimales : là où la
    # distance vaut zéro, le résultat peut sortir légèrement négatif (~-1e-5).
    # La forme naïve, elle, somme des carrés et ne peut jamais l'être — on
    # rétablit cette garantie, dont argmin et l'inertie dépendent.
    return np.maximum(distances, 0.0)


def find_bmu(sample, weights):
    """Indice du neurone gagnant (Best Matching Unit) pour un exemple."""
    return int(np.argmin(compute_squared_distances(sample, weights)[0]))


def assign_clusters(X, weights):
    """(n,) neurone gagnant de chaque image — l'équivalent des labels du K-means."""
    return np.argmin(compute_squared_distances(X, weights), axis=1).astype(np.int64)


def compute_inertia(X, weights, labels=None):
    """Somme des distances² de chaque image à son neurone gagnant.

    Même quantité que l'inertie du K-means, donc directement comparable — mais
    Kohonen ne la minimise pas : il minimise cette erreur SOUS la contrainte que
    les neurones voisins restent proches. Un SOM affiche donc une inertie un peu
    plus haute qu'un K-means de même k ; c'est le prix de l'organisation
    topologique, pas un défaut de l'entraînement.
    """
    X = np.asarray(X, dtype=np.float32)
    distances = compute_squared_distances(X, weights)
    if labels is None:
        return float(distances.min(axis=1).sum())
    labels = np.asarray(labels).astype(int)
    return float(distances[np.arange(len(X)), labels].sum())


def fit_kohonen(X, rows, cols, n_epochs, alpha, gamma, seed, verbose,
                return_history=False):
    """Entraîne une carte de Kohonen. Retourne (weights, labels).

    Args:
        X:        (n, d) données d'entraînement.
        rows,cols: forme de la grille ; k = rows · cols neurones.
        n_epochs: passes sur les données. Une époque = n tirages, soit chaque
                  exemple présenté une fois (dans un ordre remélangé à chaque
                  fois). L'algo tire un exemple au hasard ; parcourir une
                  permutation revient au même tirage sans remise près, et
                  garantit qu'aucun exemple n'est ignoré.
        alpha:    pas d'apprentissage, constant.
        gamma:    largeur du voisinage, constante.
        seed:     rend l'initialisation et l'ordre des tirages reproductibles.
        verbose:  affiche l'inertie au fil des époques.
        return_history: ajoute un 3e élément — la courbe d'entraînement :
                  l'inertie AVANT entraînement, puis à la fin de chaque époque.
                  Soit 1 + n_epochs points.

                  Le point de départ compte : une époque, c'est déjà n mises à
                  jour, et à alpha constant le SOM a quasiment convergé au bout
                  de la première. Sans lui, la courbe démarre après l'essentiel
                  de la descente et semble plate.

    Note : alpha et gamma restent constants, comme dans la formule. Les SOM
    « industriels » les font décroître au fil du temps (voisinage large puis
    resserré) ; ce n'est pas implémenté ici.
    """
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    k = int(rows) * int(cols)
    if k > n:
        raise ValueError(
            f"Grille {rows}x{cols} = {k} neurones pour {n} exemples : "
            f"l'initialisation tire k exemples distincts, il en faut au moins k."
        )

    coords = grid_coords(rows, cols)
    H = neighborhood_matrix(coords, gamma)      # (k, k), calculé une seule fois
    weights = initialize_weights(X, k, seed)

    rng = np.random.default_rng(seed)

    # Point de départ : l'inertie des poids initiaux (k exemples tirés au
    # hasard), avant la moindre mise à jour. C'est la référence à laquelle
    # comparer la suite.
    history = [compute_inertia(X, weights)]

    for epoch in range(int(n_epochs)):
        for j in rng.permutation(n):
            sample = X[j]

            # 1. Le neurone gagnant : celui dont le feature vector est le plus
            #    proche de l'exemple tiré.
            bmu = find_bmu(sample, weights)

            # 2. Wi <- Wi + alpha · H[i, bmu] · (Sj - Wi), pour TOUS les i.
            #    H[:, bmu] est la colonne des coefficients de voisinage ; le
            #    [:, None] la diffuse sur les 784 pixels de chaque poids.
            weights += alpha * H[:, bmu][:, None] * (sample - weights)

        inertia = compute_inertia(X, weights)
        history.append(inertia)

        if verbose:
            print(f"Époque {epoch + 1}/{n_epochs} : inertie = {inertia:.4f}")

    labels = assign_clusters(X, weights)

    if return_history:
        return weights, labels, history
    return weights, labels
