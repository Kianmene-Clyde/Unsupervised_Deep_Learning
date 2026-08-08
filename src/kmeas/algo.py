"""Adaptateur K-means — ce que src/kmeas présente à l'app.

Ce fichier ne contient AUCUN calcul : tout vit dans utils/. Il ne fait que
traduire (noms d'arguments, tenseurs TensorFlow -> numpy, métadonnées) pour que
l'app n'ait rien à savoir du K-means.

Il vit ici, et non dans un dossier d'adaptateurs à part : le K-means est tout
entier sous src/kmeas/ — son algo, son codec, son registre, ses visus et la
façon dont il se présente à l'app.

algo_base vit à la racine de src/, comme data_import : ces deux-là sont le
terrain commun des algos.
"""

import numpy as np

from .utils.codec import decode, encode
from .utils.generation import cluster_stats, sample_synthetic
from .utils.kmeas import (
    assign_clusters,
    compute_inertia,
    compute_squared_distances,
    fit_kmeans,
)
from .utils.metrics import mse_from_inertia
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils.visualization import (
    plot_centroid_map,
    plot_class_distribution,
    plot_inertia,
    show_images,
    show_reconstructions,
)

from algo_base import Algo, Param, history_of

IMAGE_DIM = 784


class KMeans(Algo):
    key = "kmeans"
    label = "K-means"
    title = "Entraîner un K-means"

    dict_label = "Centroïdes"
    code_label = "centroïde"
    k_label = "K — nombre de clusters"

    params = [
        Param("k", "K — nombre de clusters (sans plafond)", "int", 10, minimum=1,
              info="La taille du dictionnaire, donc le taux de compression."),
        Param("max_iter", "Itérations max (sans plafond)", "int", 100, minimum=1,
              info="Plafond : la boucle s'arrête d'elle-même à convergence."),
        Param(
            "tolerance", "Tolérance de convergence", "choice", 1e-4,
            choices=[("1e-6", 1e-6), ("1e-5", 1e-5), ("1e-4", 1e-4), ("1e-3", 1e-3)],
            info="Déplacement des centroïdes sous lequel on considère avoir convergé.",
        ),
        Param("seed", "Seed", "int", 42, minimum=0,
              info="Tirage des centroïdes initiaux."),
    ]

    # ------------------------------------------------------------ Entraînement

    def k_of_params(self, p):
        return int(p.get("k") or 0)

    def check(self, p, n_samples):
        k = int(p.get("k") or 0)
        if k < 1:
            raise ValueError("K doit valoir au moins 1.")
        if int(p.get("max_iter") or 0) < 1:
            raise ValueError("Il faut au moins 1 itération.")
        # initialize_centroids tire k indices parmi n sans remise : si k > n, elle
        # renvoie silencieusement moins de k centroïdes et fit_kmeans casse plus
        # loin sur un « slice index out of bounds » incompréhensible.
        if k > n_samples:
            raise ValueError(
                f"K={k} dépasse le nombre d'images ({n_samples}) : impossible d'avoir "
                f"plus de clusters que de points. Baisse K ou augmente les images."
            )

    def auto_name(self, ds_key, p):
        return f"{ds_key}_k{int(p.get('k') or 0)}_n{int(p.get('n_samples') or 0)}"

    def train(self, X, p):
        centroids, labels, history = fit_kmeans(
            X=X,
            k=int(p["k"]),
            max_iter=int(p["max_iter"]),
            tolerance=float(p["tolerance"]),
            seed=int(p["seed"]),
            verbose=False,
            return_history=True,
        )
        # fit_kmeans rend des tenseurs TF : l'app et le registre veulent du numpy.
        centroids = np.asarray(centroids, dtype=np.float32)
        inertia = float(compute_inertia(X, labels, centroids))

        return centroids, {
            "k": int(p["k"]),
            "max_iter": int(p["max_iter"]),
            "tolerance": float(p["tolerance"]),
            "seed": int(p["seed"]),
            "inertia": inertia,
            # Persisté avec le modèle : sans ça, la courbe d'entraînement serait
            # perdue à la fin de fit_kmeans et introuvable au rechargement.
            "inertia_history": history,
        }

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        return np.asarray(assign_clusters(compute_squared_distances(X, weights)))

    def encode(self, image, weights):
        return int(np.asarray(encode(image, weights)))

    def decode(self, code, weights):
        return np.asarray(decode(code, weights), dtype=np.float32)

    # --------------------------------------------------------------- Registre

    def list_models(self):
        return list_models()

    def load(self, name):
        return load_model(name)

    def save(self, name, weights, meta):
        return save_model(name, weights, meta)

    def delete(self, name):
        delete_model(name)

    # ------------------------------------------------------------------- Vues

    def plot_reconstructions(self, originals, reconstructions, top_labels,
                             bottom_labels, title):
        return show_reconstructions(
            originals, reconstructions, top_labels=top_labels,
            bottom_labels=bottom_labels, title=title, show=False,
        )

    def plot_distribution(self, codes, y_true, class_names, k, title):
        return plot_class_distribution(
            codes, y_true, class_names=class_names, k=k, title=title, show=False,
        )

    # plot_latent : pas d'override — le latent du K-means est un entier discret,
    # il n'y a rien à projeter (défaut None d'algo_base, l'app masque les nuages).

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        k = int(meta["k"])

        # La grille s'adapte à K, sinon 10 colonnes fixes donnent une figure de
        # 170 pouces de haut à K=1000 (23 s de rendu). En grille ~carrée le même
        # K=1000 tient en 16x16 pouces. Au-delà de 50 centroïdes les titres
        # « code i » deviennent illisibles : on les retire plutôt que d'agrandir.
        if k <= 50:
            n_cols = min(k, 10)
            titles = [f"code {i}" for i in range(k)]
            cell_w, cell_h = 1.5, 1.7
        else:
            n_cols = int(np.ceil(np.sqrt(k)))
            titles = None
            cell_w = cell_h = 0.5

        n_rows = int(np.ceil(k / n_cols))
        fig = show_images(
            weights, titles=titles, n_rows=n_rows, n_cols=n_cols,
            figsize=(n_cols * cell_w, n_rows * cell_h), show=False,
        )

        note = (
            f"**Les {k} centroïdes du modèle**, en grille {n_rows}×{n_cols}. Chaque "
            "centroïde est l'image moyenne de son cluster. `decode(code)` renvoie "
            "exactement une de ces images : c'est tout le vocabulaire du codec."
        )
        if titles is None:
            note += (
                f"\n\n*Les numéros de code sont masqués au-delà de 50 centroïdes "
                f"(illisibles à cette taille) : ils se lisent de gauche à droite, "
                f"ligne par ligne, de 0 à {k - 1}.*"
            )
        return fig, note

    def plot_curve(self, history, n_samples):
        fig = plot_inertia(history, n_samples=n_samples, image_dim=IMAGE_DIM, show=False)

        fr = lambda v: f"{v:,.0f}".replace(",", " ")
        mse_start = mse_from_inertia(history[0], n_samples, IMAGE_DIM)
        mse_end = mse_from_inertia(history[-1], n_samples, IMAGE_DIM)

        note = (
            f"**{len(history)} itération(s)** — l'inertie passe de {fr(history[0])} à "
            f"{fr(history[-1])}, soit −{(1 - history[-1] / history[0]) * 100:.1f} %.\n\n"
            "L'inertie est la somme des distances au carré de chaque point à son "
            "centroïde : c'est exactement ce que K-means minimise, **sa loss**. Elle "
            "**ne peut que décroître** — une remontée signalerait un bug, pas un "
            "mauvais réglage. Le plateau final est le point fixe où `has_converged` "
            "arrête la boucle.\n\n"
            f"L'axe **rouge de droite** gradue la même courbe en MSE : "
            f"{mse_start:.4f} → **{mse_end:.4f}** par pixel. Une seule courbe et non "
            f"deux, car ce sont **les mêmes valeurs** : le codec reconstruit chaque "
            f"image par son centroïde, celui-là même dont l'inertie mesure l'écart. "
            f"D'où `MSE = inertie / (n × 784)`."
        )
        return fig, note

    def extra_figures(self, weights, meta, labels, y_true, class_names):
        # plot_centroid_map trie par taille de cluster et annote la classe
        # dominante : il lui faut les assignations, pas juste les centroïdes.
        return [(
            "Cartographie des centroïdes",
            plot_centroid_map(
                weights, cluster_labels=labels, y_true=y_true,
                sort_by_size=True, title="Cartographie des centroïdes", show=False,
            ),
        )]

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Les vues sans données + la GÉNÉRATION, qui a besoin des images.

        La 3e tâche du projet : le K-means génère en tirant d'un mélange de
        gaussiennes diagonales ajusté sur ses clusters (voir utils/generation.py).
        La figure montre honnêtement le résultat — des centroïdes bruités — et
        c'est le but : chiffrer pourquoi la génération demande un VAE ou un GAN.
        """
        figures = self.extra_figures(weights, meta, labels, y_true, class_names)

        k = int(meta["k"])
        seed = int(meta.get("seed", 42))
        props, sigmas = cluster_stats(X, labels, k)
        images, chosen = sample_synthetic(weights, props, sigmas,
                                          n_images=16, seed=seed)
        fig = show_images(
            images, titles=[f"cluster {c}" for c in chosen],
            n_rows=4, n_cols=4, figsize=(8, 8.6), show=False,
        )
        fig.suptitle(
            "Génération — mélange de gaussiennes sur les clusters\n"
            "centroïde + σ·ε par pixel : le bruit s'ajoute sans respecter les traits",
            fontsize=12, fontweight="bold",
        )
        figures.append(("Génération synthétique (tâche 3)", fig))
        return figures

    def describe_rows(self, meta):
        rows = [
            ("K — clusters", f"{meta['k']}"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("Inertie finale", f"{meta['inertia']:.1f}"),
        ]
        history = history_of(meta)
        if history:
            rows.append(("Itérations effectuées", f"{len(history)} / {meta['max_iter']} max"))
        rows += [
            ("Tolérance", f"{meta['tolerance']:.0e}"),
            ("Seed", f"{meta['seed']}"),
        ]
        return rows
