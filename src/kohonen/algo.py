"""Adaptateur Kohonen (SOM) — ce que src/kohonen présente à l'app.

Comme pour le K-means, aucun calcul ici : uniquement la traduction vers
l'interface commune (algo_base, à la racine de src/).

La grille (rows, cols) est le point délicat : elle n'a pas d'équivalent
K-means. Elle voyage donc dans les MÉTADONNÉES du modèle, et toutes les vues la
relisent de là — jamais d'un réglage courant de l'UI, qui pourrait ne pas
correspondre au modèle chargé.
"""

import numpy as np

from .utils.codec import decode, encode
from .utils.generation import activation_weights, sample_synthetic
from .utils.kohonen import assign_clusters, fit_kohonen
from .utils.metrics import mse_from_inertia
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils.visualization import (
    plot_class_distribution,
    plot_inertia,
    plot_prototype_map,
    show_images,
    show_reconstructions,
)

from algo_base import Algo, Param

IMAGE_DIM = 784


class Kohonen(Algo):
    key = "kohonen"
    label = "Kohonen (SOM)"
    title = "Entraîner une carte de Kohonen"

    dict_label = "Feature vectors"
    code_label = "feature vector"
    k_label = "Neurones (lignes × colonnes)"

    # minimum=0 et non 0,0001 pour alpha et gamma : les valeurs légales d'un champ
    # sont `minimum + n · step`, donc un minimum non aligné sur le step rend toute
    # la grille de valeurs bancale (0,0001 / 0,0101 / 0,0201…). Zéro est aligné sur
    # n'importe quel step ; c'est check() qui refuse le zéro lui-même, côté serveur.
    params = [
        Param("rows", "Lignes de la grille", "int", 10, minimum=1,
              info="k = lignes × colonnes"),
        Param("cols", "Colonnes de la grille", "int", 10, minimum=1,
              info="k = lignes × colonnes"),
        Param("alpha", "Alpha — pas d'apprentissage", "float", 0.1,
              minimum=0.0, maximum=1.0, step=0.01,
              info="Part du chemin vers l'exemple parcourue à chaque présentation. "
                   "Grand = rapide mais oscille davantage."),
        Param("gamma", "Gamma — largeur du voisinage", "float", 1.0,
              minimum=0.0, step=0.05,
              info="Portée du voisinage, en cases de grille. À 1, un voisin direct "
                   "reçoit 61 % de la correction. Sous ~0,25 le SOM dégénère en "
                   "K-means (plus d'organisation) ; au-delà de ~5 tous les neurones "
                   "apprennent la même image. À réajuster si tu changes la grille."),
        Param("n_epochs", "Époques", "int", 8, minimum=1,
              info="À alpha constant, l'essentiel de la descente tient dans la 1re."),
        Param("seed", "Seed", "int", 42, minimum=0,
              info="Tirage initial + ordre de présentation."),
    ]

    # ------------------------------------------------------------ Entraînement

    def k_of_params(self, p):
        return int(p.get("rows") or 0) * int(p.get("cols") or 0)

    def check(self, p, n_samples):
        rows, cols = int(p.get("rows") or 0), int(p.get("cols") or 0)
        if rows < 1 or cols < 1:
            raise ValueError("La grille doit avoir au moins 1 ligne et 1 colonne.")
        if float(p.get("gamma") or 0) <= 0:
            raise ValueError("Gamma doit être strictement positif.")
        if float(p.get("alpha") or 0) <= 0:
            raise ValueError("Alpha doit être strictement positif.")
        if int(p.get("n_epochs") or 0) < 1:
            raise ValueError("Il faut au moins 1 époque.")
        # initialize_weights tire k exemples DISTINCTS : sous ce seuil, fit_kohonen
        # lève de lui-même, mais autant le dire clairement ici.
        if rows * cols > n_samples:
            raise ValueError(
                f"Grille {rows}x{cols} = {rows * cols} neurones pour {n_samples} images : "
                f"l'initialisation tire un exemple distinct par neurone. Réduis la "
                f"grille ou augmente les images."
            )

    def auto_name(self, ds_key, p):
        return (f"{ds_key}_som{int(p.get('rows') or 0)}x{int(p.get('cols') or 0)}"
                f"_n{int(p.get('n_samples') or 0)}")

    def train(self, X, p):
        rows, cols = int(p["rows"]), int(p["cols"])
        weights, labels, history = fit_kohonen(
            X=X,
            rows=rows,
            cols=cols,
            n_epochs=int(p["n_epochs"]),
            alpha=float(p["alpha"]),
            gamma=float(p["gamma"]),
            seed=int(p["seed"]),
            verbose=False,
            return_history=True,
        )
        weights = np.asarray(weights, dtype=np.float32)

        return weights, {
            # k pour l'app (taille du dictionnaire) ; rows/cols pour les vues,
            # qui ne sauraient rien afficher sans la forme de la grille.
            "k": rows * cols,
            "rows": rows,
            "cols": cols,
            "alpha": float(p["alpha"]),
            "gamma": float(p["gamma"]),
            "n_epochs": int(p["n_epochs"]),
            "seed": int(p["seed"]),
            "inertia": float(history[-1]),
            "inertia_history": history,
        }

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        return np.asarray(assign_clusters(X, weights))

    def encode(self, image, weights):
        return int(encode(image, weights))

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

    def _grid(self, meta):
        """(rows, cols) du modèle — lus des métadonnées, jamais de l'UI."""
        return int(meta["rows"]), int(meta["cols"])

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

    # plot_latent : pas d'override — le latent du SOM est un entier discret,
    # il n'y a rien à projeter (défaut None d'algo_base, l'app masque les nuages).

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        rows, cols = self._grid(meta)
        fig = plot_prototype_map(
            weights, rows, cols, cluster_labels=labels, y_true=y_true,
            class_names=class_names,
            title="Carte de Kohonen — feature vectors", show=False,
        )
        note = (
            f"**Les {rows * cols} feature vectors**, chacun à sa place sur la grille "
            f"hexagonale. C'est la vue signature du SOM : les cases **voisines se "
            f"ressemblent**, ce qu'impose la règle de voisinage. La même vue pour un "
            f"K-means n'est qu'une grille d'affichage arbitraire — ses centroïdes "
            f"n'ont pas de voisins.\n\n"
            f"Les cases grisées sont des **neurones morts** : jamais gagnants, leur "
            f"poids n'a été appris que passivement, par voisinage. Le chiffre en cyan "
            f"est la classe majoritaire du neurone."
        )
        return fig, note

    def plot_curve(self, history, n_samples):
        fig = plot_inertia(history, n_samples=n_samples, image_dim=IMAGE_DIM, show=False)

        fr = lambda v: f"{v:,.0f}".replace(",", " ")
        mse_end = mse_from_inertia(history[-1], n_samples, IMAGE_DIM)

        note = (
            f"**{len(history) - 1} époque(s)** — l'inertie passe de {fr(history[0])} "
            f"(avant entraînement) à {fr(history[-1])}, soit "
            f"−{(1 - history[-1] / history[0]) * 100:.1f} %.\n\n"
            "Le point **0** est l'état initial. À alpha constant, l'essentiel de la "
            "descente tient dans la première époque : les suivantes oscillent autour "
            "d'un équilibre.\n\n"
            "⚠️ Contrairement au K-means, une **remontée n'est pas un bug**. Le SOM ne "
            "minimise pas l'inertie seule mais un compromis entre elle et la contrainte "
            "de voisinage, et comme `alpha` ne décroît jamais, chaque exemple continue "
            "de tirer les poids vers lui. Le SOM ne se fige pas : il fluctue autour de "
            "son équilibre.\n\n"
            f"MSE finale : **{mse_end:.4f}** par pixel. À k égal, elle est un peu plus "
            f"haute que celle d'un K-means — c'est le prix de l'organisation "
            f"topologique, pas un défaut d'entraînement."
        )
        return fig, note

    def extra_figures(self, weights, meta, labels, y_true, class_names):
        # Les histogrammes de composition par neurone — la même lecture de
        # l'espace latent (discret) que pour le K-means : un groupe de barres
        # par code, une barre par classe réelle.
        rows, cols = self._grid(meta)
        return [(
            "Distribution des classes réelles par neurone",
            plot_class_distribution(
                labels, y_true, class_names=class_names, k=rows * cols,
                title="Distribution des classes réelles par neurone", show=False,
            ),
        )]

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Les vues sans données + la GÉNÉRATION par interpolation topologique.

        La 3e tâche du projet : le SOM génère en interpolant entre neurones
        VOISINS de sa grille (voir utils/generation.py) — c'est sa topologie qui
        rend l'interpolation plausible, là où celle de deux centroïdes K-means
        quelconques traverserait le vide. Les images restent des mélanges de
        prototypes : les formes vraiment neuves demandent un VAE ou un GAN.
        """
        figures = self.extra_figures(weights, meta, labels, y_true, class_names)

        rows, cols = self._grid(meta)
        seed = int(meta.get("seed", 42))
        act = activation_weights(labels, rows * cols)
        images, starts = sample_synthetic(weights, rows, cols, act,
                                          n_images=16, seed=seed)
        fig = show_images(
            images, titles=[f"depuis n°{c}" for c in starts],
            n_rows=4, n_cols=4, figsize=(8, 8.6), show=False,
        )
        fig.suptitle(
            "Génération — interpolation entre neurones voisins\n"
            "la topologie de la grille rend le « entre-deux » plausible",
            fontsize=12, fontweight="bold",
        )
        figures.append(("Génération synthétique (tâche 3)", fig))
        return figures

    def describe_rows(self, meta):
        rows, cols = self._grid(meta)
        return [
            ("Grille", f"{rows} × {cols} = {rows * cols} neurones"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("Inertie finale", f"{meta['inertia']:.1f}"),
            ("Époques", f"{meta['n_epochs']}"),
            ("Alpha", f"{meta['alpha']}"),
            ("Gamma", f"{meta['gamma']}"),
            ("Seed", f"{meta['seed']}"),
        ]
