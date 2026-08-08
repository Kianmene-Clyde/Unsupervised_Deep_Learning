"""Adaptateur PCA — ce que src/pca présente à l'app.

Comme pour K-means et Kohonen : aucun calcul « métier » ici. La PCA du projet
vit dans src/utils/pca/ (dim_reduction, compression, generation — la matière
des notebooks de ce dossier) et N'EST PAS modifiée. Ce fichier ne fait que la
traduire vers l'interface commune (algo_base) :

- Les poids de l'app sont UN tableau (k+1, 784) : ligne 0 l'image moyenne,
  lignes 1..k les composantes principales. Le spectre complet des variances
  voyage dans les métadonnées — scree plot et courbe MSE(k) s'en déduisent
  sans jamais réentraîner.
- Le code d'une image est un VECTEUR de k flottants, pas un entier :
  utils/codec.py l'enveloppe (hashable, affichable) pour que l'onglet codec
  de l'app fonctionne tel quel. codec_note / latent_note remplacent le récit
  « code entier » des autres algos, qui mentirait ici.
- Les tracés d'utils/pca sont écrits pour les notebooks (plt.show(), pas de
  figure retournée) : utils/visualization.py les capture via le backend Agg.
  CHAQUE vue des notebooks est ainsi reprise : eigen-images (dictionnaire),
  compromis MSE/compression (courbe), et en vues propres le spectre, les
  heatmaps ±, la comparaison des centrages, les reconstructions à k croissant,
  la MSE normalisée et les trois vues de génération.
"""

import numpy as np

from utils.pca.compression import (
    compress,
    plot_normalized_comparison,
    plot_reconstruction_grid,
    plot_tradeoff,
    reconstruct,
    reconstruction_error,
)
from utils.pca.dim_reduction import (
    PCA as PCAModel,
    fit_pca,
    plot_component_heatmaps,
    plot_eigenimages,
    plot_pc1_vs_mean,
    plot_spectrum,
    plot_spectrum_comparison,
)
from utils.pca.generation import (
    fit_latent_gaussian,
    plot_generated_grid,
    plot_interpolation,
    plot_real_vs_generated,
)

from .utils import codec
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils.visualization import (
    captured,
    plot_dominant_distribution,
    plot_projection_panels,
    show_reconstructions,
)

from algo_base import Algo, Param

IMAGE_DIM = 784


class PCAAlgo(Algo):
    key = "pca"
    label = "PCA"
    title = "Entraîner une PCA (analyse en composantes principales)"

    dict_label = "Composantes principales"
    code_label = "vecteur de coefficients"
    k_label = "k — nombre de composantes"

    params = [
        Param("n_components", "k — nombre de composantes", "int", 50,
              minimum=1, maximum=IMAGE_DIM,
              info="Axes principaux conservés : la taille du code de chaque image "
                   "(784 = reconstruction exacte)."),
        Param(
            "center", "Centrage", "choice", "feature",
            choices=[("Par pixel — PCA standard", "feature"),
                     ("Global — démo : PC1 gaspillée sur l'image moyenne", "global")],
            info="Quelle moyenne est soustraite avant la décomposition. Voir la vue "
                 "« Centrage — PC1 vs image moyenne ».",
        ),
    ]

    # ------------------------------------------------------------ Entraînement

    def k_of_params(self, p):
        return int(p.get("n_components") or 0)

    def check(self, p, n_samples):
        k = int(p.get("n_components") or 0)
        if k < 1:
            raise ValueError("Il faut au moins 1 composante.")
        if k > IMAGE_DIM:
            raise ValueError(
                f"k={k} dépasse les {IMAGE_DIM} pixels : au-delà, il n'existe "
                f"plus d'axe à extraire."
            )
        if n_samples < 2:
            raise ValueError("Il faut au moins 2 images : la covariance divise par n − 1.")
        # La covariance de n images est de rang ≤ n − 1 : les composantes
        # au-delà ont une variance nulle, ce ne serait que du bruit numérique.
        if k >= n_samples:
            raise ValueError(
                f"k={k} pour {n_samples} images : la covariance est de rang au plus "
                f"{n_samples - 1}, les composantes au-delà seraient du bruit. Baisse "
                f"k ou augmente les images."
            )

    def auto_name(self, ds_key, p):
        return (f"{ds_key}_pca{int(p.get('n_components') or 0)}"
                f"_{p.get('center', 'feature')}_n{int(p.get('n_samples') or 0)}")

    def train(self, X, p):
        k = int(p["n_components"])
        center = str(p["center"])
        X = np.asarray(X, dtype=np.float64)
        n = len(X)

        # Décomposition COMPLÈTE puis tronquée à k : l'eigendécomposition coûte
        # pareil, et le spectre entier (784 variances) permet de tracer le scree
        # plot et la courbe MSE(k) plus tard sans jamais refitter.
        full = fit_pca(X, n_components=None, center=center)
        pca = PCAModel(
            mean_=full.mean_,
            components_=full.components_[:k],
            explained_variance_=full.explained_variance_[:k],
            explained_variance_ratio_=full.explained_variance_ratio_[:k],
        )

        # Inertie exacte — mesurée comme le codec la produira, clip [0, 1] compris.
        X_hat = reconstruct(pca, compress(pca, X))
        inertia = reconstruction_error(X, X_hat) * n * IMAGE_DIM

        # « Courbe d'entraînement » : la PCA n'itère pas, sa courbe est la MSE si
        # l'on ne garde que 1, 2, … k composantes. Elle se déduit du spectre :
        # l'erreur à j composantes est la variance des axes abandonnés
        # (facteur n−1 : les variances sont estimées avec ddof=1).
        eig = full.explained_variance_
        residual = eig.sum() - np.cumsum(eig[:k])
        history = [float(v) for v in (n - 1) * residual]
        history[-1] = float(inertia)  # le dernier point colle à la mesure exacte

        weights = np.vstack([pca.mean_, pca.components_]).astype(np.float32)
        return weights, {
            "k": k,
            "center": center,
            "inertia": float(inertia),
            "inertia_history": history,
            # Le spectre COMPLET (784 valeurs), pas seulement les k gardées : le
            # scree plot doit montrer la variance cumulée jusqu'à 100 %.
            "explained_variance": [float(v) for v in eig],
            "explained_variance_ratio": [float(v) for v in full.explained_variance_ratio_],
            "explained_variance_kept": float(full.explained_variance_ratio_[:k].sum()),
        }

    # ------------------------------------------------------------------ Codec

    def _pca_of(self, weights, meta):
        """L'objet PCA du modèle, variances exactes comprises (lues des métadonnées)."""
        weights = np.asarray(weights, dtype=np.float64)
        k = int(meta["k"])
        ev = np.asarray(meta["explained_variance"], dtype=np.float64)
        ratio = np.asarray(meta["explained_variance_ratio"], dtype=np.float64)
        return PCAModel(
            mean_=weights[0],
            components_=weights[1:],
            explained_variance_=ev[:k],
            explained_variance_ratio_=ratio[:k],
        )

    def _spectrum_pca_of(self, weights, meta):
        """Même modèle, mais portant le spectre COMPLET (784 variances).

        plot_spectrum trace la variance cumulée jusqu'à 100 % : tronquée à k,
        la courbe s'arrêterait avant les seuils 90/95 % qu'elle annote.
        """
        weights = np.asarray(weights, dtype=np.float64)
        return PCAModel(
            mean_=weights[0],
            components_=weights[1:],
            explained_variance_=np.asarray(meta["explained_variance"], dtype=np.float64),
            explained_variance_ratio_=np.asarray(
                meta["explained_variance_ratio"], dtype=np.float64
            ),
        )

    def assign(self, X, weights):
        """La composante dominante de chaque image : argmax |zᵢ|.

        La PCA n'a pas de clusters ; c'est son équivalent d'une assignation
        discrète, utilisé par la vue « distribution des classes ».
        """
        Z = codec.pca_from_weights(weights).transform(np.asarray(X, dtype=np.float64))
        return np.argmax(np.abs(np.atleast_2d(Z)), axis=1)

    def encode(self, image, weights):
        return codec.encode(image, weights)

    def decode(self, code, weights):
        return codec.decode(code, weights)

    # --------------------------------------------------------------- Registre

    def list_models(self):
        return list_models()

    def load(self, name):
        return load_model(name)

    def save(self, name, weights, meta):
        return save_model(name, weights, meta)

    def delete(self, name):
        delete_model(name)

    # ------------------------------------------------------------------ Récits

    def codec_note(self, meta, ds_key, n_images, seed, stats):
        k = int(meta["k"])
        bits = k * 32  # k flottants float32, contre UN entier pour K-means/SOM
        ratio = IMAGE_DIM * 8 / bits
        mse = meta["inertia"] / (meta["n_samples"] * IMAGE_DIM)
        kept = meta.get("explained_variance_kept", 0.0)
        return (
            f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}**, "
            f"encodées par **PCA** (graine {seed} — relance le tirage pour en voir "
            f"d'autres).\n\n"
            f"Contrairement à K-means ou au SOM, le code n'est **pas un entier** : chaque "
            f"image (784 pixels) est transmise comme **{k} flottant(s)** — ses coordonnées "
            f"sur les composantes principales — soit {bits} bits en float32, un taux de "
            f"compression de **{ratio:.1f}:1** face aux {IMAGE_DIM * 8} bits de l'image "
            f"brute (la moyenne et les composantes n'étant transmises qu'une fois).\n\n"
            f"Le décodeur reconstruit `image ≈ moyenne + Σ zᵢ·PCᵢ`. Le code étant "
            f"**continu**, chaque image reçoit le sien et aucune colonne ne se répète — "
            f"d'où les {stats['Train']}/{n_images} codes distincts en train et "
            f"{stats['Test']}/{n_images} en test : c'est attendu, pas un bug.\n\n"
            f"Le prix de la compression est la **MSE : {mse:.4f}** par pixel — les {k} "
            f"composante(s) conservent {kept * 100:.1f} % de la variance du train ; le "
            f"flou des reconstructions est la variance abandonnée."
        )

    def latent_note(self, meta):
        k = int(meta["k"])
        return (
            f"Ici, l'espace latent **est** continu : ℝ^{k}, les coordonnées de chaque "
            f"image sur les composantes principales du modèle. Les nuages montrent "
            f"PC1 × PC2 (et PC1 × PC2 × PC3) — **les vraies premières dimensions du "
            f"code**, pas une projection d'illustration.\n\n"
            "✅ Contrairement aux codecs discrets, train et test sont projetés sur **les "
            "axes du modèle** : les deux figures partagent le même repère et se comparent "
            "directement, positions comprises.\n\n"
            "**Distribution** : la PCA n'assigne pas de cluster ; chaque image est rangée "
            "sous sa **composante dominante** (celle où |zᵢ| est maximal). La carte "
            "classes × composantes montre si certaines composantes « appartiennent » à "
            "certaines classes — la pureté se lit comme pour des clusters."
        )

    # ------------------------------------------------------------------- Vues

    def plot_reconstructions(self, originals, reconstructions, top_labels,
                             bottom_labels, title):
        return show_reconstructions(
            originals, reconstructions, top_labels=top_labels,
            bottom_labels=bottom_labels, title=title,
        )

    def plot_distribution(self, codes, y_true, class_names, k, title):
        return plot_dominant_distribution(
            codes, y_true, class_names=class_names, k=k, title=title,
        )

    def plot_latent(self, X, weights, meta, y_true, class_names, y_label, title):
        pca = self._pca_of(weights, meta)
        coords = np.atleast_2d(pca.transform(np.asarray(X, dtype=np.float64)))
        labels = np.argmax(np.abs(coords), axis=1)
        return plot_projection_panels(
            coords, y=y_true, class_names=class_names, labels=labels,
            ratios=pca.explained_variance_ratio_, title=title, y_label=y_label,
        )

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        pca = self._pca_of(weights, meta)
        k = int(meta["k"])
        n_show = min(k, 12)
        fig = captured(
            plot_eigenimages, pca, n=n_show,
            suptitle="Image moyenne et composantes principales (eigen-images)",
        )
        note = (
            f"**L'image moyenne + les {n_show} première(s) des {k} composantes "
            f"principales** du modèle — les « eigen-images ». Contrairement à un "
            f"centroïde, une composante n'est pas une image type mais une **direction "
            f"de variation** : le décodeur reconstruit `image ≈ moyenne + Σ zᵢ·PCᵢ`. "
            f"C'est exactement {{moyenne, PC₁…PC{k}}} qui constitue le dictionnaire "
            f"transmis une fois au décodeur "
            f"({meta.get('explained_variance_kept', 0) * 100:.1f} % de la variance "
            f"du train conservée)."
        )
        if n_show < k:
            note += (
                f"\n\n*Seules les {n_show} premières sont affichées : plus on descend "
                f"dans le spectre, plus les composantes ressemblent à du bruit haute "
                f"fréquence.*"
            )
        return fig, note

    def plot_curve(self, history, n_samples):
        ks = np.arange(1, len(history) + 1)
        mses = np.asarray(history, dtype=np.float64) / (n_samples * IMAGE_DIM)
        ratios = IMAGE_DIM / ks
        fig = captured(
            plot_tradeoff, ks, mses, ratios,
            suptitle="Compromis qualité / taille — MSE et taux de compression selon k",
        )
        note = (
            f"La PCA n'a **pas d'entraînement itératif** : la décomposition est exacte, "
            f"en un calcul. Sa « courbe d'entraînement » est donc le compromis "
            f"**qualité / taille** : la MSE de reconstruction si l'on ne gardait que "
            f"1, 2, … {len(history)} composantes — déduite du spectre des variances, "
            f"sans réentraîner — face au taux de compression 784∕k.\n\n"
            f"MSE finale à k={len(history)} : **{mses[-1]:.4f}** par pixel. La courbe "
            f"plonge vite puis s'aplatit : les premières composantes portent l'essentiel "
            f"de la variance — c'est tout l'intérêt de la PCA, et ce que le scree plot "
            f"(onglet « Vues de l'algo ») montre sous un autre angle."
        )
        return fig, note

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Toutes les vues des notebooks PCA qui n'ont pas d'onglet dédié.

        A besoin des IMAGES, pas seulement des poids : comparaison des centrages
        (refits sur X), reconstructions à k croissant, gaussiennes latentes pour
        la génération et l'interpolation.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y_true).astype(int).ravel()
        k = int(meta["k"])
        n = len(X)
        center = meta.get("center", "feature")
        pca = self._pca_of(weights, meta)
        figures = []

        # --- Le spectre du modèle : scree plot + variance cumulée (complète).
        figures.append((
            "Spectre — scree plot et variance cumulée",
            captured(
                plot_spectrum, self._spectrum_pca_of(weights, meta),
                thresholds=(0.90, 0.95), n_show=min(50, IMAGE_DIM),
                suptitle="Spectre du modèle (variances issues de l'entraînement)",
            ),
        ))

        # --- Les composantes en heatmaps ± (rouge ajoute de l'encre, bleu en retire).
        if k >= 2:  # la fonction indexe axes[i] : il lui faut au moins 2 panneaux
            figures.append((
                "Composantes en heatmaps ±",
                captured(plot_component_heatmaps, pca, n=min(k, 8)),
            ))

        # --- L'effet du centrage : deux refits (par pixel / global) sur X.
        pca_feature = fit_pca(X, n_components=None, center="feature")
        pca_global = fit_pca(X, n_components=None, center="global")
        figures.append((
            "Centrage — spectres comparés",
            captured(
                plot_spectrum_comparison, [pca_feature, pca_global],
                ["par pixel", "global"],
                suptitle="Le centrage global gaspille PC1 sur l'image moyenne",
            ),
        ))
        figures.append((
            "Centrage — PC1 vs image moyenne",
            captured(plot_pc1_vs_mean, X),
        ))

        # --- Reconstructions à k croissant : le détail revient composante à composante.
        counts = np.bincount(y, minlength=len(class_names))
        n_per_class = 2 if counts.min() >= 2 else (1 if counts.min() >= 1 else 0)
        if n_per_class:
            k_max = min(IMAGE_DIM, max(2, n - 1))
            ks_grid = sorted({
                kk for kk in (max(1, k // 10), max(2, k // 2), k, min(k * 4, k_max))
                if 1 <= kk <= k_max
            })
            figures.append((
                "Reconstructions à k croissant",
                captured(
                    plot_reconstruction_grid, X, y, class_names, ks_grid,
                    n_per_class=n_per_class, seed=0, center=center,
                    suptitle="Original (haut) vs reconstruction à k croissant",
                ),
            ))

        # --- MSE normalisée : la fraction de variance inexpliquée selon k. Déduite
        # du spectre du refit « par pixel » ci-dessus — aucun fit supplémentaire.
        rank = min(IMAGE_DIM, n)
        ks_curve = [kk for kk in (1, 2, 5, 10, 20, 50, 100, 200, 400, IMAGE_DIM)
                    if kk <= rank]
        eig = pca_feature.explained_variance_
        mses = [float(eig[kk:].sum() * (n - 1) / n / IMAGE_DIM) for kk in ks_curve]
        total_var = float(np.mean(np.var(X, axis=0)))
        if total_var > 0:
            figures.append((
                "Compression normalisée — variance inexpliquée vs k",
                captured(
                    plot_normalized_comparison, ks_curve,
                    [("train (déduit du spectre)", mses, total_var, "steelblue")],
                    suptitle="MSE normalisée : la part de variance que k composantes laissent",
                ),
            ))

        # --- Génération : une gaussienne latente par classe, échantillonnée puis
        # décodée. np.cov exige au moins 2 images par classe dans la tranche.
        if len(class_names) >= 2 and counts.min() >= 2:
            per_class = fit_latent_gaussian(pca, X, y=y)
            figures.append((
                "Génération — tirages par classe",
                captured(plot_generated_grid, pca, per_class, class_names,
                         suptitle="Images générées (une gaussienne latente par classe)"),
            ))
            figures.append((
                "Génération — moyenne réelle vs moyenne latente",
                captured(plot_real_vs_generated, pca, X, y, per_class, class_names),
            ))
            figures.append((
                "Génération — interpolation latente",
                captured(plot_interpolation, pca, X, y, 0, 1, class_names),
            ))

        return figures

    def describe_rows(self, meta):
        return [
            ("k — composantes", f"{meta['k']}"),
            ("Centrage", "par pixel" if meta.get("center", "feature") == "feature"
             else "global"),
            ("Variance expliquée", f"{meta.get('explained_variance_kept', 0) * 100:.1f} %"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("Inertie finale", f"{meta['inertia']:.1f}"),
        ]
