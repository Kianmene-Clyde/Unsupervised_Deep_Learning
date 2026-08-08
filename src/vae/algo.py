"""Adaptateur VAE — ce que src/vae présente à l'app.

Même façade que l'autoencodeur (src/autoencoder/algo.py), dont le VAE hérite la
structure encodeur–décodeur et la plupart des vues. Trois choses le distinguent,
et elles vivent ici :

- l'encodeur prédit une DISTRIBUTION (z_mean, z_log_var) ; le code d'une image,
  pour compresser et projeter, est z_mean — le point latent déterministe (le
  reste, le tirage z = μ + σ·ε, ne sert qu'à l'entraînement, dans model.py) ;
- la loss ajoute la KL divergence vers N(0, 1), pondérée par kl_weight : c'est
  ce que raconte codec_note / latent_note, et ce que la vue « courbes de loss »
  décompose ;
- le latent étant tiré vers N(0, 1), la GÉNÉRATION échantillonne z ~ N(0, I)
  directement — là où l'autoencodeur simple devait passer par la gaussienne
  empirique de ses codes. C'est la vraie récompense du VAE, et sa vue signature.

Les poids voyagent comme pour l'autoencodeur : un vecteur float32 (tous les
tenseurs des deux modèles), l'architecture dans les métadonnées (voir model.py).
"""

import numpy as np

from .utils import codec
from .utils.model import (
    IMAGE_DIM,
    decode_batch,
    encode_batch,
    flat_weights,
    remember,
    train_vae,
)
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils import visualization as viz

from algo_base import Algo, Param

# Presets d'empilement : Param ne décrit que des scalaires ou des choix — une
# liste libre de tailles de couches n'a pas de widget. Repris de l'autoencodeur.
HIDDEN_PRESETS = {
    "128-32": (128, 32),
    "256-64": (256, 64),
    "512-128-32": (512, 128, 32),
    "64": (64,),
}
CONV_PRESETS = {
    "32-64": (32, 64),
    "16-32": (16, 32),
    "64-128": (64, 128),
}
MAX_LATENT = 128

# Traversée du décodeur : pas en unités d'écart-type autour du latent moyen.
TRAVERSAL_STEPS = (-2.0, -1.0, 0.0, 1.0, 2.0)
MANIFOLD_GRID = 15
N_GENERATED = 16


class VAE(Algo):
    key = "vae"
    label = "VAE"
    title = "Entraîner un VAE (autoencodeur variationnel — latent régularisé)"

    dict_label = "Décodeur"
    code_label = "vecteur latent"
    k_label = "Dimension latente"

    params = [
        Param(
            "architecture", "Architecture", "choice", "dense",
            choices=[("Dense — couches fully connected", "dense"),
                     ("Convolutionnelle — Conv2D / Conv2DTranspose", "conv")],
            info="Dense aplatit l'image ; conv préserve sa structure 2D.",
        ),
        Param("latent_dim", "Dimension latente — taille du code", "int", 2,
              minimum=1, maximum=MAX_LATENT,
              info="Le goulot d'étranglement : chaque image devient ce nombre "
                   "de flottants. À 2, l'espace latent se dessine en entier."),
        Param(
            "kl_weight", "Poids de la KL", "choice", 1e-3,
            choices=[("0 — autoencodeur simple (aucune régularisation)", 0.0),
                     ("1e-4", 1e-4), ("1e-3", 1e-3), ("1e-2", 1e-2),
                     ("1 — VAE « pur »", 1.0)],
            info="Force qui tire le latent vers N(0, 1). Plus grand = latent "
                 "plus lisse et générable, mais reconstruction moins fidèle.",
        ),
        Param(
            "hidden_dims", "Couches cachées (dense)", "choice", "128-32",
            choices=[("128 → 32", "128-32"), ("256 → 64", "256-64"),
                     ("512 → 128 → 32", "512-128-32"), ("64", "64")],
            info="L'encodeur ; le décodeur est son miroir. Ignoré en conv.",
        ),
        Param(
            "conv_filters", "Filtres (conv)", "choice", "32-64",
            choices=[("32 → 64", "32-64"), ("16 → 32", "16-32"),
                     ("64 → 128", "64-128")],
            info="Stride 2 par couche : 28 → 14 → 7. Ignoré en dense.",
        ),
        Param(
            "hidden_activation", "Activation cachée", "choice", "relu",
            choices=[("ReLU", "relu"), ("ELU", "elu"), ("tanh", "tanh"),
                     ("Leaky ReLU (pente 0,2)", "leaky_relu")],
            info="La non-linéarité des couches — ce qui distingue le réseau "
                 "d'une PCA.",
        ),
        Param(
            "latent_activation", "Activation latente (z_mean)", "choice", "linear",
            choices=[("Linéaire — aucune (défaut)", "linear"),
                     ("tanh — μ borné [-1, 1]", "tanh"),
                     ("Sigmoïde — μ borné [0, 1]", "sigmoid"),
                     ("ReLU", "relu"),
                     ("Leaky ReLU (pente 0,2)", "leaky_relu")],
            info="L'activation SUR z_mean (le code). z_log_var reste toujours "
                 "linéaire : une log-variance doit rester libre sur ℝ pour la KL.",
        ),
        Param(
            "output_activation", "Activation de sortie", "choice", "sigmoid",
            choices=[("Sigmoïde — pixels dans [0, 1]", "sigmoid"),
                     ("Linéaire", "linear")],
            info="Sigmoïde borne la reconstruction comme les pixels d'entrée.",
        ),
        Param(
            "optimizer", "Optimiseur", "choice", "adam",
            choices=[("Adam", "adam"), ("SGD", "sgd"), ("RMSprop", "rmsprop")],
            info="Adam converge vite sans réglage ; SGD montre la descente brute.",
        ),
        Param(
            "learning_rate", "Pas d'apprentissage", "choice", 1e-3,
            choices=[("1e-2", 1e-2), ("1e-3", 1e-3), ("1e-4", 1e-4)],
            info="Grand = rapide mais peut diverger ; petit = lent mais stable.",
        ),
        Param(
            "batch_size", "Taille de batch", "choice", 256,
            choices=[("64", 64), ("128", 128), ("256", 256), ("512", 512)],
            info="Images par pas de gradient.",
        ),
        Param("epochs", "Époques", "int", 20, minimum=1,
              info="Passages sur tout le jeu. La courbe de loss dit quand "
                   "ça suffit."),
        Param("seed", "Seed", "int", 42, minimum=0,
              info="Initialisation des poids + ordre de présentation + tirage ε."),
    ]

    # ------------------------------------------------------------ Entraînement

    @staticmethod
    def _config_of(p):
        """Le dict de config d'utils/model.py, depuis les hyperparamètres de
        l'app OU les métadonnées d'un modèle sauvegardé (mêmes clés)."""
        hidden = p.get("hidden_dims", "128-32")
        conv = p.get("conv_filters", "32-64")
        filters = CONV_PRESETS[conv] if isinstance(conv, str) else tuple(conv)
        return {
            "architecture": str(p.get("architecture", "dense")),
            "latent_dim": int(p["latent_dim"]),
            "kl_weight": float(p.get("kl_weight", 1e-3)),
            "hidden_dims": HIDDEN_PRESETS[hidden] if isinstance(hidden, str)
                           else tuple(hidden),
            "conv_filters": filters,
            "conv_strides": tuple(2 for _ in filters),
            "kernel_size": 3,
            "hidden_activation": str(p.get("hidden_activation", "relu")),
            "latent_activation": str(p.get("latent_activation", "linear")),
            "output_activation": str(p.get("output_activation", "sigmoid")),
            "optimizer": str(p.get("optimizer", "adam")),
            "learning_rate": float(p.get("learning_rate", 1e-3)),
            "loss_name": "mse",
            "batch_size": int(p.get("batch_size", 256)),
            "epochs": int(p.get("epochs", 20)),
            "seed": int(p.get("seed", 42)),
        }

    def k_of_params(self, p):
        return int(p.get("latent_dim") or 0)

    def check(self, p, n_samples):
        latent = int(p.get("latent_dim") or 0)
        if not 1 <= latent <= MAX_LATENT:
            raise ValueError(f"Dimension latente : entre 1 et {MAX_LATENT}.")
        if float(p.get("kl_weight", 0.0)) < 0:
            raise ValueError("Le poids de la KL doit être positif ou nul.")
        if int(p.get("epochs") or 0) < 1:
            raise ValueError("Il faut au moins 1 époque.")
        if n_samples < 2:
            raise ValueError(
                "Il faut au moins 2 images : les statistiques latentes "
                "estiment une dispersion."
            )
        if latent >= n_samples:
            raise ValueError(
                f"latent_dim={latent} pour {n_samples} images : le goulot est plus "
                f"large que le jeu, le réseau apprendrait par cœur. Baisse la "
                f"dimension latente ou augmente les images."
            )

    def auto_name(self, ds_key, p):
        return (f"{ds_key}_z{int(p.get('latent_dim') or 0)}"
                f"_{p.get('architecture', 'dense')}"
                f"_kl{float(p.get('kl_weight', 1e-3)):g}"
                f"_e{int(p.get('epochs') or 0)}_n{int(p.get('n_samples') or 0)}")

    def train(self, X, p):
        cfg = self._config_of(p)
        X = np.asarray(X, dtype=np.float32)
        n = len(X)

        encoder, decoder, history = train_vae(X, cfg, progress=p.get("progress"))
        weights = flat_weights(encoder, decoder)
        remember(weights, cfg, encoder, decoder)

        # Inertie exacte — mesurée comme le codec la produira (z_mean → décodeur,
        # clip [0, 1] compris), plutôt que la loss de la dernière époque (moyenne
        # mouvante PENDANT la descente, donc toujours un peu au-dessus).
        Z = encode_batch(X, weights)  # z_mean
        X_hat = decode_batch(Z, weights)
        mse = float(np.mean((X - X_hat) ** 2))
        inertia = mse * n * IMAGE_DIM

        # Les statistiques latentes voyagent dans les métadonnées : la traversée
        # du décodeur (onglet Dictionnaire) n'a pas les données sous la main.
        z_mean = Z.mean(axis=0)
        z_std = Z.std(axis=0)

        return weights, {
            "k": cfg["latent_dim"],
            **{key: (list(v) if isinstance(v, tuple) else v)
               for key, v in cfg.items()},
            "inertia": float(inertia),
            # La loss de reconstruction « mse » de Keras est déjà par pixel : la
            # remettre à l'échelle inertie (× n × 784) donne à l'app la même unité
            # que K-means/SOM — model_rows refait la division pour afficher la MSE.
            "inertia_history": [float(l) * n * IMAGE_DIM
                                for l in history["reconstruction_loss"]],
            # Propres au VAE : gardés pour la vue « courbes de loss ».
            "kl_history": [float(l) for l in history["kl_loss"]],
            "total_history": [float(l) for l in history["total_loss"]],
            "latent_mean": [float(v) for v in z_mean],
            "latent_std": [float(v) for v in z_std],
        }

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        """La dimension latente dominante de chaque image : argmax |zᵢ − z̄ᵢ|.

        Le VAE n'a pas de clusters ; c'est son équivalent d'une assignation
        discrète, utilisé par la vue « distribution des classes ». Le centrage
        compte : sans lui, une dimension à moyenne élevée absorberait tout.
        """
        Z = encode_batch(X, weights)
        centered = Z - Z.mean(axis=0, keepdims=True)
        return np.argmax(np.abs(np.atleast_2d(centered)), axis=1)

    def encode(self, image, weights):
        return codec.encode(image, weights)

    def decode(self, code, weights):
        return codec.decode(code, weights)

    # --------------------------------------------------------------- Registre

    def list_models(self):
        return list_models()

    def load(self, name):
        weights, meta = load_model(name)
        # C'est ICI que le cache apprend l'architecture de ce vecteur de poids :
        # encode()/decode() ne reçoivent que les poids et ne pourraient pas la
        # retrouver seuls. Chaque vue de l'app charge avant d'encoder.
        remember(np.asarray(weights, dtype=np.float32), self._config_of(meta))
        return weights, meta

    def save(self, name, weights, meta):
        return save_model(name, weights, meta)

    def delete(self, name):
        delete_model(name)

    # ------------------------------------------------------------------ Récits

    def codec_note(self, meta, ds_key, n_images, seed, stats):
        k = int(meta["k"])
        kl_weight = float(meta.get("kl_weight", 1e-3))
        bits = k * 32  # latent_dim flottants float32, contre UN entier pour K-means/SOM
        ratio = IMAGE_DIM * 8 / bits
        mse = meta["inertia"] / (meta["n_samples"] * IMAGE_DIM)
        return (
            f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}**, "
            f"encodées par le **VAE** (graine {seed} — relance le tirage pour en "
            f"voir d'autres).\n\n"
            f"Comme l'autoencodeur, le code n'est **pas un entier** : chaque image "
            f"(784 pixels) est transmise comme **{k} flottant(s)** — mais ici c'est "
            f"**z_mean**, la moyenne de la distribution latente que l'encodeur "
            f"prédit (μ, σ). Soit {bits} bits en float32, un taux de compression de "
            f"**{ratio:.1f}:1** face aux {IMAGE_DIM * 8} bits de l'image brute (le "
            f"décodeur n'étant transmis qu'une fois).\n\n"
            f"Ce qui fait le VAE, c'est la **KL divergence** (poids {kl_weight:g}) "
            f"ajoutée à la loss : elle pousse l'encodeur à ranger les codes autour "
            f"de **N(0, 1)**. On perd un peu de fidélité de reconstruction, on gagne "
            f"un espace latent **lisse et générable** — voir l'onglet « Vues de "
            f"l'algo ».\n\n"
            f"Le code étant continu, chaque image reçoit le sien : "
            f"{stats['Train']}/{n_images} codes distincts en train et "
            f"{stats['Test']}/{n_images} en test — c'est attendu.\n\n"
            f"Le prix de la compression est la **MSE : {mse:.4f}** par pixel "
            f"(mesurée sur le train à l'entraînement)."
        )

    def latent_note(self, meta):
        k = int(meta["k"])
        kl_weight = float(meta.get("kl_weight", 1e-3))
        return (
            f"L'espace latent du VAE est continu : ℝ^{k}, les z_mean de chaque "
            f"image. Les nuages montrent z₁ × z₂ (et z₁ × z₂ × z₃) — **les vraies "
            f"premières dimensions du code**, pas une projection d'illustration.\n\n"
            "✅ Train et test passent par **le même encodeur** : les deux figures "
            "partagent le même repère et se comparent directement.\n\n"
            f"⭐ Contrairement à l'autoencodeur simple, ce latent est **régularisé** "
            f"par la KL divergence (poids {kl_weight:g}) : le nuage est tiré vers "
            f"**N(0, 1)** — centré sur l'origine, sans trous béants. C'est ce qui "
            f"rend la génération par échantillonnage de N(0, I) possible (onglet "
            f"« Vues de l'algo »). Avec kl_weight = 0, on retombe exactement sur un "
            f"autoencodeur simple.\n\n"
            "**Distribution** : le réseau n'assigne pas de cluster ; chaque image "
            "est rangée sous sa **dimension dominante** (celle où |zᵢ − z̄ᵢ| est "
            "maximal). La carte classes × dimensions montre si certaines dimensions "
            "« appartiennent » à certaines classes."
        )

    # ------------------------------------------------------------------- Vues

    def plot_reconstructions(self, originals, reconstructions, top_labels,
                             bottom_labels, title):
        return viz.show_reconstructions(
            originals, reconstructions, top_labels=top_labels,
            bottom_labels=bottom_labels, title=title,
        )

    def plot_distribution(self, codes, y_true, class_names, k, title):
        return viz.plot_dominant_distribution(
            codes, y_true, class_names=class_names, k=k, title=title,
        )

    def plot_latent(self, X, weights, meta, y_true, class_names, y_label, title):
        Z = encode_batch(X, weights)
        centered = Z - Z.mean(axis=0, keepdims=True)
        labels = np.argmax(np.abs(np.atleast_2d(centered)), axis=1)
        return viz.plot_latent_panels(
            Z, y=y_true, class_names=class_names, labels=labels,
            title=title, y_label=y_label,
        )

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        """La traversée du décodeur — l'équivalent continu d'un dictionnaire.

        Pour chaque dimension latente, decode(z̄ + t·σᵢ·eᵢ) avec t de −2 à +2,
        les statistiques z̄, σ ayant été mesurées à l'entraînement. Sur un VAE
        régularisé, z̄ ≈ 0 et σ ≈ 1 : la traversée parcourt donc peu ou prou les
        quantiles de N(0, 1).
        """
        k = int(meta["k"])
        mean = np.asarray(meta["latent_mean"], dtype=np.float32)
        std = np.asarray(meta["latent_std"], dtype=np.float32)
        n_show = min(k, 8)

        zs = []
        for i in range(n_show):
            for t in TRAVERSAL_STEPS:
                z = mean.copy()
                z[i] += t * max(std[i], 1e-6)
                zs.append(z)
        images = decode_batch(np.stack(zs), weights).reshape(
            n_show, len(TRAVERSAL_STEPS), IMAGE_DIM
        )

        fig = viz.plot_traversal_grid(
            images, TRAVERSAL_STEPS, [f"z{i + 1}" for i in range(n_show)],
            suptitle="Traversée du décodeur — ce que chaque dimension latente encode",
        )
        note = (
            f"**Le dictionnaire du VAE n'est pas une liste d'images : c'est le "
            f"décodeur lui-même**, une fonction apprise de ℝ^{k} vers l'image. "
            f"Chaque ligne parcourt UNE dimension latente de −2σ à +2σ autour du "
            f"latent moyen z̄ (statistiques mesurées à l'entraînement), les autres "
            f"restant fixées : ce qui change le long d'une ligne est ce que cette "
            f"dimension encode. Le VAE étant régularisé vers N(0, 1), z̄ est proche "
            f"de l'origine et σ proche de 1 — la traversée balaie donc les valeurs "
            f"plausibles du prior."
        )
        if n_show < k:
            note += (
                f"\n\n*Seules les {n_show} premières des {k} dimensions sont "
                f"affichées.*"
            )
        return fig, note

    def plot_curve(self, history, n_samples):
        mses = np.asarray(history, dtype=np.float64) / (n_samples * IMAGE_DIM)
        fig = viz.plot_loss_curve(
            mses, suptitle="VAE — MSE de reconstruction par époque"
        )
        note = (
            f"La **MSE de reconstruction par pixel** au fil des {len(mses)} "
            f"époque(s) — la part reconstruction de la loss, moyennée sur chaque "
            f"époque. MSE finale : **{mses[-1]:.4f}**.\n\n"
            f"⚠️ Ce n'est PAS toute la loss du VAE : la descente minimise "
            f"**reconstruction + kl_weight · KL**. La reconstruction peut donc "
            f"stagner un peu au-dessus de ce qu'atteindrait un autoencodeur simple "
            f"— c'est la KL divergence qui « paie » pour un latent régularisé. "
            f"L'onglet « Vues de l'algo » décompose les trois courbes."
        )
        return fig, note

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Les vues signature du VAE. A besoin des IMAGES : tout part de Z = z_mean."""
        X = np.asarray(X, dtype=np.float32)
        k = int(meta["k"])
        kl_weight = float(meta.get("kl_weight", 1e-3))
        seed = int(meta.get("seed", 42))
        Z = encode_batch(X, weights)
        figures = []

        # --- Les trois courbes de loss (totale / reconstruction / KL) : la vue
        # qui montre l'arbitrage propre au VAE. Disponible dès qu'on a l'historique.
        recon = meta.get("inertia_history")
        kl = meta.get("kl_history")
        if recon and kl:
            recon_pp = [v / (int(meta["n_samples"]) * IMAGE_DIM) for v in recon]
            figures.append((
                "Courbes de loss — reconstruction vs KL",
                viz.plot_vae_losses(recon_pp, kl, kl_weight),
            ))

        # --- La grille de l'espace latent 2D. Uniquement à latent_dim=2 : c'est
        # là que le plan SE DESSINE en entier. Pour un VAE, on balaie la grille
        # sur les quantiles de N(0, 1) (le prior), pas sur les percentiles des
        # codes — le latent EST censé suivre ce prior.
        if k == 2:
            lo, hi = -3.0, 3.0
            grid_z = np.array(
                [[z1, z2]
                 for z2 in np.linspace(hi, lo, MANIFOLD_GRID)
                 for z1 in np.linspace(lo, hi, MANIFOLD_GRID)],
                dtype=np.float32,
            )
            canvas = viz.make_manifold_canvas(
                decode_batch(grid_z, weights), MANIFOLD_GRID
            )
            figures.append((
                "Espace latent 2D — grille N(0,1) décodée",
                viz.plot_manifold(canvas, (lo, hi), (lo, hi)),
            ))

        # --- Génération : LA récompense du VAE. Le latent étant régularisé vers
        # N(0, 1), on échantillonne z ~ N(0, I) DIRECTEMENT — pas besoin de la
        # gaussienne empirique dont l'autoencodeur simple avait besoin.
        rng = np.random.default_rng(seed)
        sampled = rng.standard_normal(size=(N_GENERATED, k)).astype(np.float32)
        titles = [
            "z=(" + ", ".join(f"{v:+.1f}" for v in z[:2])
            + (", …)" if k > 2 else ")")
            for z in sampled
        ]
        figures.append((
            "Génération — échantillonnage de N(0, I)",
            viz.plot_generated_grid(
                decode_batch(sampled, weights), titles,
                suptitle="Images générées — z ~ N(0, I) décodés par le VAE",
            ),
        ))

        return figures

    def describe_rows(self, meta):
        arch = meta.get("architecture", "dense")
        if arch == "conv":
            stack = " → ".join(str(f) for f in meta.get("conv_filters", []))
            stack_label = "Filtres conv"
        else:
            stack = " → ".join(str(d) for d in meta.get("hidden_dims", []))
            stack_label = "Couches cachées"
        return [
            ("Architecture", "convolutionnelle" if arch == "conv" else "dense"),
            ("Dimension latente", f"{meta['k']}"),
            ("Poids KL", f"{float(meta.get('kl_weight', 1e-3)):g}"),
            (stack_label, stack or "—"),
            ("Activations", f"{meta.get('hidden_activation', 'relu')} (cachée) / "
                            f"{meta.get('latent_activation', 'linear')} (z_mean) / "
                            f"{meta.get('output_activation', 'sigmoid')} (sortie)"),
            ("Optimiseur", f"{meta.get('optimizer', 'adam')} "
                           f"(lr {meta.get('learning_rate', 1e-3):g})"),
            ("Entraînement", f"{meta.get('epochs', '?')} époques · "
                             f"batch {meta.get('batch_size', '?')}"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("Inertie finale", f"{meta['inertia']:.1f}"),
        ]
