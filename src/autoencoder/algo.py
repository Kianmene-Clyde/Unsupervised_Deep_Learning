"""Adaptateur Autoencodeur — ce que src/autoencoder présente à l'app.

Comme pour K-means, Kohonen et PCA : la matière vient du travail notebook du
dossier (autoencodeur.py, NON modifié — et jamais importé : il s'entraîne à
l'import). utils/model.py en reprend les architectures dense et conv ; ce
fichier ne fait que la traduction vers l'interface commune (algo_base) :

- Les poids de l'app sont UN tableau numpy : tous les tenseurs Keras des deux
  modèles (encodeur + décodeur), aplatis en un vecteur float32. L'architecture
  voyage dans les métadonnées, qui permettent de reconstruire les modèles et
  de redécouper le vecteur — voir utils/model.py.
- Le code d'une image est un VECTEUR de latent_dim flottants (la sortie de
  l'encodeur), pas un entier : utils/codec.py l'enveloppe (hashable,
  affichable), exactement comme la PCA. codec_note / latent_note remplacent le
  récit « code entier » des autres algos, qui mentirait ici.
- Les statistiques latentes (moyenne, écart-type par dimension) sont mesurées
  à l'entraînement et stockées dans les métadonnées : la traversée du décodeur
  (l'onglet Dictionnaire) peut ainsi se tracer sans les données.
- Les vues signature du notebook sont toutes reprises : grille de l'espace
  latent 2D, génération par gaussienne empirique.
"""

import numpy as np

from .utils import codec
from .utils.model import (
    IMAGE_DIM,
    decode_batch,
    encode_batch,
    flat_weights,
    remember,
    train_autoencoder,
)
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils import visualization as viz

from algo_base import Algo, Param

# Presets d'empilement : Param ne décrit que des scalaires ou des choix — une
# liste libre de tailles de couches n'a pas de widget. Des presets suffisent
# pour explorer, et ils sont tous valides par construction (strides 2·2 = 4
# divise 28, la contrainte conv du notebook).
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


class Autoencoder(Algo):
    key = "autoencoder"
    label = "Autoencodeur"
    title = "Entraîner un autoencodeur (encodeur–décodeur neuronal)"

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
            "latent_activation", "Activation latente", "choice", "linear",
            choices=[("Linéaire — aucune (défaut)", "linear"),
                     ("tanh — code borné [-1, 1]", "tanh"),
                     ("Sigmoïde — code borné [0, 1]", "sigmoid"),
                     ("ReLU", "relu"),
                     ("Leaky ReLU (pente 0,2)", "leaky_relu")],
            info="L'activation SUR la couche latente (le goulot). Linéaire = le "
                 "code non contraint, comme une PCA ; tanh/sigmoïde le bornent.",
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
              info="Initialisation des poids + ordre de présentation."),
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
        if int(p.get("epochs") or 0) < 1:
            raise ValueError("Il faut au moins 1 époque.")
        if n_samples < 2:
            raise ValueError(
                "Il faut au moins 2 images : la gaussienne latente (génération) "
                "estime une covariance."
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
                f"_e{int(p.get('epochs') or 0)}_n{int(p.get('n_samples') or 0)}")

    def train(self, X, p):
        cfg = self._config_of(p)
        X = np.asarray(X, dtype=np.float32)
        n = len(X)

        encoder, decoder, losses = train_autoencoder(X, cfg, progress=p.get("progress"))
        weights = flat_weights(encoder, decoder)
        remember(weights, cfg, encoder, decoder)

        # Inertie exacte — mesurée comme le codec la produira, clip [0, 1]
        # compris, plutôt que la loss de la dernière époque (moyenne mouvante
        # PENDANT la descente, donc toujours un peu au-dessus).
        Z = encode_batch(X, weights)
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
            # La loss « mse » de Keras est déjà par pixel : la remettre à
            # l'échelle inertie (× n × 784) donne à l'app la même unité que
            # K-means/SOM — model_rows refait la division pour afficher la MSE.
            "inertia_history": [float(l) * n * IMAGE_DIM for l in losses],
            "latent_mean": [float(v) for v in z_mean],
            "latent_std": [float(v) for v in z_std],
        }

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        """La dimension latente dominante de chaque image : argmax |zᵢ − z̄ᵢ|.

        L'autoencodeur n'a pas de clusters ; c'est son équivalent d'une
        assignation discrète, utilisé par la vue « distribution des classes ».
        Le centrage compte : contrairement à la PCA, les zᵢ ne sont pas centrés
        par construction — sans lui, une dimension à moyenne élevée absorberait
        toutes les images.
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
        bits = k * 32  # latent_dim flottants float32, contre UN entier pour K-means/SOM
        ratio = IMAGE_DIM * 8 / bits
        mse = meta["inertia"] / (meta["n_samples"] * IMAGE_DIM)
        return (
            f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}**, "
            f"encodées par **l'autoencodeur** (graine {seed} — relance le tirage pour "
            f"en voir d'autres).\n\n"
            f"Contrairement à K-means ou au SOM, le code n'est **pas un entier** : "
            f"chaque image (784 pixels) est transmise comme **{k} flottant(s)** — la "
            f"sortie de l'encodeur — soit {bits} bits en float32, un taux de "
            f"compression de **{ratio:.1f}:1** face aux {IMAGE_DIM * 8} bits de "
            f"l'image brute (le décodeur n'étant transmis qu'une fois).\n\n"
            f"Le décodeur reconstruit `image ≈ décodeur(z)` — une fonction "
            f"**non linéaire** apprise, là où la PCA recombine linéairement ses "
            f"composantes. Le code étant continu, chaque image reçoit le sien et "
            f"aucune colonne ne se répète — d'où les {stats['Train']}/{n_images} "
            f"codes distincts en train et {stats['Test']}/{n_images} en test : "
            f"c'est attendu, pas un bug.\n\n"
            f"Le prix de la compression est la **MSE : {mse:.4f}** par pixel "
            f"(mesurée sur le train à l'entraînement) — ce que le goulot de "
            f"{k} flottant(s) n'a pas laissé passer."
        )

    def latent_note(self, meta):
        k = int(meta["k"])
        return (
            f"Ici, l'espace latent **est** continu : ℝ^{k}, la sortie brute de "
            f"l'encodeur pour chaque image. Les nuages montrent z₁ × z₂ (et "
            f"z₁ × z₂ × z₃) — **les vraies premières dimensions du code**, pas une "
            f"projection d'illustration.\n\n"
            "✅ Train et test passent par **le même encodeur** : les deux figures "
            "partagent le même repère et se comparent directement, positions "
            "comprises.\n\n"
            "⚠️ Contrairement à la PCA, les axes ne sont **pas ordonnés par "
            "variance** ni orthogonaux : le réseau organise son latent comme "
            "l'entraînement l'y a mené. Un autoencodeur simple n'est pas non plus "
            "régularisé (pas un VAE) : le nuage peut être troué ou étiré.\n\n"
            "**Distribution** : le réseau n'assigne pas de cluster ; chaque image "
            "est rangée sous sa **dimension dominante** (celle où |zᵢ − z̄ᵢ| est "
            "maximal). La carte classes × dimensions montre si certaines dimensions "
            "« appartiennent » à certaines classes — la pureté se lit comme pour "
            "des clusters."
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

        K-means expose k images types ; le « dictionnaire » de l'autoencodeur
        est une FONCTION, le décodeur. On la montre en la parcourant : pour
        chaque dimension latente, decode(z̄ + t·σᵢ·eᵢ) avec t de −2 à +2 —
        les statistiques z̄, σ ayant été mesurées à l'entraînement.
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
            f"**Le dictionnaire de l'autoencodeur n'est pas une liste d'images : "
            f"c'est le décodeur lui-même**, une fonction apprise de ℝ^{k} vers "
            f"l'image. Chaque ligne parcourt UNE dimension latente de −2σ à +2σ "
            f"autour du latent moyen z̄ (statistiques mesurées à l'entraînement), "
            f"les autres restant fixées : ce qui change le long d'une ligne est ce "
            f"que cette dimension encode — inclinaison, épaisseur, forme… La "
            f"colonne z̄ est identique partout : c'est l'image du latent moyen."
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
            mses, suptitle="Autoencodeur — MSE de reconstruction par époque"
        )
        note = (
            f"La **MSE de reconstruction par pixel** au fil des {len(mses)} "
            f"époque(s) — la loss que la descente de gradient minimise, moyennée "
            f"sur chaque époque. MSE finale : **{mses[-1]:.4f}**.\n\n"
            f"Une courbe qui plonge puis s'aplatit a convergé : des époques en "
            f"plus n'apporteraient rien. Une courbe encore en pente à la dernière "
            f"époque dit l'inverse — réentraîne avec plus d'époques. Des "
            f"oscillations trahissent un pas d'apprentissage trop grand."
        )
        return fig, note

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Les vues du notebook qui n'ont pas d'onglet dédié, plus deux propres
        au réseau. A besoin des IMAGES : tout part de Z = encoder(X)."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_true).astype(int).ravel()
        k = int(meta["k"])
        seed = int(meta.get("seed", 42))
        Z = encode_batch(X, weights)
        figures = []

        # --- La grille de l'espace latent 2D — la vue signature du notebook.
        # Uniquement à latent_dim=2 : c'est là que le plan SE DESSINE en entier
        # (le notebook faisait la même réserve).
        if k == 2:
            z1_min, z1_max = np.percentile(Z[:, 0], [1, 99])
            z2_min, z2_max = np.percentile(Z[:, 1], [1, 99])
            grid_z = np.array(
                [[z1, z2]
                 for z2 in np.linspace(z2_max, z2_min, MANIFOLD_GRID)
                 for z1 in np.linspace(z1_min, z1_max, MANIFOLD_GRID)],
                dtype=np.float32,
            )
            canvas = viz.make_manifold_canvas(
                decode_batch(grid_z, weights), MANIFOLD_GRID
            )
            figures.append((
                "Espace latent 2D — grille décodée",
                viz.plot_manifold(canvas, (z1_min, z1_max), (z2_min, z2_max)),
            ))

        # --- Génération : gaussienne EMPIRIQUE des vrais codes (le notebook) —
        # un autoencodeur simple n'est pas régularisé, N(0, I) tomberait à côté.
        if len(X) >= 2:
            latent_mean = Z.mean(axis=0)
            latent_cov = np.atleast_2d(np.cov(Z, rowvar=False)) + 1e-6 * np.eye(k)
            rng = np.random.default_rng(seed)
            sampled = rng.multivariate_normal(
                mean=latent_mean, cov=latent_cov, size=N_GENERATED
            ).astype(np.float32)
            titles = [
                "z=(" + ", ".join(f"{v:+.1f}" for v in z[:2])
                + (", …)" if k > 2 else ")")
                for z in sampled
            ]
            figures.append((
                "Génération — échantillonnage du latent",
                viz.plot_generated_grid(decode_batch(sampled, weights), titles),
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
            (stack_label, stack or "—"),
            ("Activations", f"{meta.get('hidden_activation', 'relu')} (cachée) / "
                            f"{meta.get('latent_activation', 'linear')} (latente) / "
                            f"{meta.get('output_activation', 'sigmoid')} (sortie)"),
            ("Optimiseur", f"{meta.get('optimizer', 'adam')} "
                           f"(lr {meta.get('learning_rate', 1e-3):g})"),
            ("Entraînement", f"{meta.get('epochs', '?')} époques · "
                             f"batch {meta.get('batch_size', '?')}"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("Inertie finale", f"{meta['inertia']:.1f}"),
        ]
