"""Adaptateur GAN — ce que src/gan présente à l'app.

Même façade que l'autoencodeur et le VAE, mais l'objet est inversé : là où eux
apprennent à COMPRIMER (et génèrent en prime), le GAN n'apprend QUE la
génération — deux réseaux NumPy écrits from scratch (voir utils/model.py) qui
s'affrontent jusqu'à ce que le générateur produise des images que le
discriminateur ne sait plus distinguer des vraies.

Ce que ça change pour les 3 tâches du projet :

- GÉNÉRATION : la vue signature. z ~ N(0, I) → G(z), directement — le GAN est
  entraîné exactement pour ça, sans le flou de moyenne des reconstructions MSE
  d'un autoencodeur (une MSE minimisée pousse vers la moyenne des possibles,
  l'adversaire de D pousse vers un exemplaire NET) ;
- COMPRESSION : le GAN n'a PAS d'encodeur — encoder demande d'INVERSER G, ici
  approximé par plus proche voisin dans une banque de générations (voir
  utils/codec.py). La MSE élevée qui en résulte n'est pas un bug d'affichage :
  c'est la démonstration chiffrée qu'un GAN est un mauvais codec ;
- PROJECTION : le nuage des z inversés reste lisible (le prior est N(0, I)),
  mais il projette l'image sur ce que G SAIT dessiner, pas sur ce que l'image
  contient — à lire avec cette réserve.

Les poids voyagent comme pour l'autoencodeur : un vecteur float32 (générateur
puis discriminateur), l'architecture dans les métadonnées.
"""

import matplotlib.pyplot as plt
import numpy as np

from .utils import codec
from .utils.model import (
    IMAGE_DIM,
    decode_batch,
    encode_batch,
    flat_weights,
    generate_batch,
    remember,
    train_gan,
)
from .utils.registry import delete_model, list_models, load_model, save_model
from .utils.registry import model_path as registry_model_path
from .utils import visualization as viz

from algo_base import Algo, Param

# Presets d'empilement : Param ne décrit que des scalaires ou des choix — une
# liste libre de tailles de couches n'a pas de widget. Repris de l'autoencodeur.
G_PRESETS = {
    "256-512": (256, 512),
    "128-256": (128, 256),
    "256-512-1024": (256, 512, 1024),
}
D_PRESETS = {
    "512-256": (512, 256),
    "256-128": (256, 128),
    "1024-512-256": (1024, 512, 256),
}
MAX_LATENT = 128

N_GENERATED = 16
INTERP_STEPS = 8
INTERP_ROWS = 3


class GAN(Algo):
    key = "gan"
    label = "GAN"
    title = "Entraîner un GAN (deux réseaux adverses — générateur contre discriminateur)"

    dict_label = "Générateur"
    code_label = "vecteur latent (inversé)"
    k_label = "Dimension latente"

    params = [
        Param("latent_dim", "Dimension latente — taille du bruit z", "int", 64,
              minimum=1, maximum=MAX_LATENT,
              info="L'entrée du générateur : chaque image générée naît d'un "
                   "tirage z ~ N(0, I) de cette dimension."),
        Param(
            "g_hidden", "Couches du générateur", "choice", "256-512",
            choices=[("256 → 512", "256-512"), ("128 → 256", "128-256"),
                     ("256 → 512 → 1024", "256-512-1024")],
            info="De z vers l'image : les couches s'élargissent jusqu'aux "
                 "784 pixels.",
        ),
        Param(
            "d_hidden", "Couches du discriminateur", "choice", "512-256",
            choices=[("512 → 256", "512-256"), ("256 → 128", "256-128"),
                     ("1024 → 512 → 256", "1024-512-256")],
            info="De l'image vers un verdict : les couches se resserrent "
                 "jusqu'à un logit vrai/généré.",
        ),
        Param(
            "learning_rate", "Pas d'apprentissage", "choice", 2e-4,
            choices=[("2e-4 (DCGAN)", 2e-4), ("1e-3", 1e-3), ("1e-4", 1e-4)],
            info="2e-4 avec Adam β₁=0,5 : le réglage DCGAN, le point de départ "
                 "sûr d'un entraînement adversarial.",
        ),
        Param(
            "label_smoothing", "Étiquette des vraies images", "choice", 0.9,
            choices=[("0,9 — lissage (recommandé)", 0.9),
                     ("1,0 — sans lissage", 1.0)],
            info="À 0,9, D ne peut jamais être sûr à 100 % : il continue "
                 "d'envoyer du gradient utile à G.",
        ),
        Param(
            "batch_size", "Taille de batch", "choice", 128,
            choices=[("64", 64), ("128", 128), ("256", 256)],
            info="Images (vraies et générées) par pas de gradient de chaque "
                 "réseau.",
        ),
        Param("epochs", "Époques", "int", 40, minimum=1,
              info="Passages sur tout le jeu. Les courbes du duel disent si "
                   "l'équilibre tient."),
        Param("seed", "Seed", "int", 42, minimum=0,
              info="Initialisation des poids + tirages z + ordre des batches."),
    ]

    # ------------------------------------------------------------ Entraînement

    @staticmethod
    def _config_of(p):
        """Le dict de config d'utils/model.py, depuis les hyperparamètres de
        l'app OU les métadonnées d'un modèle sauvegardé (mêmes clés)."""
        g = p.get("g_hidden", "256-512")
        d = p.get("d_hidden", "512-256")
        return {
            "latent_dim": int(p["latent_dim"]),
            "g_hidden": G_PRESETS[g] if isinstance(g, str) else tuple(g),
            "d_hidden": D_PRESETS[d] if isinstance(d, str) else tuple(d),
            "learning_rate": float(p.get("learning_rate", 2e-4)),
            "label_smoothing": float(p.get("label_smoothing", 0.9)),
            "batch_size": int(p.get("batch_size", 128)),
            "epochs": int(p.get("epochs", 40)),
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
        if n_samples < int(p.get("batch_size", 128)):
            raise ValueError(
                "Il faut au moins un batch d'images : le duel G/D n'a pas de "
                "sens sur une poignée d'exemples."
            )

    def auto_name(self, ds_key, p):
        return (f"{ds_key}_z{int(p.get('latent_dim') or 0)}"
                f"_g{p.get('g_hidden', '256-512')}_d{p.get('d_hidden', '512-256')}"
                f"_e{int(p.get('epochs') or 0)}_n{int(p.get('n_samples') or 0)}")

    def train(self, X, p):
        cfg = self._config_of(p)
        X = np.asarray(X, dtype=np.float32)
        n = len(X)

        g_layers, d_layers, history, snapshots = train_gan(
            X, cfg, progress=p.get("progress")
        )
        weights = flat_weights(g_layers, d_layers)
        remember(weights, cfg, g_layers, d_layers)

        # « Inertie » : la MSE de l'inversion par banque, mesurée comme le codec
        # la produira (plus proche génération, voir utils/codec.py). Elle est
        # VOLONTAIREMENT élevée — c'est le chiffre qui prouve que le GAN n'est
        # pas un codec, comparable trait pour trait à celui des autres algos.
        sample = X[:min(n, 2048)]
        X_hat = decode_batch(encode_batch(sample, weights), weights)
        mse = float(np.mean((sample - X_hat) ** 2))
        inertia = mse * n * IMAGE_DIM

        return weights, {
            "k": cfg["latent_dim"],
            **{key: (list(v) if isinstance(v, tuple) else v)
               for key, v in cfg.items()},
            "inertia": float(inertia),
            # La courbe « standard » de l'app : la loss du générateur. Les
            # courbes complètes du duel vivent dans g/d_history + accuracies.
            "inertia_history": [float(v) for v in history["g_loss"]],
            "g_history": [float(v) for v in history["g_loss"]],
            "d_history": [float(v) for v in history["d_loss"]],
            "d_real_acc": [float(v) for v in history["d_real_acc"]],
            "d_fake_acc": [float(v) for v in history["d_fake_acc"]],
            # Instantanés de maturation : les mêmes 16 z décodés à ~5 époques.
            # Quantifiés sur 0-255 (uint8) pour rester légers dans le JSON.
            "snapshots": [
                {"epoch": int(e),
                 "images": np.round(np.asarray(imgs) * 255).astype(int).tolist()}
                for e, imgs in snapshots
            ],
        }

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        """La dimension latente dominante du z inversé : argmax |zᵢ − z̄ᵢ|.

        Même convention que l'autoencodeur et le VAE — le GAN n'a pas de
        clusters, c'est son équivalent d'une assignation discrète.
        """
        Z = encode_batch(X, weights)
        centered = Z - Z.mean(axis=0, keepdims=True)
        return np.argmax(np.abs(np.atleast_2d(centered)), axis=1)

    def encode(self, image, weights):
        return codec.encode(image, weights)

    def decode(self, code, weights):
        return codec.decode(code, weights)

    # --------------------------------------------------------------- Registre

    def model_path(self, name):
        return registry_model_path(name)

    def list_models(self):
        return list_models()

    def load(self, name):
        weights, meta = load_model(name)
        # C'est ICI que le cache apprend l'architecture de ce vecteur de poids :
        # generate/encode/decode ne reçoivent que les poids.
        remember(np.asarray(weights, dtype=np.float32), self._config_of(meta))
        return weights, meta

    def save(self, name, weights, meta):
        return save_model(name, weights, meta)

    def delete(self, name):
        delete_model(name)

    # ------------------------------------------------------------------ Récits

    def codec_note(self, meta, ds_key, n_images, seed, stats):
        k = int(meta["k"])
        bits = k * 32
        ratio = IMAGE_DIM * 8 / bits
        mse = meta["inertia"] / (meta["n_samples"] * IMAGE_DIM)
        return (
            f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}**, "
            f"« encodées » par le **GAN** (graine {seed}).\n\n"
            f"⚠️ **Un GAN n'a pas d'encodeur** : il n'apprend que le sens "
            f"z → image. Le code affiché est une **inversion approximative** — "
            f"le z de l'image générée la plus proche parmi une banque de "
            f"candidates (voir src/gan/utils/codec.py). Sur le papier, le débit "
            f"est excellent : {k} flottants ({bits} bits), soit **{ratio:.1f}:1** "
            f"face aux {IMAGE_DIM * 8} bits de l'image brute.\n\n"
            f"Mais la reconstruction ne rend PAS l'image de départ : elle rend "
            f"« ce que le générateur sait dessiner de plus ressemblant ». D'où la "
            f"**MSE : {mse:.4f}** par pixel — comparez-la à celle de la PCA ou de "
            f"l'autoencodeur à débit égal : c'est la démonstration chiffrée que "
            f"**le GAN est un générateur, pas un codec**.\n\n"
            f"Codes distincts sur ce tirage : {stats['Train']}/{n_images} en "
            f"train, {stats['Test']}/{n_images} en test."
        )

    def latent_note(self, meta):
        k = int(meta["k"])
        return (
            f"L'espace latent du GAN est l'ENTRÉE de son générateur : ℝ^{k}, "
            f"échantillonné à l'entraînement selon N(0, I). Contrairement à "
            f"l'autoencodeur ou au VAE, **aucune image réelle n'y a jamais été "
            f"rangée** — les points affichés sont les z *inversés* (plus proche "
            f"génération) de chaque image, une projection sur ce que G sait "
            f"dessiner.\n\n"
            f"À lire avec cette réserve : un nuage bien mélangé peut vouloir "
            f"dire que G couvre toutes les classes… ou que l'inversion est trop "
            f"grossière pour les séparer. La vraie lecture de santé du GAN est "
            f"dans l'onglet « Vues de l'algo » : le duel des losses et la grille "
            f"de générations.\n\n"
            "**Distribution** : chaque image est rangée sous la dimension "
            "dominante de son z inversé (|zᵢ − z̄ᵢ| maximal) — même convention "
            "que l'autoencodeur et le VAE."
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
        """Le dictionnaire du GAN n'existe pas : son générateur EST le dictionnaire.

        On en montre un échantillon — 16 tirages z ~ N(0, I) décodés — mais
        contrairement aux centroïdes d'un K-means, cette liste ne s'épuise
        jamais : chaque nouveau z est une nouvelle entrée.
        """
        k = int(meta["k"])
        seed = int(meta.get("seed", 42))
        rng = np.random.default_rng(seed + 2)
        Z = rng.standard_normal((N_GENERATED, k)).astype(np.float32)
        titles = ["z=(" + ", ".join(f"{v:+.1f}" for v in z[:2])
                  + (", …)" if k > 2 else ")") for z in Z]
        fig = viz.plot_generated_grid(
            generate_batch(Z, weights), titles,
            suptitle="Le « dictionnaire » du GAN — 16 tirages z ~ N(0, I) décodés",
        )
        note = (
            f"**Le dictionnaire du GAN est son générateur** : une fonction "
            f"apprise de ℝ^{k} vers l'image, dont voici 16 entrées tirées au "
            f"hasard. Là où K-means fige k prototypes et où le SOM fige une "
            f"grille, G en produit autant qu'on tire de z — un dictionnaire "
            f"infini, au prix de ne pas savoir retrouver l'entrée d'une image "
            f"donnée (pas d'encodeur)."
        )
        return fig, note

    def plot_curve(self, history, n_samples):
        # history est inertia_history = la loss du générateur ; le duel complet
        # (G + D + accuracies) vit dans « Vues de l'algo », qui a les métadonnées.
        fig = plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
        plt.xlabel("Époque")
        plt.ylabel("BCE du générateur")
        plt.title("GAN — loss du générateur par époque")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        note = (
            "La **loss du générateur** au fil des époques — la moitié visible du "
            "duel. ⚠️ Ne pas y chercher une descente vers 0 : dans un GAN sain, "
            "G et D s'annulent mutuellement et leurs losses **oscillent autour "
            "d'un équilibre**. La vue complète (G + D + accuracies du "
            "discriminateur) est dans l'onglet « Vues de l'algo »."
        )
        return fig, note

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Les vues signature du GAN : le duel, la maturation, la génération."""
        X = np.asarray(X, dtype=np.float32)
        k = int(meta["k"])
        seed = int(meta.get("seed", 42))
        figures = []

        # --- Le duel G / D : les deux losses et les deux accuracies de D.
        if meta.get("g_history") and meta.get("d_history"):
            figures.append((
                "Duel G / D — losses et verdict du discriminateur",
                viz.plot_gan_losses(
                    meta["g_history"], meta["d_history"],
                    meta.get("d_real_acc", []), meta.get("d_fake_acc", []),
                ),
            ))

        # --- La maturation du générateur : mêmes z, époques croissantes.
        snaps = meta.get("snapshots")
        if snaps:
            snapshots = [
                (s["epoch"], np.asarray(s["images"], dtype=np.float32) / 255.0)
                for s in snaps
            ]
            figures.append((
                "Maturation du générateur — du bruit à l'image",
                viz.plot_snapshot_grid(snapshots),
            ))

        # --- Génération : LA tâche du GAN. z ~ N(0, I), directement — comme le
        # VAE, mais sans passer par une loss de reconstruction qui moyenne.
        rng = np.random.default_rng(seed)
        sampled = rng.standard_normal(size=(N_GENERATED, k)).astype(np.float32)
        titles = ["z=(" + ", ".join(f"{v:+.1f}" for v in z[:2])
                  + (", …)" if k > 2 else ")") for z in sampled]
        figures.append((
            "Génération — échantillonnage de N(0, I)",
            viz.plot_generated_grid(
                generate_batch(sampled, weights), titles,
                suptitle="Images générées — z ~ N(0, I) décodés par le générateur",
            ),
        ))

        # --- Interpolation latente : la preuve d'une surface continue d'images.
        rows = []
        for r in range(INTERP_ROWS):
            za = rng.standard_normal(k).astype(np.float32)
            zb = rng.standard_normal(k).astype(np.float32)
            ts = np.linspace(0.0, 1.0, INTERP_STEPS, dtype=np.float32)[:, None]
            Z = (1.0 - ts) * za[None, :] + ts * zb[None, :]
            rows.append(generate_batch(Z, weights))
        figures.append((
            "Interpolation latente — de z_a à z_b",
            viz.plot_interpolation(rows),
        ))

        # --- Vraies contre générées : le test d'œil nu, côte à côte.
        n_show = min(8, len(X))
        idx = rng.choice(len(X), size=n_show, replace=False)
        gen = generate_batch(
            rng.standard_normal((n_show, k)).astype(np.float32), weights
        )
        figures.append((
            "Vraies vs générées — le test d'œil nu",
            viz.show_reconstructions(
                X[idx], gen,
                top_labels=["vraie"] * n_show, bottom_labels=["générée"] * n_show,
                title="Haut : images réelles · Bas : tirages du générateur",
            ),
        ))

        return figures

    def describe_rows(self, meta):
        return [
            ("Générateur", " → ".join(str(d) for d in meta.get("g_hidden", []))
                           + f" → {IMAGE_DIM}"),
            ("Discriminateur", f"{IMAGE_DIM} → "
                               + " → ".join(str(d) for d in meta.get("d_hidden", []))
                               + " → 1"),
            ("Dimension latente", f"{meta['k']}"),
            ("Pas d'apprentissage", f"{float(meta.get('learning_rate', 2e-4)):g} "
                                    f"(Adam β₁=0,5)"),
            ("Lissage d'étiquettes", f"{float(meta.get('label_smoothing', 0.9)):g}"),
            ("Entraînement", f"{meta.get('epochs', '?')} époques · "
                             f"batch {meta.get('batch_size', '?')}"),
            ("Images d'entraînement", f"{meta['n_samples']}"),
            ("MSE d'inversion (banque)", f"{meta['inertia'] / (meta['n_samples'] * IMAGE_DIM):.4f}"),
        ]
