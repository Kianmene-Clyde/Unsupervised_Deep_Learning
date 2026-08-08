# %% [markdown]
# # VAE — Plan d'expériences · Quick, Draw! (cat, apple, car)
# 
# Ce notebook reprend le VAE du notebook original avec **trois modifications** (repérées
# par `# >>> MODIFICATION` dans le code) :
# 
# 1. **Nouvel hyperparamètre `latent_activation`** : fonction d'activation appliquée à la
#    couche latente `z_mean` (`linear` par défaut, comme dans le VAE standard).
#    `z_log_var` reste linéaire : une activation bornée sur `log(σ²)` contraindrait la
#    variance à un intervalle arbitraire et casserait le reparameterization trick.
# 2. **Dataset Quick, Draw!** (cat, apple, car) chargé via `data_import.load_dataset("quickdraw")`
#    au lieu de MNIST.
# 3. **Instrumentation** : chaque run sauvegarde ses figures + un `metrics.json`
#    (reconstruction test, KL, silhouette de l'espace latent, nb de paramètres, durée)
#    pour comparer les 24 configurations à la fin.
# 
# ## Les trois tâches évaluées
# 
# | Tâche | Mesure quantitative | Mesure qualitative |
# |---|---|---|
# | **1. Reconstruction** | `reconstruction_loss` sur le test (MSE) | `02_reconstructions_test.png` |
# | **2. Structure de l'espace latent** | silhouette sur `z_mean` (séparation cat/apple/car) + `kl_loss` | scatter `03_scatter_latent.png` + grille `04_grille_latente.png` |
# | **3. Génération `z ~ N(0,1)`** | — (jugement visuel) | `05_generation.png` — **les mêmes z sont utilisés pour tous les runs**, la comparaison est donc directe |
# 
# ## Plan d'expériences — un seul hyperparamètre varie à la fois
# 
# La configuration de référence **A0** est celle du notebook original (dense, latent_dim=2,
# hidden=(128,32), relu, adam 1e-3, batch 256, mse, kl_weight=1e-3), avec `epochs=50`
# pour que les 24 runs restent réalisables (≈ 1 h sur CPU, quelques minutes sur GPU).
# Les meilleures configurations pourront être ré-entraînées à 200 epochs pour les
# figures finales du rapport.
# 
# | Série | Hyperparamètre | Valeurs (référence = A0) | Hypothèse à vérifier |
# |---|---|---|---|
# | **B** | `kl_weight` (β) | 0 · 1e-4 · **1e-3** · 1e-2 · 1e-1 | Compromis central du VAE : β↓ → meilleure reconstruction mais espace latent à « trous » → génération dégradée ; β↑ → espace ~N(0,1) mais reconstructions floues (posterior collapse à l'extrême) |
# | **C** | `latent_dim` | **2** · 4 · 8 · 16 | Plus de dimensions = goulot moins serré → reconstruction ↑ ; la grille 2D n'est possible qu'à dim 2 ; à dim élevée l'écart posterior/prior peut dégrader la génération |
# | **D** | `latent_activation` | **linear** · tanh · sigmoid · relu | Le prior N(0,1) est centré et symétrique : sigmoid/relu forcent μ ≥ 0 → la moitié de N(0,1) tombe dans une zone jamais vue par le décodeur → génération dégradée ; tanh (borné mais symétrique) devrait rester proche de linear |
# | **E** | `hidden_activation` | **relu** · tanh · elu · leaky_relu | Effets de second ordre : tanh sature → convergence plus lente ; elu/leaky évitent les ReLU « mortes » |
# | **F** | capacité / architecture | **(128,32) dense** · (32,) · (256,128,64) · conv | (32,) sous-apprend (flou) ; conv exploite la structure spatiale des traits → meilleures reconstruction et génération à budget comparable |
# | **G** | `learning_rate` | 1e-4 · **1e-3** · 1e-2 | 1e-4 : sous-convergence à budget fixé ; 1e-2 : instabilité/oscillations |
# | **H** | `batch_size` | 64 · **256** · 1024 | Petit batch = plus de mises à jour par epoch (gradient bruité, souvent ≈ ou mieux) ; 1024 = moins de pas → sous-convergence à epochs fixés |
# | **I** | `optimizer` | **adam** · rmsprop · sgd | SGD sans momentum à lr 1e-3 converge très lentement ; rmsprop ≈ adam |
# | **J** | `loss_name` | **mse** · binary_crossentropy | BCE (adaptée à des pixels ∈ [0,1]) donne des traits plus nets ; ⚠️ son échelle diffère de la MSE, donc le β *effectif* change — à discuter dans le rapport |
# %%
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# >>> MODIFICATION 2 : accès à data_import.py (Quick, Draw!)
# Adapter ce chemin vers le dossier qui contient data_import.py.
# ---------------------------------------------------------------------------
DATA_IMPORT_DIR = Path("../src/data_import.py")
sys.path.insert(0, str(DATA_IMPORT_DIR.resolve()))

try:
    from data_import import load_dataset
except ImportError as err:
    load_dataset = None
    print("ATTENTION : data_import.py introuvable — ajustez DATA_IMPORT_DIR.")
    print("Erreur :", err)

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("scikit-learn absent : silhouette et projection PCA désactivées.")

# Dossier racine des résultats : un sous-dossier par run.
OUTPUT_ROOT = Path("outputs_vae_quickdraw")

# Mettre à True pour afficher les figures dans le notebook en plus de les
# sauvegarder (déconseillé pour les 24 runs : ~120 figures inline).
SHOW_FIGURES = False
# %%
@dataclass
class VAEConfig:
    # "dense" ou "conv"
    architecture: str = "dense"
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (128, 32)

    # Activations
    hidden_activation: str = "relu"
    # >>> MODIFICATION 1 : activation de la couche latente (appliquée à z_mean).
    # "linear" = VAE standard. À tester : "tanh", "sigmoid", "relu".
    latent_activation: str = "linear"
    output_activation: str = "sigmoid"
    leaky_relu_slope: float = 0.2

    # Architecture convolutionnelle
    conv_filters: Tuple[int, ...] = (32, 64)
    conv_strides: Tuple[int, ...] = (2, 2)
    kernel_size: int = 3

    # Initialisation
    kernel_initializer: str = "glorot_uniform"

    # Apprentissage
    learning_rate: float = 1e-3
    batch_size: int = 256
    epochs: int = 50
    optimizer_name: str = "adam"
    loss_name: str = "mse"

    # Poids de la KL divergence dans la loss totale
    kl_weight: float = 1e-3

    # Préparation des données
    normalization: str = "zero_one"

    # Reproductibilité / affichage
    seed: int = 42
    n_examples: int = 10
    n_generated_images: int = 16
    latent_grid_size: int = 20
    latent_visualization_range: float = 3.0

    # Sorties
    output_dir: str = "outputs_vae_quickdraw/run"


class VariationalAutoencoder:
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1
    INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH

    def __init__(self, config: VAEConfig):
        self.config = config
        self._validate_config()

        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder, self.decoder = self._build_models()
        self.optimizer = self._build_optimizer()
        self.loss_fn = tf.keras.losses.get(config.loss_name)

        self.history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "kl_loss": [],
        }

    # ------------------------------------------------------------- validation
    def _validate_config(self):
        cfg = self.config

        if cfg.architecture not in {"dense", "conv"}:
            raise ValueError("architecture doit être 'dense' ou 'conv'.")

        if cfg.latent_dim < 1:
            raise ValueError("latent_dim doit être >= 1.")

        if cfg.architecture == "dense" and not cfg.hidden_dims:
            raise ValueError("hidden_dims ne peut pas être vide.")

        if cfg.architecture == "conv":
            if len(cfg.conv_filters) != len(cfg.conv_strides):
                raise ValueError(
                    "conv_filters et conv_strides doivent avoir la même longueur."
                )
            total_stride = int(np.prod(cfg.conv_strides))
            if self.IMAGE_HEIGHT % total_stride != 0:
                raise ValueError("Le produit de conv_strides doit diviser 28.")

        if cfg.normalization not in {"zero_one", "minus_one_one"}:
            raise ValueError("normalization doit être 'zero_one' ou 'minus_one_one'.")

    # ------------------------------------------------------------ activations
    def _apply_activation(self, x, activation_name: Optional[str], layer_name: str):
        if activation_name in {None, "linear"}:
            return x

        if activation_name == "leaky_relu":
            try:
                layer = tf.keras.layers.LeakyReLU(
                    negative_slope=self.config.leaky_relu_slope, name=layer_name
                )
            except TypeError:  # TF < 2.16 : l'argument s'appelle alpha
                layer = tf.keras.layers.LeakyReLU(
                    alpha=self.config.leaky_relu_slope, name=layer_name
                )
            return layer(x)

        return tf.keras.layers.Activation(activation_name, name=layer_name)(x)

    # ---------------------------------------------------------------- modèles
    def _build_models(self):
        if self.config.architecture == "dense":
            return self._build_dense_models()
        return self._build_conv_models()

    def _latent_heads(self, x):
        """Couches z_mean / z_log_var, communes aux deux architectures."""
        cfg = self.config

        z_mean = tf.keras.layers.Dense(
            cfg.latent_dim,
            kernel_initializer=cfg.kernel_initializer,
            name="z_mean",
        )(x)

        # >>> MODIFICATION 1 : activation optionnelle sur z_mean uniquement.
        z_mean = self._apply_activation(
            z_mean, cfg.latent_activation, "z_mean_activation"
        )

        z_log_var = tf.keras.layers.Dense(
            cfg.latent_dim,
            kernel_initializer=cfg.kernel_initializer,
            name="z_log_var",
        )(x)

        return z_mean, z_log_var

    def _build_dense_models(self):
        cfg = self.config

        # ---------------- ENCODEUR ----------------
        encoder_input = tf.keras.Input(shape=(self.INPUT_DIM,), name="image_flat")
        x = encoder_input

        for index, units in enumerate(cfg.hidden_dims):
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"encoder_dense_{index + 1}",
            )(x)
            x = self._apply_activation(
                x, cfg.hidden_activation, f"encoder_activation_{index + 1}"
            )

        z_mean, z_log_var = self._latent_heads(x)

        encoder = tf.keras.Model(
            encoder_input, [z_mean, z_log_var], name="vae_dense_encoder"
        )

        # ---------------- DÉCODEUR ----------------
        decoder_input = tf.keras.Input(shape=(cfg.latent_dim,), name="latent_vector")
        x = decoder_input

        for index, units in enumerate(reversed(cfg.hidden_dims)):
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"decoder_dense_{index + 1}",
            )(x)
            x = self._apply_activation(
                x, cfg.hidden_activation, f"decoder_activation_{index + 1}"
            )

        x = tf.keras.layers.Dense(
            self.INPUT_DIM,
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="reconstruction_linear",
        )(x)

        decoder_output = self._apply_activation(
            x, cfg.output_activation, "reconstruction_activation"
        )

        decoder = tf.keras.Model(decoder_input, decoder_output, name="vae_dense_decoder")

        return encoder, decoder

    def _build_conv_models(self):
        cfg = self.config

        # ---------------- ENCODEUR ----------------
        encoder_input = tf.keras.Input(
            shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS),
            name="image",
        )
        x = encoder_input

        for index, (filters, stride) in enumerate(
            zip(cfg.conv_filters, cfg.conv_strides)
        ):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=cfg.kernel_size,
                strides=stride,
                padding="same",
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"encoder_conv_{index + 1}",
            )(x)
            x = self._apply_activation(
                x, cfg.hidden_activation, f"encoder_conv_activation_{index + 1}"
            )

        shape_before_flatten = tuple(int(d) for d in x.shape[1:])
        x = tf.keras.layers.Flatten(name="encoder_flatten")(x)

        z_mean, z_log_var = self._latent_heads(x)

        encoder = tf.keras.Model(
            encoder_input, [z_mean, z_log_var], name="vae_conv_encoder"
        )

        # ---------------- DÉCODEUR ----------------
        decoder_input = tf.keras.Input(shape=(cfg.latent_dim,), name="latent_vector")

        x = tf.keras.layers.Dense(
            int(np.prod(shape_before_flatten)),
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="decoder_dense_projection",
        )(decoder_input)
        x = self._apply_activation(
            x, cfg.hidden_activation, "decoder_projection_activation"
        )
        x = tf.keras.layers.Reshape(shape_before_flatten, name="decoder_reshape")(x)

        for reverse_index in range(len(cfg.conv_filters) - 1, 0, -1):
            x = tf.keras.layers.Conv2DTranspose(
                filters=cfg.conv_filters[reverse_index - 1],
                kernel_size=cfg.kernel_size,
                strides=cfg.conv_strides[reverse_index],
                padding="same",
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"decoder_deconv_{reverse_index}",
            )(x)
            x = self._apply_activation(
                x, cfg.hidden_activation, f"decoder_deconv_activation_{reverse_index}"
            )

        x = tf.keras.layers.Conv2DTranspose(
            filters=self.IMAGE_CHANNELS,
            kernel_size=cfg.kernel_size,
            strides=cfg.conv_strides[0],
            padding="same",
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="reconstruction_conv",
        )(x)

        decoder_output = self._apply_activation(
            x, cfg.output_activation, "reconstruction_activation"
        )

        decoder = tf.keras.Model(decoder_input, decoder_output, name="vae_conv_decoder")

        return encoder, decoder

    def _build_optimizer(self):
        name = self.config.optimizer_name.lower()
        learning_rate = self.config.learning_rate

        if name == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if name == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        if name == "rmsprop":
            return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        raise ValueError("optimizer_name doit être 'adam', 'sgd' ou 'rmsprop'.")

    def count_params(self) -> int:
        return int(self.encoder.count_params() + self.decoder.count_params())

    # ----------------------------------------------------------------- données
    def _make_dataset(self, x_train):
        return (
            tf.data.Dataset.from_tensor_slices(x_train)
            .shuffle(
                buffer_size=len(x_train),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # -------------------------------------------------- reparameterization trick
    def sample_latent(self, z_mean, z_log_var):
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def forward(self, x, training=False):
        z_mean, z_log_var = self.encoder(x, training=training)
        z = self.sample_latent(z_mean, z_log_var)
        reconstruction = self.decoder(z, training=training)
        return z_mean, z_log_var, z, reconstruction

    # ------------------------------------------------------------------- losses
    def _compute_losses(self, x_true, x_reconstructed, z_mean, z_log_var):
        reconstruction_loss = tf.reduce_mean(self.loss_fn(x_true, x_reconstructed))

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1,
        )
        kl_loss = tf.reduce_mean(kl_loss)

        total_loss = reconstruction_loss + self.config.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, _, reconstruction = self.forward(x_batch, training=True)
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(
                x_batch, reconstruction, z_mean, z_log_var
            )

        variables = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        gradients = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return total_loss, reconstruction_loss, kl_loss

    def fit(self, x_train, verbose_every: int = 10):
        train_dataset = self._make_dataset(x_train)
        self.history = {"total_loss": [], "reconstruction_loss": [], "kl_loss": []}

        for epoch in range(1, self.config.epochs + 1):
            total_metric = tf.keras.metrics.Mean()
            reconstruction_metric = tf.keras.metrics.Mean()
            kl_metric = tf.keras.metrics.Mean()

            for x_batch in train_dataset:
                total_loss, reconstruction_loss, kl_loss = self.train_step(x_batch)
                total_metric.update_state(total_loss)
                reconstruction_metric.update_state(reconstruction_loss)
                kl_metric.update_state(kl_loss)

            self.history["total_loss"].append(float(total_metric.result().numpy()))
            self.history["reconstruction_loss"].append(
                float(reconstruction_metric.result().numpy())
            )
            self.history["kl_loss"].append(float(kl_metric.result().numpy()))

            if epoch == 1 or epoch % verbose_every == 0 or epoch == self.config.epochs:
                print(
                    f"  Epoch {epoch:03d}/{self.config.epochs} "
                    f"- total: {self.history['total_loss'][-1]:.6f} "
                    f"- reconstruction: {self.history['reconstruction_loss'][-1]:.6f} "
                    f"- KL: {self.history['kl_loss'][-1]:.6f}"
                )

        return self.history

    def evaluate(self, x_test):
        total_metric = tf.keras.metrics.Mean()
        reconstruction_metric = tf.keras.metrics.Mean()
        kl_metric = tf.keras.metrics.Mean()

        dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(
            self.config.batch_size
        )

        for x_batch in dataset:
            z_mean, z_log_var, _, reconstruction = self.forward(x_batch, training=False)
            total_loss, reconstruction_loss, kl_loss = self._compute_losses(
                x_batch, reconstruction, z_mean, z_log_var
            )
            total_metric.update_state(total_loss)
            reconstruction_metric.update_state(reconstruction_loss)
            kl_metric.update_state(kl_loss)

        return {
            "total_loss": float(total_metric.result().numpy()),
            "reconstruction_loss": float(reconstruction_metric.result().numpy()),
            "kl_loss": float(kl_metric.result().numpy()),
        }

    # ---------------------------------------------------------- encode / decode
    def encode(self, x):
        z_mean, z_log_var = self.encoder.predict(
            x, batch_size=self.config.batch_size, verbose=0
        )
        epsilon = (
            np.random.default_rng(self.config.seed)
            .normal(size=z_mean.shape)
            .astype(np.float32)
        )
        z = z_mean + np.exp(0.5 * z_log_var) * epsilon
        return z_mean, z_log_var, z

    def decode(self, z):
        return self.decoder.predict(z, batch_size=self.config.batch_size, verbose=0)

    def reconstruct(self, x):
        z_mean, z_log_var, z = self.encode(x)
        return z_mean, z_log_var, z, self.decode(z)

    # ------------------------------------------------------------------ figures
    def _finish_figure(self, fig, filename: str):
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if SHOW_FIGURES:
            plt.show()
        plt.close(fig)
        return path

    def plot_history(self):
        fig = plt.figure(figsize=(9, 5))
        epochs = range(1, len(self.history["total_loss"]) + 1)

        plt.plot(epochs, self.history["total_loss"], label="Loss totale")
        plt.plot(epochs, self.history["reconstruction_loss"], label="Reconstruction")
        plt.plot(epochs, self.history["kl_loss"], label="KL divergence")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Quick, Draw! - évolution des losses")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._finish_figure(fig, "01_losses.png")

    def plot_reconstructions(self, x, n_examples: Optional[int] = None):
        if n_examples is None:
            n_examples = self.config.n_examples

        x_examples = x[:n_examples]
        _, _, _, reconstructed = self.reconstruct(x_examples)

        fig, axes = plt.subplots(2, n_examples, figsize=(1.8 * n_examples, 4.2))
        if n_examples == 1:
            axes = np.array(axes).reshape(2, 1)

        for index in range(n_examples):
            axes[0, index].imshow(
                self._to_display_image(x_examples[index]), cmap="gray"
            )
            axes[1, index].imshow(
                self._to_display_image(reconstructed[index]), cmap="gray"
            )
            axes[0, index].axis("off")
            axes[1, index].axis("off")

        fig.suptitle("Haut : originales (test) — Bas : reconstructions")
        fig.tight_layout()

        return self._finish_figure(fig, "02_reconstructions_test.png")

    # >>> MODIFICATION 3 : scatter de l'espace latent coloré par classe
    # (cat / apple / car) — indispensable pour juger la tâche 2 sur Quick, Draw!.
    def plot_latent_scatter(self, x, y, class_names, max_points: int = 3000):
        z_mean, _, _ = self.encode(x[:max_points])
        labels = np.asarray(y[:max_points])

        title = "Espace latent (z_mean) coloré par classe"
        if z_mean.shape[1] > 2:
            if not SKLEARN_OK:
                print("latent_dim > 2 et sklearn absent : scatter ignoré.")
                return None
            z_mean = PCA(n_components=2).fit_transform(z_mean)
            title += " — projection PCA 2D"

        fig = plt.figure(figsize=(7, 6))
        for label_value, class_name in enumerate(class_names):
            mask = labels == label_value
            plt.scatter(
                z_mean[mask, 0],
                z_mean[mask, 1],
                s=4,
                alpha=0.5,
                label=class_name,
            )
        plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title(title)
        plt.legend(markerscale=3)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return self._finish_figure(fig, "03_scatter_latent.png")

    def latent_silhouette(self, x, y, max_points: int = 3000):
        """Silhouette des classes dans l'espace latent (z_mean) : ∈ [-1, 1],
        plus c'est haut, mieux les 3 classes sont séparées."""
        if not SKLEARN_OK:
            return None
        try:
            z_mean, _, _ = self.encode(x[:max_points])
            if not np.all(np.isfinite(z_mean)):
                return None
            return float(silhouette_score(z_mean, np.asarray(y[:max_points])))
        except Exception as err:
            print("Silhouette non calculable :", err)
            return None

    def visualize_latent_space(self, grid_size: Optional[int] = None):
        if self.config.latent_dim != 2:
            print("Grille latente : seulement pour latent_dim=2, ignorée.")
            return None

        if grid_size is None:
            grid_size = self.config.latent_grid_size

        limit = self.config.latent_visualization_range
        z1_values = np.linspace(-limit, limit, grid_size, dtype=np.float32)
        z2_values = np.linspace(limit, -limit, grid_size, dtype=np.float32)

        latent_points = np.array(
            [[z1, z2] for z2 in z2_values for z1 in z1_values], dtype=np.float32
        )
        decoded = self.decode(latent_points)

        canvas = np.zeros(
            (self.IMAGE_HEIGHT * grid_size, self.IMAGE_WIDTH * grid_size),
            dtype=np.float32,
        )

        index = 0
        for row in range(grid_size):
            for column in range(grid_size):
                image = self._to_display_image(decoded[index])
                row_start = row * self.IMAGE_HEIGHT
                column_start = column * self.IMAGE_WIDTH
                canvas[
                    row_start : row_start + self.IMAGE_HEIGHT,
                    column_start : column_start + self.IMAGE_WIDTH,
                ] = image
                index += 1

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(
            canvas,
            cmap="gray",
            extent=[-limit, limit, -limit, limit],
            origin="upper",
            aspect="auto",
        )
        plt.xlabel("Dimension latente z1")
        plt.ylabel("Dimension latente z2")
        plt.title("Grille latente : chaque position (z1, z2) est décodée")
        plt.tight_layout()

        return self._finish_figure(fig, "04_grille_latente.png")

    def generate_images(self, n_images: Optional[int] = None):
        if n_images is None:
            n_images = self.config.n_generated_images

        # Le seed est fixe : les MÊMES z sont échantillonnés pour toutes les
        # configurations de même latent_dim -> comparaison visuelle directe.
        rng = np.random.default_rng(self.config.seed)
        sampled_z = rng.normal(
            loc=0.0, scale=1.0, size=(n_images, self.config.latent_dim)
        ).astype(np.float32)

        generated = self.decode(sampled_z)

        n_cols = int(np.ceil(np.sqrt(n_images)))
        n_rows = int(np.ceil(n_images / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
        axes = np.array(axes).reshape(-1)

        for index, axis in enumerate(axes):
            axis.axis("off")
            if index < n_images:
                axis.imshow(self._to_display_image(generated[index]), cmap="gray")
                if self.config.latent_dim <= 4:
                    axis.set_title(
                        f"z={np.round(sampled_z[index], 2)}", fontsize=7
                    )

        fig.suptitle("Images générées avec z ~ N(0,1)")
        fig.tight_layout()

        return self._finish_figure(fig, "05_generation.png")

    def _to_display_image(self, image):
        image = np.asarray(image)

        if self.config.architecture == "dense":
            image = image.reshape(self.IMAGE_HEIGHT, self.IMAGE_WIDTH)
        else:
            image = np.squeeze(image, axis=-1)

        if self.config.normalization == "minus_one_one":
            image = (image + 1.0) / 2.0

        return np.clip(image, 0.0, 1.0)
# %% [markdown]
# ## Chargement des données — Quick, Draw! (cat, apple, car)
# 
# `load_dataset("quickdraw")` renvoie X aplati `(n, 784)` en float32 ∈ [0, 1] : exactement
# le format attendu par l'architecture dense. Pour l'architecture conv, on remet en
# `(n, 28, 28, 1)`. Le premier appel télécharge trois `.npy` (~100 Mo chacun).
# %%
if load_dataset is None:
    raise RuntimeError(
        "data_import.py est introuvable : corrigez DATA_IMPORT_DIR plus haut."
    )

DATASET = load_dataset("quickdraw")  # cat, apple, car — seed 42, split 6/7 - 1/7
print(DATASET.describe())
print("X_train :", DATASET.X_train.shape, "| X_test :", DATASET.X_test.shape)


def prepare_arrays(dataset, config: VAEConfig):
    """Adapte le format commun (n, 784) ∈ [0,1] à l'architecture et à la normalisation."""
    X_train, X_test = dataset.X_train, dataset.X_test

    if config.normalization == "minus_one_one":
        X_train = X_train * 2.0 - 1.0
        X_test = X_test * 2.0 - 1.0

    if config.architecture == "conv":
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_test


# Aperçu de quelques dessins
fig, axes = plt.subplots(1, 9, figsize=(14, 2))
for axis, index in zip(axes, range(9)):
    axis.imshow(DATASET.X_train[index].reshape(28, 28), cmap="gray")
    axis.set_title(DATASET.class_names[DATASET.y_train[index]], fontsize=9)
    axis.axis("off")
plt.tight_layout()
plt.show()
# %% [markdown]
# ## Définition des 24 configurations
# 
# `BASE_PARAMS` = configuration de référence **A0** (identique au notebook original,
# hormis `epochs=50` pour le budget de calcul). Chaque expérience ne modifie
# **qu'une seule clé** par rapport à A0 — c'est ce qui rend les comparaisons
# interprétables dans le rapport.
# %%
BASE_PARAMS = dict(
    architecture="dense",
    latent_dim=2,
    hidden_dims=(128, 32),
    hidden_activation="relu",
    latent_activation="linear",
    output_activation="sigmoid",
    kernel_initializer="glorot_uniform",
    learning_rate=1e-3,
    batch_size=256,
    epochs=50,
    optimizer_name="adam",
    loss_name="mse",
    kl_weight=1e-3,
    normalization="zero_one",
    seed=42,
)

# (nom_du_run, {hyperparamètre modifié}, hypothèse résumée)
EXPERIMENTS = [
    ("A0_baseline", {}, "Référence : la config du notebook original."),

    # --- Série B : kl_weight (β), LE paramètre propre au VAE -----------------
    ("B1_kl_0", {"kl_weight": 0.0},
     "β=0 => AE pur : meilleure reconstruction, espace latent non régularisé "
     "=> génération z~N(0,1) dégradée (trous)."),
    ("B2_kl_1e-4", {"kl_weight": 1e-4},
     "β faible : proche AE, régularisation légère."),
    ("B3_kl_1e-2", {"kl_weight": 1e-2},
     "β fort : espace plus proche de N(0,1), reconstructions plus floues, "
     "génération plus fiable."),
    ("B4_kl_1e-1", {"kl_weight": 1e-1},
     "β très fort : risque de posterior collapse (tout au centre, images "
     "moyennes)."),

    # --- Série C : latent_dim ------------------------------------------------
    ("C1_latent_4", {"latent_dim": 4},
     "Goulot moins serré => reconstruction meilleure."),
    ("C2_latent_8", {"latent_dim": 8},
     "Encore mieux en reconstruction, rendements décroissants."),
    ("C3_latent_16", {"latent_dim": 16},
     "Meilleure reconstruction attendue ; grille 2D impossible ; écart "
     "posterior/prior possible en génération."),

    # --- Série D : latent_activation (sur z_mean) ----------------------------
    ("D1_latent_tanh", {"latent_activation": "tanh"},
     "μ borné dans (-1,1) mais symétrique et centré : cohérent avec N(0,1), "
     "espace compressé."),
    ("D2_latent_sigmoid", {"latent_activation": "sigmoid"},
     "μ ∈ (0,1) : incompatible avec un prior centré en 0 => moitié de N(0,1) "
     "jamais couverte => génération dégradée."),
    ("D3_latent_relu", {"latent_activation": "relu"},
     "μ ≥ 0 : même problème d'asymétrie + risque de dimensions mortes."),

    # --- Série E : hidden_activation -----------------------------------------
    ("E1_hidden_tanh", {"hidden_activation": "tanh"},
     "Saturation => convergence plus lente à budget fixé."),
    ("E2_hidden_elu", {"hidden_activation": "elu"},
     "Évite les ReLU mortes, souvent ≈ relu ou légèrement mieux."),
    ("E3_hidden_leaky", {"hidden_activation": "leaky_relu"},
     "Idem elu : gradient non nul pour x<0."),

    # --- Série F : capacité / architecture -----------------------------------
    ("F1_petit_32", {"hidden_dims": (32,)},
     "Sous-capacité => reconstructions floues partout."),
    ("F2_grand_256_128_64", {"hidden_dims": (256, 128, 64)},
     "Plus de capacité => meilleure reconstruction, coût x plusieurs."),
    ("F3_conv", {"architecture": "conv"},
     "Biais inductif spatial (traits) => meilleur rapport qualité/paramètres "
     "attendu en reconstruction ET en génération."),

    # --- Série G : learning_rate ---------------------------------------------
    ("G1_lr_1e-4", {"learning_rate": 1e-4},
     "Trop lent : sous-convergence à 50 epochs."),
    ("G2_lr_1e-2", {"learning_rate": 1e-2},
     "Trop grand : instabilité / oscillations, voire divergence."),

    # --- Série H : batch_size ------------------------------------------------
    ("H1_batch_64", {"batch_size": 64},
     "4x plus de mises à jour par epoch : souvent ≈ ou mieux, plus lent à "
     "l'horloge."),
    ("H2_batch_1024", {"batch_size": 1024},
     "4x moins de pas => sous-convergence à epochs fixés."),

    # --- Série I : optimizer -------------------------------------------------
    ("I1_rmsprop", {"optimizer_name": "rmsprop"},
     "Adaptatif comme adam : résultats proches attendus."),
    ("I2_sgd", {"optimizer_name": "sgd"},
     "Sans momentum ni adaptation, lr 1e-3 : convergence très lente => tout "
     "est dégradé (montre l'importance de l'optimiseur)."),

    # --- Série J : loss de reconstruction ------------------------------------
    ("J1_bce", {"loss_name": "binary_crossentropy"},
     "BCE adaptée aux pixels ∈ [0,1] : traits plus nets attendus. ⚠️ échelle "
     "différente de la MSE => β effectif différent (à discuter)."),
]

print(f"{len(EXPERIMENTS)} configurations définies.")
# %% [markdown]
# ## Runner
# 
# Chaque run écrit dans `outputs_vae_quickdraw/<nom_du_run>/` :
# les 5 figures + `metrics.json`. Un run déjà terminé (metrics.json présent)
# est **sauté** : on peut interrompre et relancer la cellule sans tout refaire.
# %%
def run_experiment(run_name, overrides, hypothese, dataset):
    run_dir = OUTPUT_ROOT / run_name
    metrics_path = run_dir / "metrics.json"

    if metrics_path.exists():
        print(f"[skip] {run_name} : déjà fait.")
        with open(metrics_path, encoding="utf-8") as handle:
            return json.load(handle)

    print(f"\n========== {run_name} | modif: {overrides or 'aucune (baseline)'} ==========")

    tf.keras.backend.clear_session()
    config = VAEConfig(**{**BASE_PARAMS, **overrides}, output_dir=str(run_dir))
    vae = VariationalAutoencoder(config)

    X_train, X_test = prepare_arrays(dataset, config)

    start = time.time()
    vae.fit(X_train)
    duration = time.time() - start

    test_metrics = vae.evaluate(X_test)
    silhouette = vae.latent_silhouette(X_test, dataset.y_test)

    # Figures (sauvegardées dans run_dir)
    vae.plot_history()
    vae.plot_reconstructions(X_test)
    vae.plot_latent_scatter(X_test, dataset.y_test, dataset.class_names)
    vae.visualize_latent_space()
    vae.generate_images()

    row = {
        "run": run_name,
        "overrides": {k: str(v) for k, v in overrides.items()},
        "hypothese": hypothese,
        "test_total_loss": test_metrics["total_loss"],
        "test_reconstruction_loss": test_metrics["reconstruction_loss"],
        "test_kl_loss": test_metrics["kl_loss"],
        "silhouette_latent": silhouette,
        "train_reconstruction_loss_final": vae.history["reconstruction_loss"][-1],
        "n_params": vae.count_params(),
        "duree_s": round(duration, 1),
        "config": {k: str(v) for k, v in asdict(config).items()},
    }

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(row, handle, ensure_ascii=False, indent=2)

    print(
        f"[ok] {run_name} : recon test = {row['test_reconstruction_loss']:.5f} | "
        f"KL = {row['test_kl_loss']:.3f} | silhouette = {silhouette} | "
        f"{row['duree_s']} s"
    )
    return row
# %%
# ============================ LANCEMENT DES 24 RUNS ============================
all_rows = []
for run_name, overrides, hypothese in EXPERIMENTS:
    try:
        all_rows.append(run_experiment(run_name, overrides, hypothese, DATASET))
    except Exception as err:
        print(f"[ÉCHEC] {run_name} : {err}")

print(f"\nTerminé : {len(all_rows)}/{len(EXPERIMENTS)} runs disponibles.")
# %% [markdown]
# ## Tableau récapitulatif
# 
# Rappels de lecture :
# 
# - **Tâche 1 (reconstruction)** : plus `test_reconstruction_loss` est basse, mieux c'est.
#   ⚠️ La loss du run `J1_bce` est une BCE : elle n'est **pas comparable numériquement**
#   aux MSE des autres runs (comparer J1 visuellement, figure 02).
# - **Tâche 2 (espace latent)** : `silhouette_latent` haute = classes bien séparées ;
#   `test_kl_loss` basse = espace proche de N(0,1). Les deux se lisent avec les
#   figures 03 (scatter) et 04 (grille).
# - **Tâche 3 (génération)** : jugement visuel sur la figure 05 — mêmes z pour tous
#   les runs de même latent_dim, donc comparaison directe case par case.
# %%
import pandas as pd

rows = []
for metrics_file in sorted(OUTPUT_ROOT.glob("*/metrics.json")):
    with open(metrics_file, encoding="utf-8") as handle:
        rows.append(json.load(handle))

df = pd.DataFrame(
    [
        {
            "run": r["run"],
            "modif": ", ".join(f"{k}={v}" for k, v in r["overrides"].items()) or "—",
            "recon_test": r["test_reconstruction_loss"],
            "kl_test": r["test_kl_loss"],
            "silhouette": r["silhouette_latent"],
            "params": r["n_params"],
            "duree_s": r["duree_s"],
        }
        for r in rows
    ]
)

print("--- Classement tâche 1 : reconstruction (hors J1_bce, échelle différente) ---")
display(
    df[df["run"] != "J1_bce"]
    .sort_values("recon_test")
    .reset_index(drop=True)
)

print("--- Classement tâche 2 : séparation des classes dans le latent ---")
display(df.sort_values("silhouette", ascending=False).reset_index(drop=True))
# %%
# Graphiques comparatifs
plot_df = df[df["run"] != "J1_bce"].sort_values("recon_test")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

ax1.barh(plot_df["run"], plot_df["recon_test"], color="steelblue")
ax1.invert_yaxis()
ax1.set_xlabel("Reconstruction loss test (MSE)")
ax1.set_title("Tâche 1 — reconstruction (plus bas = mieux)")
ax1.grid(True, axis="x", alpha=0.3)

sil_df = df.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False)
ax2.barh(sil_df["run"], sil_df["silhouette"], color="darkorange")
ax2.invert_yaxis()
ax2.set_xlabel("Silhouette des classes sur z_mean")
ax2.set_title("Tâche 2 — structure de l'espace latent (plus haut = mieux)")
ax2.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "00_comparaison_globale.png", dpi=150, bbox_inches="tight")
plt.show()
# %% [markdown]
# ## Grille d'analyse pour le rapport — quelle config gagne quelle tâche, et pourquoi
# 
# ### Tâche 1 — Reconstruction
# Gagnants attendus : `B1_kl_0` / `B2_kl_1e-4`, `C3_latent_16`, `F2_grand`, `F3_conv`.
# **Pourquoi** : la reconstruction est la partie « autoencodeur » pure de la loss. Tout ce
# qui relâche la contrainte l'améliore : β→0 supprime la pénalité KL (le modèle redevient
# un AE) ; un latent plus grand élargit le goulot d'étranglement ; plus de capacité
# (couches, conv) permet de modéliser des traits plus fins. Le point clé du rapport :
# `B1` gagne la tâche 1 **en sacrifiant les tâches 2 et 3** — c'est le compromis
# reconstruction/régularisation qui définit le VAE.
# 
# ### Tâche 2 — Structure de l'espace latent
# Gagnants attendus : `A0` ou `B3_kl_1e-2` (β intermédiaire), avec `latent_dim=2`.
# **Pourquoi** : la KL tire chaque q(z|x) vers N(0,1) → espace compact, continu, centré,
# sans « trous » entre les classes ; la grille 04 montre alors des transitions douces
# chat↔pomme↔voiture. Si β est trop faible (B1/B2) l'espace s'étire librement (bonne
# silhouette possible mais zones vides) ; trop fort (B4), tout s'effondre au centre :
# silhouette en chute, classes indiscernables (posterior collapse). Côté série D :
# `sigmoid`/`relu` confinent μ dans un quadrant positif → nuage asymétrique décalé du
# prior, visible immédiatement sur le scatter 03.
# 
# ### Tâche 3 — Génération z ~ N(0,1)
# Gagnants attendus : `B3_kl_1e-2` (ou `A0`), et `F3_conv` pour la netteté des traits.
# **Pourquoi** : la génération réussit si le décodeur a « vu » pendant l'entraînement les
# régions où N(0,1) échantillonne — autrement dit si le posterior agrégé recouvre le
# prior, ce que seul un β suffisant garantit. β=0 → les z tirés tombent dans des trous →
# images dégénérées ; β trop grand → le décodeur ne produit plus que des moyennes floues.
# `D2_sigmoid`/`D3_relu` échouent partiellement : ~la moitié des z ~ N(0,1) (composantes
# négatives) n'a jamais été couverte par des μ positifs. C'est la démonstration
# expérimentale de pourquoi la couche latente d'un VAE est **linéaire** dans la
# littérature.
# 
# ### Pour finir le rapport
# 1. Ré-entraîner les 2-3 meilleures configs à `epochs=200` pour les figures finales.
# 2. Croiser les deux gagnantes si pertinent (ex. conv + β=1e-2) — en le présentant
#    comme une expérience de confirmation, pas comme une variation « un seul paramètre ».