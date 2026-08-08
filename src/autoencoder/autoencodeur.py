# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


@dataclass
class AutoencoderConfig:
    # "dense" ou "conv"
    architecture: str = "dense"
    latent_dim: int = 2
    hidden_dims: Tuple[int, ...] = (128, 32)

    # Activations
    # Exemples : "relu", "elu", "tanh", "leaky_relu", "linear"
    hidden_activation: str = "relu"
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
    epochs: int = 200
    optimizer_name: str = "adam"
    loss_name: str = "mse"

    # Préparation des données
    # "zero_one" OU "minus_one_one"
    normalization: str = "zero_one"

    # Reproductibilité
    seed: int = 42
    n_examples: int = 10
    n_generated_images: int = 16
    latent_grid_size: int = 20

    # Sorties
    output_dir: str = "outputs_autoencoder_mnist"


class SimpleAutoencoder:
    IMAGE_HEIGHT = 28
    IMAGE_WIDTH = 28
    IMAGE_CHANNELS = 1
    INPUT_DIM = IMAGE_HEIGHT * IMAGE_WIDTH

    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self._validate_config()

        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder, self.decoder = self._build_models()
        self.optimizer = self._build_optimizer()
        self.loss_fn = tf.keras.losses.get(config.loss_name)

        self.history = []

    def _validate_config(self):
        cfg = self.config

        if cfg.architecture not in {"dense", "conv"}:
            raise ValueError(
                "architecture doit être 'dense' ou 'conv'."
            )

        if cfg.latent_dim < 1:
            raise ValueError("latent_dim doit être >= 1.")

        if not cfg.hidden_dims and cfg.architecture == "dense":
            raise ValueError(
                "hidden_dims doit contenir au moins une couche "
                "pour l'architecture dense."
            )

        if cfg.architecture == "conv":
            if len(cfg.conv_filters) != len(cfg.conv_strides):
                raise ValueError(
                    "conv_filters et conv_strides doivent avoir "
                    "la même longueur."
                )

            if not cfg.conv_filters:
                raise ValueError(
                    "conv_filters ne peut pas être vide."
                )

            total_stride = int(np.prod(cfg.conv_strides))

            if self.IMAGE_HEIGHT % total_stride != 0:
                raise ValueError(
                    "Pour MNIST, le produit de conv_strides doit "
                    "diviser exactement 28. "
                    f"Produit actuel = {total_stride}."
                )

        if cfg.normalization not in {"zero_one", "minus_one_one"}:
            raise ValueError(
                "normalization doit être 'zero_one' "
                "ou 'minus_one_one'."
            )

        if (
                cfg.normalization == "zero_one"
                and cfg.output_activation == "tanh"
        ):
            print(
                "AVERTISSEMENT : données dans [0,1] avec sortie tanh. "
                "Pour tanh en sortie, 'minus_one_one' est généralement "
                "plus cohérent."
            )

        if (
                cfg.normalization == "minus_one_one"
                and cfg.output_activation == "sigmoid"
        ):
            print(
                "AVERTISSEMENT : données dans [-1,1] avec sortie sigmoid. "
            )

    def _apply_activation(
            self,
            x,
            activation_name: Optional[str],
            layer_name: str,
    ):

        if activation_name in {None, "linear"}:
            return x

        if activation_name == "leaky_relu":
            return tf.keras.layers.LeakyReLU(
                negative_slope=self.config.leaky_relu_slope,
                name=layer_name,
            )(x)

        return tf.keras.layers.Activation(
            activation_name,
            name=layer_name,
        )(x)

    def _build_models(self):
        if self.config.architecture == "dense":
            return self._build_dense_models()

        return self._build_conv_models()

    def _build_dense_models(self):
        cfg = self.config

        # ---------------- ENCODEUR DENSE ----------------
        encoder_input = tf.keras.Input(
            shape=(self.INPUT_DIM,),
            name="image_flat",
        )

        x = encoder_input

        for index, units in enumerate(cfg.hidden_dims):
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"encoder_dense_{index + 1}",
            )(x)

            x = self._apply_activation(
                x,
                cfg.hidden_activation,
                f"encoder_activation_{index + 1}",
            )

        latent = tf.keras.layers.Dense(
            cfg.latent_dim,
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="latent_linear",
        )(x)

        latent = self._apply_activation(
            latent,
            cfg.latent_activation,
            "latent_activation",
        )

        encoder = tf.keras.Model(
            encoder_input,
            latent,
            name="dense_encoder",
        )

        # ---------------- DÉCODEUR DENSE ----------------
        decoder_input = tf.keras.Input(
            shape=(cfg.latent_dim,),
            name="latent_vector",
        )

        x = decoder_input

        for index, units in enumerate(reversed(cfg.hidden_dims)):
            x = tf.keras.layers.Dense(
                units,
                activation=None,
                kernel_initializer=cfg.kernel_initializer,
                name=f"decoder_dense_{index + 1}",
            )(x)

            x = self._apply_activation(
                x,
                cfg.hidden_activation,
                f"decoder_activation_{index + 1}",
            )

        x = tf.keras.layers.Dense(
            self.INPUT_DIM,
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="reconstruction_linear",
        )(x)

        decoder_output = self._apply_activation(
            x,
            cfg.output_activation,
            "reconstruction_activation",
        )

        decoder = tf.keras.Model(
            decoder_input,
            decoder_output,
            name="dense_decoder",
        )

        return encoder, decoder

    def _build_conv_models(self):
        cfg = self.config

        # ---------------- ENCODEUR CONV ----------------
        encoder_input = tf.keras.Input(
            shape=(
                self.IMAGE_HEIGHT,
                self.IMAGE_WIDTH,
                self.IMAGE_CHANNELS,
            ),
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
                x,
                cfg.hidden_activation,
                f"encoder_conv_activation_{index + 1}",
            )

        shape_before_flatten = tuple(
            int(dimension)
            for dimension in x.shape[1:]
        )

        x = tf.keras.layers.Flatten(
            name="encoder_flatten"
        )(x)

        latent = tf.keras.layers.Dense(
            cfg.latent_dim,
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="latent_linear",
        )(x)

        latent = self._apply_activation(
            latent,
            cfg.latent_activation,
            "latent_activation",
        )

        encoder = tf.keras.Model(
            encoder_input,
            latent,
            name="conv_encoder",
        )

        # ---------------- DÉCODEUR CONV ----------------
        decoder_input = tf.keras.Input(
            shape=(cfg.latent_dim,),
            name="latent_vector",
        )

        x = tf.keras.layers.Dense(
            int(np.prod(shape_before_flatten)),
            activation=None,
            kernel_initializer=cfg.kernel_initializer,
            name="decoder_dense_projection",
        )(decoder_input)

        x = self._apply_activation(
            x,
            cfg.hidden_activation,
            "decoder_projection_activation",
        )

        x = tf.keras.layers.Reshape(
            shape_before_flatten,
            name="decoder_reshape",
        )(x)

        for reverse_index in range(
                len(cfg.conv_filters) - 1,
                0,
                -1,
        ):
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
                x,
                cfg.hidden_activation,
                f"decoder_deconv_activation_{reverse_index}",
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
            x,
            cfg.output_activation,
            "reconstruction_activation",
        )

        decoder = tf.keras.Model(
            decoder_input,
            decoder_output,
            name="conv_decoder",
        )

        expected_shape = (
            None,
            self.IMAGE_HEIGHT,
            self.IMAGE_WIDTH,
            self.IMAGE_CHANNELS,
        )

        if decoder.output_shape != expected_shape:
            raise ValueError(
                "La configuration convolutionnelle ne reconstruit pas "
                f"une image 28x28x1. Shape obtenue : {decoder.output_shape}. "
            )

        return encoder, decoder

    def _build_optimizer(self):
        name = self.config.optimizer_name.lower()
        learning_rate = self.config.learning_rate

        if name == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=learning_rate
            )

        if name == "sgd":
            return tf.keras.optimizers.SGD(
                learning_rate=learning_rate
            )

        if name == "rmsprop":
            return tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate
            )

        raise ValueError(
            "optimizer_name doit être 'adam', 'sgd' ou 'rmsprop'."
        )

    # DONNÉES
    def load_mnist(self):
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.mnist.load_data()
        )

        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)

        if self.config.normalization == "zero_one":
            x_train = x_train / 255.0
            x_test = x_test / 255.0
        else:
            x_train = (x_train / 127.5) - 1.0
            x_test = (x_test / 127.5) - 1.0

        if self.config.architecture == "dense":
            x_train = x_train.reshape(-1, self.INPUT_DIM)
            x_test = x_test.reshape(-1, self.INPUT_DIM)
        else:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)

        print("\n================ DONNÉES MNIST ================")
        print("Architecture :", self.config.architecture)
        print("x_train      :", x_train.shape)
        print("x_test       :", x_test.shape)
        print("y_train      :", y_train.shape)
        print("Pixel min/max:", x_train.min(), "/", x_train.max())

        return x_train, y_train, x_test, y_test

    def _make_dataset(self, x_train):
        return (
            tf.data.Dataset
            .from_tensor_slices(x_train)
            .shuffle(
                buffer_size=len(x_train),
                seed=self.config.seed,
                reshuffle_each_iteration=True,
            )
            .batch(self.config.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

    # FORWARD
    def forward(self, x, training=False):
        z = self.encoder(x, training=training)
        reconstruction = self.decoder(
            z,
            training=training,
        )
        return z, reconstruction

    def _compute_loss(self, x_true, x_reconstructed):
        loss = self.loss_fn(
            x_true,
            x_reconstructed,
        )
        return tf.reduce_mean(loss)

    @tf.function
    def train_step(self, x_batch):
        with tf.GradientTape() as tape:
            _, reconstruction = self.forward(
                x_batch,
                training=True,
            )

            loss = self._compute_loss(
                x_batch,
                reconstruction,
            )

        variables = (
                self.encoder.trainable_variables
                + self.decoder.trainable_variables
        )

        gradients = tape.gradient(
            loss,
            variables,
        )

        self.optimizer.apply_gradients(
            zip(gradients, variables)
        )

        return loss

    def fit(self, x_train):
        train_dataset = self._make_dataset(x_train)
        self.history = []

        print("\n================ ENTRAÎNEMENT ================")
        self.print_config()

        for epoch in range(
                1,
                self.config.epochs + 1,
        ):
            metric = tf.keras.metrics.Mean()

            for x_batch in train_dataset:
                loss = self.train_step(x_batch)
                metric.update_state(loss)

            epoch_loss = float(metric.result().numpy())
            self.history.append(epoch_loss)

            print(
                f"Epoch {epoch:02d}/{self.config.epochs} "
                f"- loss: {epoch_loss:.6f}"
            )

        return self.history

    def evaluate(self, x_test):
        metric = tf.keras.metrics.Mean()

        dataset = (
            tf.data.Dataset
            .from_tensor_slices(x_test)
            .batch(self.config.batch_size)
        )

        for x_batch in dataset:
            _, reconstruction = self.forward(
                x_batch,
                training=False,
            )

            loss = self._compute_loss(
                x_batch,
                reconstruction,
            )

            metric.update_state(loss)

        value = float(metric.result().numpy())

        print(
            "\nLoss de reconstruction sur le test :",
            value,
        )

        return value

    # ENCODE / DECODE
    def encode(self, x):
        return self.encoder.predict(
            x,
            batch_size=self.config.batch_size,
            verbose=0,
        )

    def decode(self, z):
        return self.decoder.predict(
            z,
            batch_size=self.config.batch_size,
            verbose=0,
        )

    def reconstruct(self, x):
        z = self.encode(x)
        reconstruction = self.decode(z)
        return z, reconstruction

    # AFFICHAGES
    def print_config(self):
        cfg = self.config

        print("\n--- CONFIGURATION ---")
        print("architecture       :", cfg.architecture)
        print("latent_dim         :", cfg.latent_dim)
        print("hidden_dims        :", cfg.hidden_dims)
        print("hidden_activation  :", cfg.hidden_activation)
        print("latent_activation  :", cfg.latent_activation)
        print("output_activation  :", cfg.output_activation)

        if cfg.architecture == "conv":
            print("conv_filters       :", cfg.conv_filters)
            print("conv_strides       :", cfg.conv_strides)
            print("kernel_size        :", cfg.kernel_size)

        print("initializer        :", cfg.kernel_initializer)
        print("optimizer          :", cfg.optimizer_name)
        print("learning_rate      :", cfg.learning_rate)
        print("loss               :", cfg.loss_name)
        print("batch_size         :", cfg.batch_size)
        print("epochs             :", cfg.epochs)
        print("normalization      :", cfg.normalization)

    def show_summaries(self):
        print("\n================ ENCODEUR ================")
        self.encoder.summary()

        print("\n================ DÉCODEUR ================")
        self.decoder.summary()

    def plot_train_reconstructions(
            self,
            x_train,
            n_examples: Optional[int] = None,
    ):

        if n_examples is None:
            n_examples = self.config.n_examples

        x_examples = x_train[:n_examples]
        _, reconstructed = self.reconstruct(x_examples)

        fig, axes = plt.subplots(
            2,
            n_examples,
            figsize=(1.8 * n_examples, 4.5),
        )

        if n_examples == 1:
            axes = np.array(axes).reshape(2, 1)

        for index in range(n_examples):
            original_image = self._to_display_image(
                x_examples[index]
            )

            reconstructed_image = self._to_display_image(
                reconstructed[index]
            )

            axes[0, index].imshow(
                original_image,
                cmap="gray",
            )
            axes[0, index].axis("off")

            axes[1, index].imshow(
                reconstructed_image,
                cmap="gray",
            )
            axes[1, index].axis("off")

        axes[0, 0].set_ylabel(
            "Train\noriginal",
            fontsize=11,
        )

        axes[1, 0].set_ylabel(
            "Train\nreconstruit",
            fontsize=11,
        )

        fig.suptitle(
            "Reconstructions sur le jeu d'entraînement"
        )

        fig.tight_layout()

        path = (
                self.output_dir
                / "01_train_reconstructions.png"
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print("Figure sauvegardée :", path)
        plt.show()

        return reconstructed, fig

    def plot_history(self):
        if not self.history:
            raise RuntimeError(
                "Entraînez d'abord le modèle avec fit()."
            )

        fig = plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, len(self.history) + 1),
            self.history,
            marker="o",
        )

        plt.xlabel("Epoch")
        plt.ylabel(self.config.loss_name)
        plt.title(
            "AutoEncodeur MNIST - évolution de la loss"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = self.output_dir / "02_training_loss.png"
        fig.savefig(path, dpi=150)

        print("Figure sauvegardée :", path)
        plt.show()

        return fig

    def inspect_latent_and_reconstructions(
            self,
            x_test,
            y_test,
            n_examples: Optional[int] = None,
    ):

        if n_examples is None:
            n_examples = self.config.n_examples

        x_examples = x_test[:n_examples]
        y_examples = y_test[:n_examples]

        z, reconstructed = self.reconstruct(
            x_examples
        )

        print(
            "\n========== POINTS LATENTS APRÈS ENTRAÎNEMENT =========="
        )

        print("Latent shape :", z.shape)

        for index, latent_point in enumerate(z):
            latent_text = np.array2string(
                latent_point,
                precision=3,
                suppress_small=True,
            )

            print(
                f"Image {index:02d} | y={y_examples[index]} "
                f"| z={latent_text}"
            )

        fig, axes = plt.subplots(
            2,
            n_examples,
            figsize=(1.8 * n_examples, 4.8),
        )

        if n_examples == 1:
            axes = np.array(axes).reshape(2, 1)

        for index in range(n_examples):
            original_image = self._to_display_image(
                x_examples[index]
            )

            reconstructed_image = self._to_display_image(
                reconstructed[index]
            )

            axes[0, index].imshow(
                original_image,
                cmap="gray",
            )

            axes[0, index].axis("off")

            latent_title = np.array2string(
                z[index],
                precision=2,
                suppress_small=True,
            )

            axes[0, index].set_title(
                f"y={y_examples[index]}\nz={latent_title}",
                fontsize=8,
            )

            axes[1, index].imshow(
                reconstructed_image,
                cmap="gray",
            )

            axes[1, index].axis("off")

        axes[0, 0].set_ylabel(
            "Originale",
            fontsize=11,
        )

        axes[1, 0].set_ylabel(
            "Reconstruite",
            fontsize=11,
        )

        fig.suptitle(
            "Points latents et reconstructions après entraînement"
        )

        fig.tight_layout()

        path = (
                self.output_dir
                / "03_points_latents_et_reconstructions.png"
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print("Figure sauvegardée :", path)
        plt.show()

        return z, reconstructed, fig

    def visualize_latent_space(
            self,
            x_reference,
            grid_size: Optional[int] = None,
    ):

        if self.config.latent_dim != 2:
            print(
                "\nVisualisation directe ignorée : latent_dim = "
                f"{self.config.latent_dim}. "
                "Utilisez latent_dim=2 pour visualiser directement "
                "l'espace latent 2D."
            )
            return None

        if grid_size is None:
            grid_size = self.config.latent_grid_size

        z_reference = self.encode(x_reference)

        z1_min, z1_max = np.percentile(
            z_reference[:, 0],
            [1, 99],
        )

        z2_min, z2_max = np.percentile(
            z_reference[:, 1],
            [1, 99],
        )

        z1_values = np.linspace(
            z1_min,
            z1_max,
            grid_size,
            dtype=np.float32,
        )

        z2_values = np.linspace(
            z2_max,
            z2_min,
            grid_size,
            dtype=np.float32,
        )

        latent_points = np.array(
            [
                [z1, z2]
                for z2 in z2_values
                for z1 in z1_values
            ],
            dtype=np.float32,
        )

        decoded = self.decode(latent_points)

        canvas = np.zeros(
            (
                self.IMAGE_HEIGHT * grid_size,
                self.IMAGE_WIDTH * grid_size,
            ),
            dtype=np.float32,
        )

        index = 0

        for row in range(grid_size):
            for column in range(grid_size):
                image = self._to_display_image(
                    decoded[index]
                )

                row_start = row * self.IMAGE_HEIGHT
                row_end = row_start + self.IMAGE_HEIGHT

                column_start = (
                        column * self.IMAGE_WIDTH
                )

                column_end = (
                        column_start + self.IMAGE_WIDTH
                )

                canvas[
                    row_start:row_end,
                    column_start:column_end,
                ] = image

                index += 1

        print(
            "\n========== VISUALISATION ESPACE LATENT =========="
        )

        print(
            f"z1 : [{z1_min:.3f}, {z1_max:.3f}]"
        )

        print(
            f"z2 : [{z2_min:.3f}, {z2_max:.3f}]"
        )

        fig = plt.figure(
            figsize=(12, 12)
        )

        plt.imshow(
            canvas,
            cmap="gray",
            extent=[
                z1_min,
                z1_max,
                z2_min,
                z2_max,
            ],
            origin="upper",
            aspect="auto",
        )

        plt.xlabel("Dimension latente z1")
        plt.ylabel("Dimension latente z2")

        plt.title(
            "Visualisation directe de l'espace latent 2D\n"
            "Chaque position (z1, z2) est décodée en image"
        )

        plt.tight_layout()

        path = (
                self.output_dir
                / "04_visualisation_espace_latent.png"
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print("Figure sauvegardée :", path)
        plt.show()

        return latent_points, decoded, fig

    def generate_images(
            self,
            x_reference,
            n_images: Optional[int] = None,
    ):

        if n_images is None:
            n_images = self.config.n_generated_images

        z_reference = self.encode(x_reference)

        latent_mean = np.mean(
            z_reference,
            axis=0,
        )

        latent_covariance = np.cov(
            z_reference,
            rowvar=False,
        )

        latent_covariance = np.atleast_2d(
            latent_covariance
        )

        latent_covariance += (
                1e-6
                * np.eye(self.config.latent_dim)
        )

        rng = np.random.default_rng(
            self.config.seed
        )

        sampled_z = rng.multivariate_normal(
            mean=latent_mean,
            cov=latent_covariance,
            size=n_images,
        ).astype(np.float32)

        generated = self.decode(sampled_z)

        print(
            "\n========== GÉNÉRATION DE NOUVELLES IMAGES =========="
        )

        print("Moyenne latente :", latent_mean)
        print(
            "Shape des points échantillonnés :",
            sampled_z.shape,
        )

        n_cols = int(
            np.ceil(np.sqrt(n_images))
        )

        n_rows = int(
            np.ceil(n_images / n_cols)
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                2 * n_cols,
                2 * n_rows,
            ),
        )

        axes = np.array(
            axes
        ).reshape(-1)

        for index, axis in enumerate(axes):
            axis.axis("off")

            if index < n_images:
                image = self._to_display_image(
                    generated[index]
                )

                axis.imshow(
                    image,
                    cmap="gray",
                )

                latent_text = np.array2string(
                    sampled_z[index],
                    precision=2,
                    suppress_small=True,
                )

                axis.set_title(
                    f"z={latent_text}",
                    fontsize=7,
                )

        fig.suptitle(
            "Images générées par échantillonnage empirique "
            "de l'espace latent"
        )

        fig.tight_layout()

        path = (
                self.output_dir
                / "05_generation_images.png"
        )

        fig.savefig(
            path,
            dpi=150,
            bbox_inches="tight",
        )

        print("Figure sauvegardée :", path)
        plt.show()

        return sampled_z, generated, fig

    def _to_display_image(self, image):
        image = np.asarray(image)

        if self.config.architecture == "dense":
            image = image.reshape(
                self.IMAGE_HEIGHT,
                self.IMAGE_WIDTH,
            )
        else:
            image = np.squeeze(
                image,
                axis=-1,
            )

        if self.config.normalization == "minus_one_one":
            image = (
                            image + 1.0
                    ) / 2.0

        return np.clip(
            image,
            0.0,
            1.0,
        )


# 3. EXPÉRIENCE
def run_experiment(config: AutoencoderConfig):
    autoencoder = SimpleAutoencoder(config)

    autoencoder.print_config()
    autoencoder.show_summaries()

    x_train, y_train, x_test, y_test = (
        autoencoder.load_mnist()
    )

    autoencoder.fit(x_train)

    # Visualisation sur le jeu d'entraînement avant la courbe de loss.
    autoencoder.plot_train_reconstructions(
        x_train,
    )

    autoencoder.evaluate(x_test)
    autoencoder.plot_history()

    # Inspection de quelques points latents + reconstructions.
    autoencoder.inspect_latent_and_reconstructions(
        x_test,
        y_test,
    )

    # Visualisation directe uniquement si latent_dim == 2.
    autoencoder.visualize_latent_space(
        x_train,
    )

    # Génération expérimentale.
    autoencoder.generate_images(
        x_train,
    )

    return autoencoder

# %%
CONFIG = AutoencoderConfig(
    # "dense" ou "conv"
    architecture="dense",
    latent_dim=2,

    # Pour dense
    hidden_dims=(128, 32),

    # Activations
    hidden_activation="relu",
    latent_activation="linear",
    output_activation="sigmoid",

    # Pour conv
    conv_filters=(32, 64),
    conv_strides=(2, 2),
    kernel_size=3,

    # Initialisation
    kernel_initializer="glorot_uniform",

    # Entraînement
    learning_rate=1e-3,
    batch_size=256,
    epochs=200,
    optimizer_name="adam",
    loss_name="mse",

    # Normalisation
    normalization="zero_one",

    # Affichages
    n_examples=10,
    n_generated_images=16,
    latent_grid_size=20,
)

autoencoder = SimpleAutoencoder(CONFIG)
autoencoder.print_config()

# %%
x_train, y_train, x_test, y_test = autoencoder.load_mnist()

print("\nPremière image - shape :", x_train[0].shape)

fig = plt.figure(figsize=(3, 3))
plt.imshow(autoencoder._to_display_image(x_train[0]), cmap="gray")
plt.title(f"Premier exemple MNIST - label {y_train[0]}")
plt.axis("off")
plt.show()

# %%
history = autoencoder.fit(x_train)
# %%
train_reconstructions, _ = autoencoder.plot_train_reconstructions(
    x_train
)

# %% [markdown]
# ## 7. Évaluation et courbe de loss
# 
# %%
test_loss = autoencoder.evaluate(x_test)
autoencoder.plot_history()

# %%
z_examples, reconstructed_examples, _ = (
    autoencoder.inspect_latent_and_reconstructions(
        x_test,
        y_test,
    )
)

# %%
latent_visualization = autoencoder.visualize_latent_space(
    x_train
)

# %% [markdown]
# ## 10. Génération expérimentale de nouvelles images
# 
# Pour un AutoEncodeur simple, la distribution du latent n'est pas régularisée comme dans un VAE.
# 
# Nous estimons donc la moyenne et la covariance des vrais codes latents, puis nous échantillonnons cette distribution empirique.
# 
# %%
sampled_z, generated_images, _ = autoencoder.generate_images(
    x_train
)
