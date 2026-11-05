"""Construction, entraînement et (dé)sérialisation des VAE.

Repris de src/autoencoder/utils/model.py, avec ce qui fait un VAE :

- l'encodeur a DEUX têtes de sortie, z_mean et z_log_var (les paramètres de la
  gaussienne latente), au lieu d'un point latent unique ;
- l'entraînement est une boucle GradientTape maison (reprise de VAE.py) : à
  chaque pas, reparameterization trick z = μ + σ·ε puis
  Loss = reconstruction + kl_weight · KL(N(μ, σ) ‖ N(0, 1)) ;
- pour la compression et la projection, le code d'une image est z_mean — le
  point latent DÉTERMINISTE, sans le bruit ε qui n'a de sens qu'à
  l'entraînement.

Le reste — poids aplatis en UN vecteur float32, cache (vecteur → config →
modèles Keras) — est identique à l'autoencodeur : l'app fait voyager chaque
modèle comme un tableau numpy, l'architecture vivant dans les métadonnées.
"""

import numpy as np
import tensorflow as tf

from utils.array_cache import ArrayFingerprints
from utils.neural_config import validate_architecture, validate_training_config

IMG_SIZE = 28
IMAGE_DIM = IMG_SIZE * IMG_SIZE
LEAKY_RELU_SLOPE = 0.2

# vecteur.tobytes() -> config : de quoi reconstruire l'architecture (léger).
_KNOWN = {}
_KNOWN_MAX = 64
# vecteur.tobytes() -> (encoder, decoder) : les modèles Keras, eux, sont lourds.
_BUILT = {}
_BUILT_MAX = 4
_FINGERPRINTS = ArrayFingerprints()


# ----------------------------------------------------------- Architecture


def _apply_activation(x, activation_name, layer_name):
    """Repris de VAE.py : linear = identité, leaky_relu = pente 0,2."""
    if activation_name in (None, "linear"):
        return x
    if activation_name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(
            negative_slope=LEAKY_RELU_SLOPE, name=layer_name
        )(x)
    return tf.keras.layers.Activation(activation_name, name=layer_name)(x)


def _build_dense(cfg):
    """Encodeur (→ z_mean, z_log_var) / décodeur denses — le miroir de VAE.py."""
    latent_dim = int(cfg["latent_dim"])
    hidden_dims = tuple(cfg["hidden_dims"])
    act = cfg["hidden_activation"]

    encoder_input = tf.keras.Input(shape=(IMAGE_DIM,), name="image_flat")
    x = encoder_input
    for i, units in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(units, name=f"encoder_dense_{i + 1}")(x)
        x = _apply_activation(x, act, f"encoder_activation_{i + 1}")
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    # L'activation latente ne porte que sur z_mean (le code déterministe) :
    # z_log_var est une log-variance, elle doit rester libre sur tout ℝ, sinon
    # la KL divergence n'a plus de sens.
    z_mean = _apply_activation(z_mean, cfg.get("latent_activation", "linear"),
                               "z_mean_activation")
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var],
                             name="vae_dense_encoder")

    decoder_input = tf.keras.Input(shape=(latent_dim,), name="latent_vector")
    x = decoder_input
    for i, units in enumerate(reversed(hidden_dims)):
        x = tf.keras.layers.Dense(units, name=f"decoder_dense_{i + 1}")(x)
        x = _apply_activation(x, act, f"decoder_activation_{i + 1}")
    x = tf.keras.layers.Dense(IMAGE_DIM, name="reconstruction_linear")(x)
    output = _apply_activation(x, cfg["output_activation"], "reconstruction_activation")
    decoder = tf.keras.Model(decoder_input, output, name="vae_dense_decoder")
    return encoder, decoder


def _build_conv(cfg):
    """Encodeur/décodeur convolutionnels — repris de VAE.py.

    Les strides valent 2 par couche (presets de l'app) : leur produit divise 28,
    la contrainte que _validate_config vérifiait dans le notebook.
    """
    latent_dim = int(cfg["latent_dim"])
    filters = tuple(cfg["conv_filters"])
    strides = tuple(cfg["conv_strides"])
    kernel = int(cfg["kernel_size"])
    act = cfg["hidden_activation"]

    encoder_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image")
    x = encoder_input
    for i, (f, s) in enumerate(zip(filters, strides)):
        x = tf.keras.layers.Conv2D(
            f, kernel, strides=s, padding="same", name=f"encoder_conv_{i + 1}"
        )(x)
        x = _apply_activation(x, act, f"encoder_conv_activation_{i + 1}")
    shape_before_flatten = tuple(int(d) for d in x.shape[1:])
    x = tf.keras.layers.Flatten(name="encoder_flatten")(x)
    z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
    # Activation latente sur z_mean seul — z_log_var reste libre sur ℝ (voir dense).
    z_mean = _apply_activation(z_mean, cfg.get("latent_activation", "linear"),
                               "z_mean_activation")
    z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
    encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var],
                             name="vae_conv_encoder")

    decoder_input = tf.keras.Input(shape=(latent_dim,), name="latent_vector")
    x = tf.keras.layers.Dense(
        int(np.prod(shape_before_flatten)), name="decoder_dense_projection"
    )(decoder_input)
    x = _apply_activation(x, act, "decoder_projection_activation")
    x = tf.keras.layers.Reshape(shape_before_flatten, name="decoder_reshape")(x)
    for j in range(len(filters) - 1, 0, -1):
        x = tf.keras.layers.Conv2DTranspose(
            filters[j - 1], kernel, strides=strides[j], padding="same",
            name=f"decoder_deconv_{j}",
        )(x)
        x = _apply_activation(x, act, f"decoder_deconv_activation_{j}")
    x = tf.keras.layers.Conv2DTranspose(
        1, kernel, strides=strides[0], padding="same", name="reconstruction_conv"
    )(x)
    output = _apply_activation(x, cfg["output_activation"], "reconstruction_activation")
    decoder = tf.keras.Model(decoder_input, output, name="vae_conv_decoder")

    if decoder.output_shape != (None, IMG_SIZE, IMG_SIZE, 1):
        raise ValueError(
            f"La config conv ne reconstruit pas du 28×28×1 : {decoder.output_shape}."
        )
    return encoder, decoder


def build_models(cfg):
    """(encoder, decoder) neufs — poids aléatoires — pour cette config.

    L'encodeur a deux sorties : encoder(x) -> [z_mean, z_log_var].
    """
    validate_architecture(cfg)
    if cfg["architecture"] == "conv":
        return _build_conv(cfg)
    return _build_dense(cfg)


def _optimizer(cfg):
    name = str(cfg["optimizer"]).lower()
    lr = float(cfg["learning_rate"])
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    raise ValueError("optimizer doit être 'adam', 'sgd' ou 'rmsprop'.")


def shape_input(X, cfg):
    """Met les images (n, 784) de l'app dans la forme d'entrée de l'encodeur."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.size == 0 or not np.isfinite(X).all():
        raise ValueError("X doit être non vide et ne contenir que des valeurs finies.")
    if cfg["architecture"] == "conv":
        if X.ndim == 2 and X.shape[1] == IMAGE_DIM:
            return X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        if X.ndim == 3 and X.shape[1:] == (IMG_SIZE, IMG_SIZE):
            return X[..., None]
        if X.ndim == 4 and X.shape[1:] == (IMG_SIZE, IMG_SIZE, 1):
            return X
    elif X.ndim == 2 and X.shape[1] == IMAGE_DIM:
        return X
    raise ValueError(
        f"X a une forme incompatible avec l'architecture {cfg['architecture']!r} : {X.shape}."
    )


# ------------------------------------------------------------ Entraînement


def train_vae(X, cfg, progress=None):
    """Entraîne un VAE neuf sur X (n, 784). → (encoder, decoder, history).

    Boucle GradientTape maison reprise de VAE.py : reparameterization trick puis
    Loss = reconstruction + kl_weight · KL. history est un dict de trois listes
    (une valeur par époque) : "total_loss", "reconstruction_loss", "kl_loss".

    La loss de reconstruction est la « mse » de Keras : la moyenne par élément de
    sortie — donc la MSE par pixel, dense comme conv (les deux moyennent sur tous
    les éléments), la même unité que l'autoencodeur simple.
    """
    validate_training_config(cfg, variational=True)
    seed = int(cfg["seed"])
    tf.keras.utils.set_random_seed(seed)

    encoder, decoder = build_models(cfg)
    optimizer = _optimizer(cfg)
    loss_fn = tf.keras.losses.get(cfg.get("loss_name", "mse"))
    kl_weight = float(cfg["kl_weight"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])

    Xs = shape_input(X, cfg)
    dataset = (
        tf.data.Dataset.from_tensor_slices(Xs)
        .shuffle(buffer_size=len(Xs), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    @tf.function
    def train_step(x_batch):
        with tf.GradientTape() as tape:
            z_mean, z_log_var = encoder(x_batch, training=True)
            # Reparameterization trick : z = μ + exp(½·log σ²)·ε, ε ~ N(0, 1).
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            reconstruction = decoder(z, training=True)

            recon = tf.reduce_mean(loss_fn(x_batch, reconstruction))
            # KL(N(μ, σ²) ‖ N(0, 1)), sommée sur les dimensions latentes.
            kl = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
            )
            kl = tf.reduce_mean(kl)
            total = recon + kl_weight * kl
            tf.debugging.assert_all_finite(
                total, "Loss VAE non finie : réduire le pas ou vérifier les données."
            )

        variables = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(total, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return total, recon, kl

    history = {"total_loss": [], "reconstruction_loss": [], "kl_loss": []}
    for epoch in range(epochs):
        total_m = tf.keras.metrics.Mean()
        recon_m = tf.keras.metrics.Mean()
        kl_m = tf.keras.metrics.Mean()
        for x_batch in dataset:
            total, recon, kl = train_step(x_batch)
            # Epoch statistics are sample means, not means of batch means.
            # The final partial batch must not count as a full batch.
            count = tf.cast(tf.shape(x_batch)[0], tf.float32)
            total_m.update_state(total, sample_weight=count)
            recon_m.update_state(recon, sample_weight=count)
            kl_m.update_state(kl, sample_weight=count)
        history["total_loss"].append(float(total_m.result().numpy()))
        history["reconstruction_loss"].append(float(recon_m.result().numpy()))
        history["kl_loss"].append(float(kl_m.result().numpy()))
        if progress is not None:
            progress(
                (epoch + 1) / epochs,
                desc=f"Époque {epoch + 1}/{epochs} — loss "
                     f"{history['total_loss'][-1]:.4f}",
            )
    return encoder, decoder, history


# --------------------------------------------------- Poids ↔ vecteur plat


def flat_weights(encoder, decoder):
    """Tous les tenseurs des deux modèles, aplatis en UN vecteur float32."""
    parts = encoder.get_weights() + decoder.get_weights()
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def models_from_flat(flat, cfg):
    """Reconstruit (encoder, decoder) depuis le vecteur plat et la config.

    Les shapes ne sont pas stockées : on reconstruit l'architecture (mêmes
    couches, donc mêmes shapes, dans le même ordre) et on redécoupe. Les deux
    têtes z_mean/z_log_var font juste deux blocs de poids de plus, dans l'ordre
    où build_models les crée — donc parfaitement déterministe.
    """
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Le vecteur de poids doit être non vide et fini.")
    encoder, decoder = build_models(cfg)
    expected = sum(w.size for model in (encoder, decoder) for w in model.get_weights())
    if flat.size != expected:
        raise ValueError(
            f"Poids incompatibles avec cette architecture : {flat.size} valeurs "
            f"pour {expected} attendues. Le modèle vient-il d'une autre config ?"
        )
    offset = 0
    for model in (encoder, decoder):
        chunks = []
        for w in model.get_weights():
            size = int(np.prod(w.shape)) if w.shape else 1
            chunks.append(flat[offset:offset + size].reshape(w.shape).astype(w.dtype))
            offset += size
        model.set_weights(chunks)
    if offset != flat.size:
        raise ValueError(
            f"Poids incompatibles avec cette architecture : {flat.size} valeurs "
            f"pour {offset} attendues. Le modèle vient-il d'une autre config ?"
        )
    return encoder, decoder


# ------------------------------------------------------------------ Cache


def _key_of(weights):
    return _FINGERPRINTS.key(weights)


def remember(weights, cfg, encoder=None, decoder=None):
    """Associe un vecteur de poids à sa config (appelé par load/train).

    Les modèles Keras sont gardés seulement s'ils existent déjà (train vient de
    les produire) : la simple lecture des métadonnées d'un registre entier ne
    doit pas construire un graphe TF par modèle listé.
    """
    key = _FINGERPRINTS.bind(weights)
    _KNOWN[key] = cfg
    while len(_KNOWN) > _KNOWN_MAX:
        _KNOWN.pop(next(iter(_KNOWN)))
    if encoder is not None:
        _BUILT[key] = (encoder, decoder)
        while len(_BUILT) > _BUILT_MAX:
            _BUILT.pop(next(iter(_BUILT)))


def models_of(weights):
    """(encoder, decoder, cfg) du vecteur de poids — reconstruits au besoin."""
    key = _key_of(weights)
    cfg = _KNOWN.get(key)
    if cfg is None:
        raise RuntimeError(
            "Poids inconnus du cache : recharge le modèle (algo.load) avant "
            "d'encoder — l'architecture vit dans ses métadonnées."
        )
    if key not in _BUILT:
        encoder, decoder = models_from_flat(weights, cfg)
        _BUILT[key] = (encoder, decoder)
        while len(_BUILT) > _BUILT_MAX:
            _BUILT.pop(next(iter(_BUILT)))
    encoder, decoder = _BUILT[key]
    return encoder, decoder, cfg


# ------------------------------------------------------ Encode / décode


def encode_batch(X, weights):
    """(n, 784) → (n, latent_dim), le z_mean de chaque image.

    On rend z_mean, PAS un z échantillonné : le bruit ε du reparameterization
    trick n'a de sens qu'à l'entraînement. Pour compresser et projeter, le point
    latent déterministe est le bon représentant — deux appels sur la même image
    doivent donner le même code.
    """
    encoder, _, cfg = models_of(weights)
    out = encoder(shape_input(X, cfg), training=False)
    z_mean = out[0] if isinstance(out, (list, tuple)) else out
    return np.asarray(z_mean, dtype=np.float32)


def decode_batch(Z, weights):
    """(m, latent_dim) → (m, 784) dans [0, 1], via le décodeur du modèle."""
    _, decoder, cfg = models_of(weights)
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 1:
        Z = Z[None, :]
    if (Z.ndim != 2 or Z.shape[0] == 0
            or Z.shape[1] != int(cfg["latent_dim"]) or not np.isfinite(Z).all()):
        raise ValueError(
            f"Z doit être une matrice finie (n, {int(cfg['latent_dim'])}), reçu : {Z.shape}."
        )
    out = np.asarray(decoder(Z, training=False), dtype=np.float32)
    return np.clip(out.reshape(len(Z), IMAGE_DIM), 0.0, 1.0)
