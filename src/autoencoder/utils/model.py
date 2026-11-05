"""Construction, entraînement et (dé)sérialisation des autoencodeurs.

Adapté des constructeurs dense/conv d'autoencodeur.py (le travail notebook de
ce dossier), qui n'est PAS modifié — et surtout jamais importé : exécuté tel
quel, il télécharge MNIST et lance 200 époques d'entraînement. Les mêmes
architectures sont reprises ici en fonctions pures, pilotées par le dict de
config que l'adaptateur assemble depuis les hyperparamètres de l'app.

--- Les poids de l'app sont UN tableau numpy ---

L'interface Algo fait voyager chaque modèle comme un tableau float32
(app.load_active fait np.asarray dessus). Or un autoencodeur est une PAIRE de
modèles Keras. On aplatit donc tous leurs tenseurs en UN vecteur 1-D ;
l'architecture étant entièrement déterminée par la config (stockée dans les
métadonnées du registre), les shapes se retrouvent en reconstruisant les
modèles puis en redécoupant le vecteur — rien d'autre à persister.

--- Le cache ---

encode()/decode() reçoivent les poids mais PAS les métadonnées (signature de
l'interface Algo) : impossible d'y reconstruire l'architecture. remember()
mémorise donc (vecteur → config) au moment où load()/train() — qui ont les
métadonnées — passent par là, et models_of() reconstruit les modèles Keras à
la première demande. Chaque vue de l'app appelle load_active() avant
d'encoder : le cache est toujours chaud.
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
    """Repris d'autoencodeur.py : linear = identité, leaky_relu = pente 0,2."""
    if activation_name in (None, "linear"):
        return x
    if activation_name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(
            negative_slope=LEAKY_RELU_SLOPE, name=layer_name
        )(x)
    return tf.keras.layers.Activation(activation_name, name=layer_name)(x)


def _build_dense(cfg):
    """Encodeur/décodeur denses — le miroir exact d'autoencodeur.py."""
    latent_dim = int(cfg["latent_dim"])
    hidden_dims = tuple(cfg["hidden_dims"])
    act = cfg["hidden_activation"]

    encoder_input = tf.keras.Input(shape=(IMAGE_DIM,), name="image_flat")
    x = encoder_input
    for i, units in enumerate(hidden_dims):
        x = tf.keras.layers.Dense(units, name=f"encoder_dense_{i + 1}")(x)
        x = _apply_activation(x, act, f"encoder_activation_{i + 1}")
    latent = tf.keras.layers.Dense(latent_dim, name="latent_dense")(x)
    latent = _apply_activation(latent, cfg.get("latent_activation", "linear"),
                               "latent_activation")
    encoder = tf.keras.Model(encoder_input, latent, name="dense_encoder")

    decoder_input = tf.keras.Input(shape=(latent_dim,), name="latent_vector")
    x = decoder_input
    for i, units in enumerate(reversed(hidden_dims)):
        x = tf.keras.layers.Dense(units, name=f"decoder_dense_{i + 1}")(x)
        x = _apply_activation(x, act, f"decoder_activation_{i + 1}")
    x = tf.keras.layers.Dense(IMAGE_DIM, name="reconstruction_linear")(x)
    output = _apply_activation(x, cfg["output_activation"], "reconstruction_activation")
    decoder = tf.keras.Model(decoder_input, output, name="dense_decoder")
    return encoder, decoder


def _build_conv(cfg):
    """Encodeur/décodeur convolutionnels — repris d'autoencodeur.py.

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
    latent = tf.keras.layers.Dense(latent_dim, name="latent_dense")(x)
    latent = _apply_activation(latent, cfg.get("latent_activation", "linear"),
                               "latent_activation")
    encoder = tf.keras.Model(encoder_input, latent, name="conv_encoder")

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
    decoder = tf.keras.Model(decoder_input, output, name="conv_decoder")

    if decoder.output_shape != (None, IMG_SIZE, IMG_SIZE, 1):
        raise ValueError(
            f"La config conv ne reconstruit pas du 28×28×1 : {decoder.output_shape}."
        )
    return encoder, decoder


def build_models(cfg):
    """(encoder, decoder) neufs — poids aléatoires — pour cette config."""
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


def train_autoencoder(X, cfg, progress=None):
    """Entraîne un autoencodeur neuf sur X (n, 784). → (encoder, decoder, losses).

    losses : la MSE moyenne par époque (loss « mse » de Keras = moyenne par
    pixel, que l'architecture soit dense ou conv — les deux moyennent sur tous
    les éléments de sortie).
    """
    validate_training_config(cfg)
    seed = int(cfg["seed"])
    tf.keras.utils.set_random_seed(seed)

    encoder, decoder = build_models(cfg)
    inputs = tf.keras.Input(shape=encoder.input_shape[1:], name="autoencoder_input")
    model = tf.keras.Model(inputs, decoder(encoder(inputs)), name="autoencoder")
    model.compile(optimizer=_optimizer(cfg), loss="mse")

    callbacks = []
    if progress is not None:
        epochs = int(cfg["epochs"])
        callbacks.append(tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, logs: progress(
                (e + 1) / epochs,
                desc=f"Époque {e + 1}/{epochs} — loss {logs['loss']:.4f}",
            )
        ))

    Xs = shape_input(X, cfg)
    history = model.fit(
        Xs, Xs,
        batch_size=int(cfg["batch_size"]),
        epochs=int(cfg["epochs"]),
        shuffle=True,
        verbose=0,
        callbacks=callbacks,
    )
    return encoder, decoder, [float(v) for v in history.history["loss"]]


# --------------------------------------------------- Poids ↔ vecteur plat


def flat_weights(encoder, decoder):
    """Tous les tenseurs des deux modèles, aplatis en UN vecteur float32."""
    parts = encoder.get_weights() + decoder.get_weights()
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def models_from_flat(flat, cfg):
    """Reconstruit (encoder, decoder) depuis le vecteur plat et la config.

    Les shapes ne sont pas stockées : on reconstruit l'architecture (mêmes
    couches, donc mêmes shapes, dans le même ordre) et on redécoupe.
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
    """(n, 784) → (n, latent_dim), via l'encodeur du modèle."""
    encoder, _, cfg = models_of(weights)
    Z = encoder(shape_input(X, cfg), training=False)
    return np.asarray(Z, dtype=np.float32)


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
