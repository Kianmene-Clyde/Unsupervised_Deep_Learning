import numpy as np
import pytest

from autoencoder.utils.model import shape_input as ae_shape_input
from gan.utils.model import batch_ranges, generate_batch, init_params, flat_weights, remember
from vae.utils.model import shape_input as vae_shape_input
from utils.array_cache import ArrayFingerprints
from autoencoder.utils.codec import AECode
from pca.utils.codec import PCACode


@pytest.mark.parametrize("shape_input", [ae_shape_input, vae_shape_input])
def test_neural_input_shape_cannot_be_silently_reinterpreted(shape_input):
    dense = {"architecture": "dense"}
    conv = {"architecture": "conv"}
    valid = np.zeros((2, 784), dtype=np.float32)
    assert shape_input(valid, dense).shape == (2, 784)
    assert shape_input(valid, conv).shape == (2, 28, 28, 1)
    with pytest.raises(ValueError, match="incompatible"):
        shape_input(np.zeros((2, 392), dtype=np.float32), dense)


def test_gan_batches_include_the_partial_final_batch():
    assert batch_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_gan_generation_is_repeatable_and_validates_latent_shape():
    cfg = {
        "latent_dim": 3,
        "g_hidden": (4,),
        "d_hidden": (4,),
        "seed": 5,
    }
    generator, discriminator = init_params(cfg, np.random.default_rng(5))
    weights = flat_weights(generator, discriminator)
    remember(weights, cfg, generator, discriminator)
    latent = np.zeros((2, 3), dtype=np.float32)
    np.testing.assert_array_equal(generate_batch(latent, weights), generate_batch(latent, weights))
    with pytest.raises(ValueError, match="Z doit"):
        generate_batch(np.zeros((2, 2), dtype=np.float32), weights)


def test_bound_weight_fingerprint_is_reused_and_prevents_mutation():
    fingerprints = ArrayFingerprints()
    weights = np.arange(8, dtype=np.float32)
    key = fingerprints.bind(weights)
    assert fingerprints.key(weights) == key
    assert not weights.flags.writeable
    with pytest.raises(ValueError):
        weights[0] = 99.0


@pytest.mark.parametrize("code_type", [AECode, PCACode])
def test_hashable_latent_code_owns_its_values_and_respects_signed_zero(code_type):
    source = np.array([1.0, 0.0], dtype=np.float32)
    code = code_type(source)
    initial_hash = hash(code)
    source[0] = 5.0
    assert code.values[0] == 1.0
    assert hash(code) == initial_hash
    equal = code_type(np.array([1.0, -0.0], dtype=np.float32))
    assert code == equal
    assert hash(code) == hash(equal)


def test_vae_epoch_loss_weights_each_observation_equally(monkeypatch):
    import tensorflow as tf
    import vae.utils.model as model

    class Encoder:
        trainable_variables = []

        def __call__(self, X, training=False):
            values = tf.zeros((tf.shape(X)[0], 2))
            return values, values

    class Decoder:
        trainable_variables = []

        def __call__(self, Z, training=False):
            return tf.zeros((tf.shape(Z)[0], 784))

    class NoUpdates:
        def apply_gradients(self, gradients):
            pass

    monkeypatch.setattr(model, "build_models", lambda cfg: (Encoder(), Decoder()))
    monkeypatch.setattr(model, "_optimizer", lambda cfg: NoUpdates())
    X = np.zeros((17, 784), dtype=np.float32)
    X[-1] = 1.0
    cfg = dict(architecture="dense", latent_dim=2, seed=42, learning_rate=0.001,
               epochs=1, batch_size=16, kl_weight=0.001)
    _, _, history = model.train_vae(X, cfg)
    # The fixed zero decoder has exactly 1/17 reconstruction error; a mean of
    # the full-batch and singleton-batch losses would give a different answer.
    assert history["reconstruction_loss"][0] == pytest.approx(1 / 17, abs=1e-7)
    assert history["total_loss"][0] == pytest.approx(1 / 17, abs=1e-7)
