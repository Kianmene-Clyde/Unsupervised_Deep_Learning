import numpy as np
import pytest

from autoencoder.utils.model import shape_input as ae_shape_input
from gan.utils.model import batch_ranges, generate_batch, init_params, flat_weights, remember
from vae.utils.model import shape_input as vae_shape_input
from utils.array_cache import ArrayFingerprints


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
