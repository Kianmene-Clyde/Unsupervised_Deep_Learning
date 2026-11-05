"""Validation shared by the hand-built neural training entry points."""

from __future__ import annotations

import math


def positive_integer(value, name):
    """Accept an integer value without silently truncating a fractional input."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} doit être un entier strictement positif.") from exc
    if not math.isfinite(number) or number < 1 or not number.is_integer():
        raise ValueError(f"{name} doit être un entier strictement positif.")
    return int(number)


def validate_training_config(cfg, *, variational=False):
    """Reject invalid training budgets before building a TensorFlow graph."""
    for name in ("latent_dim", "batch_size", "epochs"):
        positive_integer(cfg[name], name)
    learning_rate = float(cfg["learning_rate"])
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate doit être fini et strictement positif.")
    seed = float(cfg["seed"])
    if not math.isfinite(seed) or seed < 0 or not seed.is_integer():
        raise ValueError("seed doit être un entier positif ou nul.")
    if variational:
        weight = float(cfg["kl_weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("kl_weight doit être fini et positif ou nul.")


def validate_architecture(cfg):
    """Validate the dense/conv shape contract used by AE and VAE."""
    positive_integer(cfg["latent_dim"], "latent_dim")
    architecture = cfg["architecture"]
    if architecture not in {"dense", "conv"}:
        raise ValueError("architecture doit être 'dense' ou 'conv'.")
    if architecture == "dense":
        for units in cfg["hidden_dims"]:
            positive_integer(units, "hidden_dims")
        return
    filters, strides = cfg["conv_filters"], cfg["conv_strides"]
    if not filters or len(filters) != len(strides):
        raise ValueError("conv_filters et conv_strides doivent être non vides et alignés.")
    factor = 1
    for units, stride in zip(filters, strides):
        positive_integer(units, "conv_filters")
        factor *= positive_integer(stride, "conv_strides")
    positive_integer(cfg["kernel_size"], "kernel_size")
    if 28 % factor:
        raise ValueError("Le produit des strides doit diviser 28 pour reconstruire du 28×28.")
