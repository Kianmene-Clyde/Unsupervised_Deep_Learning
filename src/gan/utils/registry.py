"""Registre sûr et atomique des GAN."""

from pathlib import Path

from utils.model_registry import (
    delete_model as _delete_model,
    list_models as _list_models,
    load_model as _load_model,
    model_path as _model_path,
    save_model as _save_model,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(name):
    return _model_path(MODELS_DIR, name)


def list_models():
    return _list_models(MODELS_DIR)


def save_model(name, weights, metadata):
    return _save_model(MODELS_DIR, name, "weights", weights, metadata)


def load_model(name):
    return _load_model(MODELS_DIR, name, "weights")


def delete_model(name):
    return _delete_model(MODELS_DIR, name)
