"""Bounded integration check on SYNTHETIC inputs, not a scientific benchmark.

Run from any working directory with an installed project environment::

    python scripts/smoke_algorithms.py --output output/validation/algorithms

Every registry is redirected to the output directory during the check; no
user model is overwritten. Each algorithm trains once and reconstructs one
input before and after save/reload. GAN reconstruction is approximate inversion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output/validation/algorithms")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Low-dimensional synthetic image combinations provide finite, bounded
    # inputs without downloading a dataset or claiming handwriting quality.
    rng = np.random.default_rng(args.seed)
    basis = rng.uniform(0, 1, (4, 784)).astype(np.float32)
    coefficients = rng.dirichlet(np.ones(4), size=96).astype(np.float32)
    X = coefficients @ basis
    specifications = [
        ("kmeas", "KMeans", dict(k=4, max_iter=3, tolerance=1e-4)),
        ("kohonen", "Kohonen", dict(rows=2, cols=2, n_epochs=1, alpha=0.1, gamma=1.0)),
        ("pca", "PCAAlgo", dict(n_components=3, center="feature")),
        ("autoencoder", "Autoencoder", dict(latent_dim=3, hidden_dims=(8,), epochs=1, batch_size=32)),
        ("vae", "VAE", dict(latent_dim=3, hidden_dims=(8,), epochs=1, batch_size=32)),
        ("gan", "GAN", dict(latent_dim=3, g_hidden=(8,), d_hidden=(8,), epochs=1, batch_size=32)),
    ]
    results = []
    for package, class_name, overrides in specifications:
        started = time.perf_counter()
        result = {"algorithm": package, "status": "failed"}
        registry = None
        original_dir = None
        try:
            module = importlib.import_module(f"{package}.algo")
            registry = importlib.import_module(f"{package}.utils.registry")
            original_dir = registry.MODELS_DIR
            registry.MODELS_DIR = output / "models" / package
            algorithm = getattr(module, class_name)()
            params = {p.name: p.default for p in algorithm.params}
            params.update(overrides, seed=args.seed, n_samples=len(X))
            algorithm.check(params, len(X))
            weights, metadata = algorithm.train(X, params)
            metadata.update(algo=algorithm.key, dataset="synthetic-smoke", n_samples=len(X))
            before = np.asarray(algorithm.decode(algorithm.encode(X[0], weights), weights))
            if before.shape != (784,) or not np.isfinite(before).all():
                raise AssertionError("Reconstruction invalide avant sauvegarde.")
            model_name = f"smoke_seed{args.seed}"
            algorithm.save(model_name, weights, metadata)
            loaded, loaded_meta = algorithm.load(model_name)
            after = np.asarray(algorithm.decode(algorithm.encode(X[0], loaded), loaded))
            np.testing.assert_allclose(after, before, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(loaded, weights)
            assert loaded_meta["dataset"] == "synthetic-smoke"
            result.update(status="passed", model_bytes=algorithm.model_path(model_name).stat().st_size,
                          reconstruction_mse=float(np.mean((X[0] - after) ** 2)),
                          capability="approximate_inversion" if package == "gan" else "native_codec")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if registry is not None and original_dir is not None:
                registry.MODELS_DIR = original_dir
        result["elapsed_s"] = time.perf_counter() - started
        results.append(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)
    report = dict(purpose="synthetic integration smoke; not scientific performance evidence",
                  seed=args.seed, n_samples=len(X), python=platform.python_version(), results=results)
    (output / "smoke_algorithms.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return 0 if all(r["status"] == "passed" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
