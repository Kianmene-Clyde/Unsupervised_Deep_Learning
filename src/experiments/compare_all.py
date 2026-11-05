"""Comparaison reproductible des six méthodes, sans sélection sur le test.

Les configurations sont fixées avant évaluation. Une validation stratifiée est
réservée dans le train ; --evaluation-split validation permet l'exploration et
--evaluation-split test doit être réservé au bilan final. FD-PCA utilise une PCA
ajustée sur le train uniquement : ce n'est pas la FID Inception standard.
Les budgets théoriques et les charges NPZ réellement sérialisées sont séparés.
Les silhouettes de clusters en pixels et de classes dans le latent ne sont pas
interchangeables. Le GAN n'a pas d'encodeur natif ; son inversion est diagnostique.

Depuis la racine :
    python src/experiments/compare_all.py --quick --epochs 1 --datasets mnist
    python src/experiments/compare_all.py --seeds 42 43 44 --datasets mnist

Les sorties par défaut sont horodatées pour préserver les résultats historiques.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import import load_dataset, split_train_validation  # noqa: E402
from utils.metrics_common import (  # noqa: E402
    compression_ratio, fit_pca_features, frechet_distance, frechet_from_images,
    generation_diagnostics, mse, pca_features, psnr, purity, serialized_storage,
    silhouette, ssim,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "comparison_runs"
SEED = 42
N_TRAIN, N_TEST, N_GEN, N_PANEL = 10000, 2000, 2000, 8


def balanced_subset(X, y, n, seed):
    """Deterministic stratification; labels affect sampling, not model fitting."""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X doit être 2D et y 1D, de même longueur.")
    if not isinstance(n, (int, np.integer)) or not 1 <= n <= len(X):
        raise ValueError(f"n doit être un entier entre 1 et {len(X)}, reçu : {n}.")
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    base, remainder = divmod(int(n), len(classes))
    selected, reserve = [], []
    for position, label in enumerate(classes):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        take = min(len(indices), base + (position < remainder))
        selected.extend(indices[:take])
        reserve.extend(indices[take:])
    if len(selected) < n:
        rng.shuffle(reserve)
        selected.extend(reserve[:n - len(selected)])
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return X[selected], y[selected]


def build_fd_context(X_train, X_reference, n_components=50):
    """Freeze one train-only PCA shared by all algorithms and model seeds."""
    dimensions = min(n_components, np.shape(X_train)[1], len(X_train) - 1)
    featurizer = fit_pca_features(X_train, n_components=dimensions)
    return featurizer, pca_features(X_reference, featurizer)


def generation_fid(X_real, X_generated, context=None):
    """Compatibility alias for FD-PCA (never standard Inception FID).

    The benchmark always supplies a train-fitted context. Without one, this
    legacy convenience function fits on X_real and is exploratory only.
    """
    if context is None:
        dimensions = min(50, np.shape(X_real)[1], len(X_real) - 1)
        return frechet_from_images(X_real, X_generated, n_components=dimensions)
    featurizer, real_features = context
    return frechet_distance(real_features, pca_features(X_generated, featurizer))


def _result(name, train_s, Xtr, Xte, yte, X_rec, X_gen, codes, model_arrays,
            metadata, bits, *, seed, fid_context, cluster_labels=None,
            latent_codes=None, reconstruction_mode="native_codec"):
    """Uniform measurements, with task-specific quantities kept separate."""
    row = {
        "algo": name, "train_s": train_s, "seed": seed,
        "theoretical_code_bits": bits,
        "theoretical_ratio": compression_ratio(bits),
        "reconstruction_mode": reconstruction_mode,
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": purity(cluster_labels, yte) if cluster_labels is not None else float("nan"),
        "silhouette_cluster_pixels": silhouette(Xte, cluster_labels, seed=seed)
        if cluster_labels is not None else float("nan"),
        "silhouette_class_latent": silhouette(latent_codes, yte, seed=seed)
        if latent_codes is not None else float("nan"),
        "fd_pca": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
        "training_config": metadata,
    }
    row.update(serialized_storage(codes, model_arrays, metadata))
    row.update(generation_diagnostics(Xtr, Xte, X_gen, seed=seed))
    return row


def run_kmeans(Xtr, ytr, Xte, yte, k=64, quick=False, fid_context=None,
               seed=SEED, n_gen=N_GEN, epochs=None):
    from kmeas.utils.kmeas import assign_clusters, compute_squared_distances, fit_kmeans
    from kmeas.utils.generation import cluster_stats, sample_synthetic

    k = min(k, len(Xtr))
    config = {"k": k, "max_iter": 30 if quick else 100, "tolerance": 1e-4, "seed": seed}
    t0 = time.perf_counter()
    centroids, labels_tr, _ = fit_kmeans(X=Xtr, **config, verbose=False, return_history=True)
    centroids = np.asarray(centroids, dtype=np.float32)
    props, sigmas = cluster_stats(Xtr, np.asarray(labels_tr), k)
    train_s = time.perf_counter() - t0
    labels_te = np.asarray(assign_clusters(compute_squared_distances(Xte, centroids)))
    X_rec = centroids[labels_te]
    X_gen, _ = sample_synthetic(centroids, props, sigmas, n_gen, seed=seed)
    return _result(f"K-means (k={k})", train_s, Xtr, Xte, yte, X_rec, X_gen,
                   labels_te.astype(np.uint8 if k <= 256 else np.uint32),
                   {"centroids": centroids}, config, int(np.ceil(np.log2(k))),
                   seed=seed, fid_context=fid_context, cluster_labels=labels_te)


def run_kohonen(Xtr, ytr, Xte, yte, rows=8, cols=8, quick=False, fid_context=None,
                seed=SEED, n_gen=N_GEN, epochs=None):
    from kohonen.utils.kohonen import assign_clusters, fit_kohonen
    from kohonen.utils.generation import activation_weights, sample_synthetic

    config = {"rows": rows, "cols": cols, "n_epochs": epochs or (2 if quick else 8),
              "alpha": .1, "gamma": 1., "seed": seed}
    t0 = time.perf_counter()
    weights, labels_tr, _ = fit_kohonen(X=Xtr, **config, verbose=False, return_history=True)
    weights = np.asarray(weights, dtype=np.float32)
    act = activation_weights(np.asarray(labels_tr), rows * cols)
    train_s = time.perf_counter() - t0
    labels_te = np.asarray(assign_clusters(Xte, weights))
    X_rec = weights[labels_te]
    X_gen, _ = sample_synthetic(weights, rows, cols, act, n_gen, seed=seed)
    return _result(f"Kohonen ({rows}x{cols})", train_s, Xtr, Xte, yte, X_rec, X_gen,
                   labels_te.astype(np.uint8 if rows * cols <= 256 else np.uint32),
                   {"weights": weights}, config, int(np.ceil(np.log2(rows * cols))),
                   seed=seed, fid_context=fid_context, cluster_labels=labels_te)


def run_pca(Xtr, ytr, Xte, yte, d=16, quick=False, fid_context=None,
            seed=SEED, n_gen=N_GEN, epochs=None):
    from utils.pca.dim_reduction import fit_pca
    from utils.pca.compression import reconstruct
    from utils.pca.generation import fit_latent_gaussian, generate, sample_latent

    d = min(d, Xtr.shape[1], len(Xtr) - 1)
    t0 = time.perf_counter()
    pca = fit_pca(Xtr, n_components=d)
    mean_z, cov_z = fit_latent_gaussian(pca, Xtr)
    train_s = time.perf_counter() - t0
    # Actual serialized codes are float32, so reconstruction measures that codec.
    codes = np.asarray(pca.transform(Xte), dtype=np.float32)
    X_rec = np.asarray(reconstruct(pca, codes), dtype=np.float32)
    X_gen = np.clip(np.asarray(generate(pca, sample_latent(mean_z, cov_z, n_gen, seed=seed)),
                               dtype=np.float32), 0, 1)
    return _result(f"PCA (d={d})", train_s, Xtr, Xte, yte, X_rec, X_gen, codes,
                   {"mean": pca.mean_, "components": pca.components_},
                   {"n_components": d, "generation": "unconditional_train_gaussian"}, d * 32,
                   seed=seed, fid_context=fid_context, latent_codes=codes)


def _run_neural(kind, Xtr, ytr, Xte, yte, z, quick, fid_context, seed, n_gen, epochs):
    z = min(z, len(Xtr) - 1)
    if kind == "autoencoder":
        from autoencoder.algo import Autoencoder as Algorithm
        from autoencoder.utils.model import decode_batch, encode_batch
    else:
        from vae.algo import VAE as Algorithm
        from vae.utils.model import decode_batch, encode_batch
    config = {"architecture": "dense", "latent_dim": z, "hidden_dims": "128-32",
              "hidden_activation": "relu", "latent_activation": "linear",
              "output_activation": "sigmoid", "optimizer": "adam", "learning_rate": 1e-3,
              "batch_size": 256, "epochs": epochs or (5 if quick else 20), "seed": seed}
    if kind == "vae":
        config["kl_weight"] = 1e-3
    t0 = time.perf_counter()
    weights, meta = Algorithm().train(Xtr, config)
    if kind == "autoencoder":
        train_codes = encode_batch(Xtr, weights)
        covariance = np.atleast_2d(np.cov(train_codes, rowvar=False)) + 1e-6 * np.eye(z)
    train_s = time.perf_counter() - t0
    codes = np.asarray(encode_batch(Xte, weights), dtype=np.float32)
    X_rec = decode_batch(codes, weights)
    rng = np.random.default_rng(seed)
    if kind == "autoencoder":
        Z_gen = rng.multivariate_normal(train_codes.mean(axis=0), covariance, size=n_gen)
    else:
        Z_gen = rng.standard_normal((n_gen, z))
    X_gen = decode_batch(Z_gen.astype(np.float32), weights)
    return _result(f"{'AutoEncoder' if kind == 'autoencoder' else 'VAE'} (z={z})",
                   train_s, Xtr, Xte, yte, X_rec, X_gen, codes, {"weights": weights},
                   meta, z * 32, seed=seed, fid_context=fid_context, latent_codes=codes)


def run_autoencoder(Xtr, ytr, Xte, yte, z=16, quick=False, fid_context=None,
                    seed=SEED, n_gen=N_GEN, epochs=None):
    return _run_neural("autoencoder", Xtr, ytr, Xte, yte, z, quick, fid_context, seed, n_gen, epochs)


def run_vae(Xtr, ytr, Xte, yte, z=16, quick=False, fid_context=None,
            seed=SEED, n_gen=N_GEN, epochs=None):
    return _run_neural("vae", Xtr, ytr, Xte, yte, z, quick, fid_context, seed, n_gen, epochs)


def run_gan(Xtr, ytr, Xte, yte, ds_key, quick=False, reuse_model=False,
            fid_context=None, seed=SEED, n_gen=N_GEN, epochs=None):
    from gan.algo import GAN
    from gan.utils.model import decode_batch, encode_batch, generate_batch
    from gan.utils.registry import list_models, load_model

    algo, eligible = GAN(), []
    if reuse_model:
        for name in list_models():
            try:
                if load_model(name)[1].get("dataset") == ds_key:
                    eligible.append(name)
            except (OSError, ValueError) as error:
                print(f"    Modèle GAN ignoré {name!r}: {error}")
    if eligible:
        weights, meta = algo.load(eligible[-1])
        train_s, source = float("nan"), f"registry:{eligible[-1]}"
        print("    ATTENTION: GAN externe au protocole, comparaison non contrôlée.")
    else:
        config = {"latent_dim": 64, "g_hidden": "256-512", "d_hidden": "512-256",
                  "learning_rate": 2e-4, "label_smoothing": .9, "batch_size": 128,
                  "epochs": epochs or (5 if quick else 40), "seed": seed}
        t0 = time.perf_counter()
        weights, meta = algo.train(Xtr, config)
        train_s, source = time.perf_counter() - t0, "trained_in_protocol"
    z = int(meta["latent_dim"])
    rng = np.random.default_rng(seed)
    X_gen = generate_batch(rng.standard_normal((n_gen, z)).astype(np.float32), weights)
    codes = np.asarray(encode_batch(Xte, weights), dtype=np.float32)
    X_rec = decode_batch(codes, weights)
    result = _result(f"GAN (z={z}, inversion diagnostique)", train_s, Xtr, Xte, yte,
                     X_rec, X_gen, codes, {"weights": weights}, meta, z * 32,
                     seed=seed, fid_context=fid_context, latent_codes=codes,
                     reconstruction_mode="bank_inversion_not_native_codec")
    result["model_source"] = source
    return result


def save_panel(rows, originals, fname, title):
    """Une ligne d'images par algo (+ la ligne des originaux si fournie)."""
    all_rows = ([("Originales (test)", originals)] if originals is not None else [])
    all_rows += rows
    n_r, n_c = len(all_rows), N_PANEL
    fig, axes = plt.subplots(n_r, n_c, figsize=(1.35 * n_c, 1.45 * n_r),
                             squeeze=False)
    for i, (label, images) in enumerate(all_rows):
        for j in range(n_c):
            ax = axes[i][j]
            ax.imshow(np.asarray(images[j]).reshape(28, 28), cmap="gray",
                      vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if j == 0:
                ax.set_ylabel(label, fontsize=8, fontweight="bold",
                              rotation=0, ha="right", va="center")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    panneau : {fname}")


COLUMNS = [
    "algo", "seed", "model_source", "reconstruction_mode", "train_s", "evaluation_s",
    "theoretical_code_bits", "theoretical_ratio", "code_npz_bytes", "model_npz_bytes",
    "amortized_npz_bits_per_image", "mse", "psnr", "ssim", "purity",
    "silhouette_cluster_pixels", "silhouette_class_latent", "fd_pca",
    "gen_unique_fraction", "gen_pixel_variance", "gen_nearest_train_mse",
    "gen_nearest_reference_mse", "diagnostic_n_generated", "diagnostic_n_train",
    "diagnostic_n_reference",
]


def fmt(value, digits=4):
    if isinstance(value, (float, np.floating)):
        return "—" if np.isnan(value) else f"{value:.{digits}g}"
    return str(value)


def _fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _versions():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "tensorflow", "keras", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def _source_fingerprints():
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "src").rglob("*.py")) + sorted(root.glob("requirements*.txt"))
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources}


def summarize_runs(results):
    """Per-model mean and sample SD; one seed cannot estimate dispersion."""
    summary = []
    for name in dict.fromkeys(row["algo"] for row in results):
        rows = [row for row in results if row["algo"] == name]
        for metric in COLUMNS:
            values = [row[metric] for row in rows if isinstance(row.get(metric), (int, float, np.number))
                      and np.isfinite(row[metric])]
            if metric in {"seed"} or not values:
                continue
            summary.append({"algo": name, "metric": metric, "n_finite": len(values),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")})
    return summary


def _write_markdown(path, rows, evaluation_split):
    lines = [f"# Comparaison sur {evaluation_split}", "",
             "Configurations fixées ; labels utilisés pour stratification et évaluation uniquement.",
             "FD-PCA : PCA apprise sur train, partagée et gelée ; ce n'est pas la FID standard.",
             "Les deux silhouettes ci-dessous ont des définitions différentes et ne se classent pas ensemble.",
             "", "| Méthode | Seed | MSE ↓ | PSNR ↑ | SSIM ↑ | Pureté ↑ | Silh. clusters/pixels | Silh. classes/latent | FD-PCA ↓ |",
             "|" + "---|" * 9]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in
                     ("algo", "seed", "mse", "psnr", "ssim", "purity",
                      "silhouette_cluster_pixels", "silhouette_class_latent", "fd_pca")) + " |")
    lines += ["", "## Stockage mesuré", "",
              "Code théorique : index idéalement compacté ou flottants float32. NPZ : octets réellement",
              "sérialisés en mémoire, en-têtes inclus ; modèle ajusté complet et métadonnées inclus dans",
              "le coût amorti sur ce lot d'évaluation. L'environnement logiciel est exclu.",
              "Le GAN fournit une inversion approximative diagnostique, pas un codec natif.", "",
              "| Méthode | Seed | Mode reconstruction | Bits théoriques/image | NPZ codes (octets) | NPZ modèle (octets) | Bits NPZ amortis/image |",
              "|" + "---|" * 7]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[key]) for key in
                     ("algo", "seed", "reconstruction_mode", "theoretical_code_bits", "code_npz_bytes",
                      "model_npz_bytes", "amortized_npz_bits_per_image")) + " |")
    lines += ["", "Les diagnostics de diversité/voisinage et leurs tailles d'échantillon figurent dans le CSV.",
              "Ils détectent des signaux grossiers ; ils ne prouvent ni qualité perceptuelle ni absence de mémorisation.",
              "Avec une seule seed, aucune estimation de dispersion n'est disponible."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparaison scientifique des six méthodes.")
    parser.add_argument("--datasets", nargs="+", choices=["mnist", "quickdraw", "quickdraw10"],
                        default=["mnist", "quickdraw"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-test", type=int, help="Nombre d'images du split d'évaluation choisi.")
    parser.add_argument("--n-gen", type=int)
    parser.add_argument("--epochs", type=int, help="Époques AE/VAE/GAN/SOM ; sans effet sur PCA/K-means.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    parser.add_argument("--data-seed", type=int, default=SEED)
    parser.add_argument("--validation-fraction", type=float, default=.2)
    parser.add_argument("--evaluation-split", choices=["validation", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--algorithms", nargs="+",
                        choices=["kmeans", "kohonen", "pca", "autoencoder", "vae", "gan"],
                        default=["kmeans", "kohonen", "pca", "autoencoder", "vae", "gan"])
    parser.add_argument("--reuse-gan", action="store_true",
                        help="GAN externe : sa provenance sera signalée comme non contrôlée.")
    args = parser.parse_args(argv)
    n_train = args.n_train if args.n_train is not None else (1000 if args.quick else N_TRAIN)
    n_test = args.n_test if args.n_test is not None else (256 if args.quick else N_TEST)
    n_gen = args.n_gen if args.n_gen is not None else (256 if args.quick else N_GEN)
    if min(n_train, n_test, n_gen) < N_PANEL:
        parser.error(f"n-train, n-test et n-gen doivent valoir au moins {N_PANEL}.")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs doit être strictement positif.")
    if min(args.seeds + [args.data_seed]) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Les seeds doivent être positives ou nulles et sans doublon.")
    if not 0 < args.validation_fraction < 1:
        parser.error("validation-fraction doit être comprise strictement entre 0 et 1.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (args.output_dir or RESULTS_DIR / timestamp).resolve()
    # Explicit output paths are allowed, but existing scientific results are never overwritten.
    if output_dir.exists() and any(output_dir.glob("results_*.csv")):
        parser.error(f"Résultats déjà présents dans {output_dir} ; choisir un nouveau dossier.")
    output_dir.mkdir(parents=True, exist_ok=True)
    runners = {"kmeans": run_kmeans, "kohonen": run_kohonen, "pca": run_pca,
               "autoencoder": run_autoencoder, "vae": run_vae, "gan": run_gan}
    for ds_key in args.datasets:
        print(f"\n=== {ds_key} ({args.evaluation_split}) ===", flush=True)
        ds = load_dataset(ds_key)
        pool_X, pool_y, val_X, val_y = split_train_validation(
            ds.X_train, ds.y_train, validation_fraction=args.validation_fraction, seed=args.data_seed)
        Xtr, ytr = balanced_subset(pool_X, pool_y, n_train, args.data_seed)
        reference_X, reference_y = ((val_X, val_y) if args.evaluation_split == "validation"
                                    else (ds.X_test, ds.y_test))
        Xte, yte = balanced_subset(reference_X, reference_y, n_test, args.data_seed + 1)
        fd_context = build_fd_context(Xtr, Xte)
        np.savez_compressed(output_dir / f"fd_pca_features_{ds_key}.npz",
                            mean=fd_context[0][0], components=fd_context[0][1])
        print(f"  {len(Xtr)} entraînement / {len(val_X)} validation réservée / {len(Xte)} évaluation")
        results = []
        for seed in args.seeds:
            for key in args.algorithms:
                print(f"  {key}, seed={seed}…", flush=True)
                options = dict(quick=args.quick, fid_context=fd_context, seed=seed,
                               n_gen=n_gen, epochs=args.epochs)
                if key == "gan":
                    options.update(ds_key=ds_key, reuse_model=args.reuse_gan)
                t0 = time.perf_counter()
                row = runners[key](Xtr, ytr, Xte, yte, **options)
                elapsed = time.perf_counter() - t0
                row["evaluation_s"] = max(0., elapsed - row["train_s"]) if np.isfinite(row["train_s"]) else elapsed
                row.setdefault("model_source", "trained_in_protocol")
                results.append(row)
        csv_path = output_dir / f"results_{ds_key}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        summary_path = output_dir / f"summary_{ds_key}.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["algo", "metric", "n_finite", "mean", "std"])
            writer.writeheader()
            writer.writerows(summarize_runs(results))
        _write_markdown(output_dir / f"results_{ds_key}.md", results, args.evaluation_split)
        protocol = {
            "protocol_version": 2, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": ds_key, "model_seeds": args.seeds, "data_seed": args.data_seed,
            "n_train": len(Xtr), "n_validation_reserved": len(val_X), "n_evaluation": len(Xte),
            "evaluation_split": args.evaluation_split, "validation_fraction": args.validation_fraction,
            "n_generated": n_gen, "quick": args.quick, "epochs_override": args.epochs,
            "algorithms": args.algorithms, "reuse_gan": args.reuse_gan,
            "label_usage": "stratified_sampling_and_external_evaluation_only",
            "selection": "fixed_configs_no_hyperparameter_selection_in_this_script",
            "train_timing": "fit_and_train_side_generation_statistics_including_adapter_diagnostics",
            "fd_pca": {"fit_split": "train", "n_components": len(fd_context[0][1]),
                       "shared_across_algorithms_and_seeds": True, "is_standard_fid": False},
            "fingerprints_sha256": {"train": _fingerprint(Xtr, ytr),
                                    "validation_pool": _fingerprint(val_X, val_y),
                                    "evaluation": _fingerprint(Xte, yte)},
            "versions": _versions(),
            "command_arguments": list(argv) if argv is not None else sys.argv[1:],
            "source_hashes_sha256": _source_fingerprints(),
            "configurations": [{"algo": row["algo"], "seed": row["seed"],
                                "model_source": row["model_source"],
                                "metadata": row["training_config"]} for row in results],
            "limitations": ["Default architectures have different compute and capacity budgets.",
                            "NPZ costs amortize the entire fitted model over this evaluation batch.",
                            "GAN inversion metrics are diagnostic, not a native compression benchmark.",
                            "Pixel-neighbor diagnostics use bounded pools and do not establish non-memorization."]}
        (output_dir / f"protocol_{ds_key}.json").write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        first_seed = [row for row in results if row["seed"] == args.seeds[0]]
        save_panel([(row["algo"], row["gen_sample"]) for row in first_seed], None,
                   output_dir / f"panel_generation_{ds_key}.png", f"Génération ({ds_key}, seed={args.seeds[0]})")
        save_panel([(row["algo"], row["rec_sample"]) for row in first_seed], Xte[:N_PANEL],
                   output_dir / f"panel_reconstruction_{ds_key}.png",
                   f"Reconstruction / inversion diagnostique ({ds_key}, seed={args.seeds[0]})")
        print(f"  Résultats et protocole : {output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()
