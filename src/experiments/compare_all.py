"""Banc de comparaison FINAL — les 6 algos, les 3 tâches, les 2 datasets.

C'est le protocole de test du rendu final : un seul script, des seeds fixes,
des métriques COMMUNES (src/utils/metrics_common.py), et un tableau de sortie
que le rapport et les slides reprennent tel quel.

Usage, depuis src/ :

    python experiments/compare_all.py                    # mnist + quickdraw
    python experiments/compare_all.py --datasets mnist
    python experiments/compare_all.py --quick            # réglages allégés

--- Le protocole, en clair ---

Pour chaque dataset (MNIST, Quick Draw! cat/apple/car) :
    train : 10 000 images du split train, tirées de façon déterministe et
            équilibrées par classe (1 000 avec --quick) ;
    test  : 2 000 images du split test, équilibrées de la même façon (256 avec
            --quick) — JAMAIS vues à
            l'entraînement, toutes les métriques de reconstruction et de
            génération sont mesurées dessus.

Configurations comparées (une par algo, raisonnable plutôt qu'optimale — le
but est la comparaison ENTRE algos, pas le record de chacun) :
    K-means      k=64                       code : 1 entier  (6 bits)
    Kohonen      8×8 = 64 neurones          code : 1 entier  (6 bits)
    PCA          16 composantes             code : 16 flottants (512 bits)
    AutoEncoder  dense 128→32, z=16         code : 16 flottants (512 bits)
    VAE          dense 128→32, z=16, kl 1e-3  code : 16 flottants (512 bits)
    GAN          z=64, entraîné dans le protocole (`--reuse-gan` pour le registre)

K-means et Kohonen à k égal (64) pour s'isoler mutuellement ; PCA/AE/VAE à
budget de code égal (16 flottants) pour la même raison. Les deux familles ne
jouent PAS à budget égal entre elles — c'est voulu et affiché : les colonnes
« bits » et « ratio » du tableau rendent l'écart explicite.

Les métriques par tâche :
    T1 projection : silhouette des VRAIES classes dans l'espace latent
                    (continu) — les classes se séparent-elles ? — et pureté
                    des clusters (discret) ;
    T2 compression : MSE, PSNR, SSIM sur le test + bits/image et ratio ;
    T3 génération : 2 000 images générées par algo, FID simplifiée
                    (distance de Fréchet sur features PCA-50, voir
                    metrics_common) contre les 2 000 images de test.

Sorties dans src/experiments/results/comparison/ :
    results_<dataset>.csv / .md    le tableau complet
    protocol_<dataset>.json           seed, volumes, versions et options
    panel_generation_<dataset>.png    8 générations par algo, côte à côte
    panel_reconstruction_<dataset>.png  8 reconstructions par algo + originaux
"""

import argparse
import csv
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

from data_import import load_dataset  # noqa: E402
from utils.metrics_common import (  # noqa: E402
    compression_ratio,
    fit_pca_features,
    frechet_distance,
    frechet_from_images,
    mse,
    pca_features,
    psnr,
    purity,
    silhouette,
    ssim,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "comparison"
SEED = 42
N_TRAIN = 10000
N_TEST = 2000
N_GEN = 2000
N_PANEL = 8  # vignettes par algo dans les panneaux


def balanced_subset(X, y, n, seed):
    """Return a deterministic, approximately class-balanced subset."""
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X et y doivent avoir la même longueur.")
    if not 1 <= int(n) <= len(X):
        raise ValueError(f"n doit être compris entre 1 et {len(X)}, reçu : {n}.")
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


def generation_fid(X_real, X_generated, context=None):
    """Use one shared PCA feature space per dataset when a context is supplied."""
    if context is None:
        return frechet_from_images(X_real, X_generated, seed=SEED)
    featurizer, real_features = context
    return frechet_distance(real_features, pca_features(X_generated, featurizer))


# ------------------------------------------------------------------ K-means


def run_kmeans(Xtr, ytr, Xte, yte, k=64, quick=False, fid_context=None):
    from kmeas.utils.kmeas import (assign_clusters, compute_squared_distances,
                                   fit_kmeans)
    from kmeas.utils.generation import cluster_stats, sample_synthetic

    t0 = time.time()
    centroids, labels_tr, _ = fit_kmeans(
        X=Xtr, k=k, max_iter=30 if quick else 100, tolerance=1e-4,
        seed=SEED, verbose=False, return_history=True,
    )
    centroids = np.asarray(centroids, dtype=np.float32)
    train_s = time.time() - t0

    labels_te = np.asarray(assign_clusters(compute_squared_distances(Xte, centroids)))
    X_rec = centroids[labels_te]

    labels_tr = np.asarray(labels_tr)
    props, sigmas = cluster_stats(Xtr, labels_tr, k)
    X_gen, _ = sample_synthetic(centroids, props, sigmas, N_GEN, seed=SEED)

    return {
        "algo": "K-means (k=64)", "train_s": train_s,
        "bits": int(np.ceil(np.log2(k))),
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": purity(labels_te, yte),
        "silhouette": silhouette(Xte, labels_te, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ------------------------------------------------------------------ Kohonen


def run_kohonen(Xtr, ytr, Xte, yte, rows=8, cols=8, quick=False, fid_context=None):
    from kohonen.utils.kohonen import assign_clusters, fit_kohonen
    from kohonen.utils.generation import activation_weights, sample_synthetic

    t0 = time.time()
    weights, labels_tr, _ = fit_kohonen(
        X=Xtr, rows=rows, cols=cols, n_epochs=2 if quick else 8,
        alpha=0.1, gamma=1.0, seed=SEED, verbose=False, return_history=True,
    )
    weights = np.asarray(weights, dtype=np.float32)
    train_s = time.time() - t0

    labels_te = np.asarray(assign_clusters(Xte, weights))
    X_rec = weights[labels_te]

    act = activation_weights(np.asarray(labels_tr), rows * cols)
    X_gen, _ = sample_synthetic(weights, rows, cols, act, N_GEN, seed=SEED)

    return {
        "algo": f"Kohonen ({rows}×{cols})", "train_s": train_s,
        "bits": int(np.ceil(np.log2(rows * cols))),
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": purity(labels_te, yte),
        "silhouette": silhouette(Xte, labels_te, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ---------------------------------------------------------------------- PCA


def run_pca(Xtr, ytr, Xte, yte, d=16, quick=False, fid_context=None):
    from utils.pca.dim_reduction import fit_pca
    from utils.pca.compression import compress, reconstruct
    from utils.pca.generation import fit_latent_gaussian, generate, sample_latent

    t0 = time.time()
    pca = fit_pca(Xtr, n_components=d)
    train_s = time.time() - t0

    Z_te = compress(pca, Xte)
    X_rec = np.clip(np.asarray(reconstruct(pca, Z_te), dtype=np.float32), 0, 1)

    mean_z, cov_z = fit_latent_gaussian(pca, Xtr)
    X_gen = np.clip(
        np.asarray(generate(pca, sample_latent(mean_z, cov_z, N_GEN, seed=SEED)),
                   dtype=np.float32), 0, 1,
    )

    return {
        "algo": f"PCA (d={d})", "train_s": train_s, "bits": d * 32,
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": float("nan"),
        "silhouette": silhouette(np.asarray(Z_te), yte, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ------------------------------------------------------------- AutoEncoder


def run_autoencoder(Xtr, ytr, Xte, yte, z=16, quick=False, fid_context=None):
    from autoencoder.algo import Autoencoder
    from autoencoder.utils.model import decode_batch, encode_batch

    algo = Autoencoder()
    p = {"architecture": "dense", "latent_dim": z, "hidden_dims": "128-32",
         "hidden_activation": "relu", "latent_activation": "linear",
         "output_activation": "sigmoid", "optimizer": "adam",
         "learning_rate": 1e-3, "batch_size": 256,
         "epochs": 5 if quick else 20, "seed": SEED}
    t0 = time.time()
    weights, meta = algo.train(Xtr, p)
    train_s = time.time() - t0

    Z_te = encode_batch(Xte, weights)
    X_rec = decode_batch(Z_te, weights)

    # Génération : gaussienne EMPIRIQUE des codes d'entraînement (l'AE simple
    # n'est pas régularisé — N(0, I) tomberait à côté du nuage).
    Z_tr = encode_batch(Xtr, weights)
    rng = np.random.default_rng(SEED)
    cov = np.cov(Z_tr, rowvar=False) + 1e-6 * np.eye(z)
    Z_gen = rng.multivariate_normal(Z_tr.mean(axis=0), cov, size=N_GEN)
    X_gen = decode_batch(Z_gen.astype(np.float32), weights)

    return {
        "algo": f"AutoEncoder (z={z})", "train_s": train_s, "bits": z * 32,
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": float("nan"),
        "silhouette": silhouette(Z_te, yte, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ---------------------------------------------------------------------- VAE


def run_vae(Xtr, ytr, Xte, yte, z=16, quick=False, fid_context=None):
    from vae.algo import VAE
    from vae.utils.model import decode_batch, encode_batch

    algo = VAE()
    p = {"architecture": "dense", "latent_dim": z, "kl_weight": 1e-3,
         "hidden_dims": "128-32", "hidden_activation": "relu",
         "latent_activation": "linear", "output_activation": "sigmoid",
         "optimizer": "adam", "learning_rate": 1e-3, "batch_size": 256,
         "epochs": 5 if quick else 20, "seed": SEED}
    t0 = time.time()
    weights, meta = algo.train(Xtr, p)
    train_s = time.time() - t0

    Z_te = encode_batch(Xte, weights)
    X_rec = decode_batch(Z_te, weights)

    # Génération : z ~ N(0, I) — LA récompense de la régularisation KL.
    rng = np.random.default_rng(SEED)
    X_gen = decode_batch(rng.standard_normal((N_GEN, z)).astype(np.float32), weights)

    return {
        "algo": f"VAE (z={z}, kl=1e-3)", "train_s": train_s, "bits": z * 32,
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": float("nan"),
        "silhouette": silhouette(Z_te, yte, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ---------------------------------------------------------------------- GAN


def run_gan(
    Xtr, ytr, Xte, yte, ds_key, quick=False, reuse_model=False, fid_context=None
):
    """Charge le GAN du registre (entraîné par gan/train_gan.py) — ou l'entraîne
    en vitesse s'il n'existe pas encore (mode --quick des vérifications)."""
    from gan.algo import GAN
    from gan.utils.model import decode_batch, encode_batch, generate_batch
    from gan.utils.registry import list_models, load_model

    algo = GAN()
    # Le bon modèle est celui dont la MÉTA dit le dataset — pas le nom, qui
    # peut varier selon qu'il vient de l'app ou de gan/train_gan.py.
    names = []
    if reuse_model:
        for name in list_models():
            try:
                if load_model(name)[1].get("dataset") == ds_key:
                    names.append(name)
            except (OSError, ValueError) as error:
                print(f"    GAN ignoré (modèle illisible {name!r}) : {error}")
    if names:
        name = names[-1]
        weights, meta = algo.load(name)
        train_s = float("nan")  # entraîné hors protocole, par gan/train_gan.py
        print(f"    GAN chargé du registre : {name}")
    else:
        p = {"latent_dim": 64, "g_hidden": "256-512", "d_hidden": "512-256",
             "learning_rate": 2e-4, "label_smoothing": 0.9, "batch_size": 128,
             "epochs": 5 if quick else 40, "seed": SEED}
        t0 = time.time()
        weights, meta = algo.train(Xtr, p)
        train_s = time.time() - t0

    z = int(meta["latent_dim"])
    rng = np.random.default_rng(SEED)
    X_gen = generate_batch(rng.standard_normal((N_GEN, z)).astype(np.float32),
                           weights)

    # « Compression » par inversion (banque) : mesurée pour être comparée —
    # le chiffre élevé EST le résultat (pas d'encodeur, voir gan/algo.py).
    Z_te = encode_batch(Xte, weights)
    X_rec = decode_batch(Z_te, weights)

    return {
        "algo": f"GAN (z={z}, inversion)", "train_s": train_s, "bits": z * 32,
        "mse": mse(Xte, X_rec), "psnr": psnr(Xte, X_rec), "ssim": ssim(Xte, X_rec),
        "purity": float("nan"),
        "silhouette": silhouette(Z_te, yte, seed=SEED),
        "fid": generation_fid(Xte, X_gen, fid_context),
        "rec_sample": X_rec[:N_PANEL], "gen_sample": X_gen[:N_PANEL],
    }


# ------------------------------------------------------------------ Panneaux


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


# ---------------------------------------------------------------------- Main


COLUMNS = ["algo", "train_s", "bits", "ratio", "mse", "psnr", "ssim",
           "purity", "silhouette", "fid"]


def fmt(v, digits=4):
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:.{digits}g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def main():
    global N_GEN
    parser = argparse.ArgumentParser(
        description="Compare les six algorithmes sur projection, compression et génération."
    )
    parser.add_argument("--datasets", nargs="+", default=["mnist", "quickdraw"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--n-gen", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--algorithms", nargs="+",
        choices=["kmeans", "kohonen", "pca", "autoencoder", "vae", "gan"],
        default=["kmeans", "kohonen", "pca", "autoencoder", "vae", "gan"],
    )
    parser.add_argument(
        "--reuse-gan", action="store_true",
        help="Réutilise un GAN sauvegardé au lieu de l'entraîner dans le protocole.",
    )
    args = parser.parse_args()

    n_train = args.n_train if args.n_train is not None else (1000 if args.quick else N_TRAIN)
    n_test = args.n_test if args.n_test is not None else (256 if args.quick else N_TEST)
    N_GEN = args.n_gen if args.n_gen is not None else (256 if args.quick else N_GEN)
    if min(n_train, n_test, N_GEN) < N_PANEL:
        parser.error(f"n-train, n-test et n-gen doivent valoir au moins {N_PANEL}.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "kmeans": run_kmeans,
        "kohonen": run_kohonen,
        "pca": run_pca,
        "autoencoder": run_autoencoder,
        "vae": run_vae,
    }

    for ds_key in args.datasets:
        print(f"\n=== {ds_key} ===")
        ds = load_dataset(ds_key)
        Xtr, ytr = balanced_subset(ds.X_train, ds.y_train, n_train, SEED)
        Xte, yte = balanced_subset(ds.X_test, ds.y_test, n_test, SEED + 1)
        fid_featurizer = fit_pca_features(Xte, n_components=min(50, Xte.shape[1]))
        fid_context = (fid_featurizer, pca_features(Xte, fid_featurizer))
        print(f"  {len(Xtr)} train / {len(Xte)} test")

        results = []
        for key in args.algorithms:
            if key == "gan":
                continue
            run = runners[key]
            print(f"  {run.__name__}…", flush=True)
            results.append(run(
                Xtr, ytr, Xte, yte, quick=args.quick, fid_context=fid_context
            ))
        if "gan" in args.algorithms:
            print("  run_gan…", flush=True)
            results.append(run_gan(
                Xtr, ytr, Xte, yte, ds_key, quick=args.quick,
                reuse_model=args.reuse_gan, fid_context=fid_context,
            ))

        for r in results:
            r["ratio"] = compression_ratio(r["bits"])

        # --- CSV
        csv_path = output_dir / f"results_{ds_key}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for r in results:
                w.writerow([r[c] for c in COLUMNS])
        print(f"  résultats : {csv_path}")

        # --- Markdown (repris tel quel par le rapport)
        md_path = output_dir / f"results_{ds_key}.md"
        header = ("| Algo | Train (s) | Bits/img | Ratio | MSE ↓ | PSNR ↑ | "
                  "SSIM ↑ | Pureté ↑ | Silhouette ↑ | FID simplifiée ↓ |")
        sep = "|" + "---|" * 10
        lines = [header, sep]
        for r in results:
            lines.append(
                "| " + " | ".join([
                    r["algo"], fmt(r["train_s"], 3), str(r["bits"]),
                    f"{r['ratio']:.0f}:1", fmt(r["mse"]), fmt(r["psnr"], 4),
                    fmt(r["ssim"], 3), fmt(r["purity"], 3),
                    fmt(r["silhouette"], 3), fmt(r["fid"], 4),
                ]) + " |"
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        protocol = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": ds_key,
            "seed": SEED,
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "n_generated": N_GEN,
            "quick": args.quick,
            "algorithms": args.algorithms,
            "reuse_gan": args.reuse_gan,
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
        protocol_path = output_dir / f"protocol_{ds_key}.json"
        protocol_path.write_text(
            json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  protocole : {protocol_path}")
        print(f"  tableau : {md_path}")

        # --- Panneaux visuels
        save_panel([(r["algo"], r["gen_sample"]) for r in results], None,
                   output_dir / f"panel_generation_{ds_key}.png",
                   f"Génération — {N_PANEL} tirages par algo ({ds_key})")
        save_panel([(r["algo"], r["rec_sample"]) for r in results], Xte[:N_PANEL],
                   output_dir / f"panel_reconstruction_{ds_key}.png",
                   f"Compression/décompression — mêmes {N_PANEL} images de test "
                   f"({ds_key})")

    print("\nTerminé. Les tableaux .md sont prêts à coller dans le rapport.")


if __name__ == "__main__":
    main()
