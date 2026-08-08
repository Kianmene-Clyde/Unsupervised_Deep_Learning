"""Banc d'expérimentation VAE + PCA — piloté par les MÊMES classes que l'app.

But : lancer en une commande beaucoup d'entraînements en faisant varier « plein
de paramètres », **surtout sur Quick, Draw! et un peu sur MNIST**, sans qu'aucun
entraînement soit trop long, et **sauvegarder chaque modèle dans le registre de
l'app** pour pouvoir tout revisualiser dans l'interface (`python app.py`).

--- Pourquoi ça « colle » à l'app ---

Rien n'est réimplémenté ici. On importe `vae.algo.VAE` et `pca.algo.PCAAlgo`
exactement comme le fait `src/app.py`, on appelle leur `train(X, p)` avec le même
dict d'hyperparamètres que les widgets, puis on `save()` avec la même méta que
l'app (`algo`, `dataset`, `n_samples`) — c'est ce sur quoi l'app filtre pour
n'afficher que les modèles du dataset courant. Un modèle produit ici est donc
strictement un modèle de l'app, juste entraîné en lot.

--- Deux façons de décrire les expériences ---

1. OFAT (« one factor at a time », le défaut) : on part d'une config de base et,
   pour chaque paramètre, on fait varier CE paramètre seul le long d'une liste de
   valeurs. Idéal pour isoler l'effet de chaque bouton.
2. GRID : produit cartésien de listes fournies dans un JSON.

Chaque dataset a une INTENSITÉ : `full` (tous les sweeps) ou `light` (une poignée
de configs clés). Par défaut : Quick, Draw! en `full`, MNIST en `light`.

Exemples
--------
    # Défaut : Quick, Draw! (full) + MNIST (light), modèles rangés dans l'app
    python run_experiments.py

    # Smoke test rapide
    python run_experiments.py --quick

    # Choisir soi-même datasets et intensités
    python run_experiments.py --datasets "quickdraw:full,quickdraw10:light"

Lancer depuis src/experiments/ (ou n'importe où : le chemin src est ajouté seul).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# TensorFlow bavarde beaucoup à l'import : on le fait taire AVANT (via vae.algo).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import matplotlib

matplotlib.use("Agg")  # backend sans fenêtre : on ne fait qu'écrire des PNG.

import matplotlib.pyplot as plt
import numpy as np

# src/ doit être importable (comme dans app.py). parents[1] = src/.
SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_import import DATASETS, load_dataset  # noqa: E402

IMAGE_DIM = 784


# --------------------------------------------------------------------------- #
# Configs de base + sweeps
# --------------------------------------------------------------------------- #
# La base du VAE = les défauts des widgets (vae/algo.py), mais epochs abaissé à 12
# pour qu'aucun run ne traîne — les bitmaps Quick, Draw! convergent vite.
VAE_BASELINE = {
    "architecture": "dense",
    "latent_dim": 10,
    "kl_weight": 1e-3,
    "hidden_dims": "128-32",
    "conv_filters": "32-64",
    "hidden_activation": "relu",
    "latent_activation": "linear",
    "output_activation": "sigmoid",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "batch_size": 256,
    "epochs": 12,
    "seed": 42,
}

# `full` : chaque axe balayé SEUL autour de la base. Valeurs reprises des choix
# des widgets. (Pas de sweep sur `epochs`/`seed` : ils allongent sans éclairer.)
VAE_SWEEPS_FULL = {
    "architecture": ["dense", "conv"],
    "latent_dim": [2, 5, 10, 20, 50],
    "kl_weight": [0.0, 1e-4, 1e-3, 1e-2, 1.0],
    "hidden_dims": ["64", "128-32", "256-64", "512-128-32"],
    "hidden_activation": ["relu", "elu", "tanh", "leaky_relu"],
    "latent_activation": ["linear", "tanh", "sigmoid"],
    "output_activation": ["sigmoid", "linear"],
    "optimizer": ["adam", "sgd", "rmsprop"],
    "learning_rate": [1e-2, 1e-3, 1e-4],
    "batch_size": [64, 128, 256, 512],
}

# `light` : juste de quoi couvrir les deux axes qui parlent le plus.
VAE_SWEEPS_LIGHT = {
    "latent_dim": [2, 10, 20],
    "kl_weight": [0.0, 1e-3, 1.0],
}

PCA_BASELINE = {"n_components": 50, "center": "feature"}
PCA_SWEEPS_FULL = {
    "n_components": [2, 5, 10, 20, 50, 100, 200],
    "center": ["feature", "global"],
}
PCA_SWEEPS_LIGHT = {"n_components": [10, 50], "center": ["feature"]}

SWEEPS = {
    "full": {"vae": VAE_SWEEPS_FULL, "pca": PCA_SWEEPS_FULL},
    "light": {"vae": VAE_SWEEPS_LIGHT, "pca": PCA_SWEEPS_LIGHT},
}

# --------------------------------------------------------------------------- #
# Mode `curated` : 10 tests conçus à la main, où PLUSIEURS hyperparamètres
# changent d'un test à l'autre (contrairement à l'OFAT qui n'en bouge qu'un).
# Chaque entrée n'écrit QUE ce qui diffère de la baseline ; le reste hérite.
# C'est ce que sélectionne `--mode curated` (le défaut).
# --------------------------------------------------------------------------- #
VAE_CURATED = [
    # 1. VAE standard, latent 2D — l'espace latent se dessine en entier dans l'app.
    {"latent_dim": 2, "kl_weight": 1e-3, "epochs": 20},
    # 2. Autoencodeur pur (kl=0, aucune régularisation), latent plus large.
    {"latent_dim": 10, "kl_weight": 0.0, "epochs": 15},
    # 3. KL forte + activation ELU : latent lisse, non-linéarité différente.
    {"latent_dim": 16, "kl_weight": 1e-2, "hidden_activation": "elu", "epochs": 15},
    # 4. Grand latent + réseau profond + KL faible : priorité à la fidélité.
    {"latent_dim": 32, "kl_weight": 1e-4, "hidden_dims": "512-128-32", "epochs": 20},
    # 5. VAE « pur » (kl=1) + z_mean borné par tanh : latent très contraint.
    {"latent_dim": 8, "kl_weight": 1.0, "latent_activation": "tanh", "epochs": 20},
    # 6. Convolutionnel puissant (filtres 64→128), latent moyen.
    {"architecture": "conv", "latent_dim": 16, "kl_weight": 1e-3,
     "conv_filters": "64-128", "epochs": 15},
    # 7. Convolutionnel léger (16→32), petit latent, KL moyenne.
    {"architecture": "conv", "latent_dim": 4, "kl_weight": 1e-2,
     "conv_filters": "16-32", "epochs": 15},
    # 8. Optimiseur SGD, pas d'apprentissage élevé, plus d'époques (descente brute).
    {"latent_dim": 10, "kl_weight": 1e-3, "optimizer": "sgd",
     "learning_rate": 1e-2, "epochs": 25},
    # 9. RMSprop + tanh + petit batch : autre dynamique d'entraînement.
    {"latent_dim": 20, "kl_weight": 1e-3, "optimizer": "rmsprop",
     "hidden_activation": "tanh", "batch_size": 128, "epochs": 15},
    # 10. Leaky ReLU + couches 256→64 + lr faible + gros batch : convergence lente.
    {"latent_dim": 24, "kl_weight": 1e-2, "hidden_activation": "leaky_relu",
     "hidden_dims": "256-64", "learning_rate": 1e-4, "batch_size": 512, "epochs": 20},
]

# La PCA n'a que 2 boutons (k, centrage) : une petite série qui les couvre.
PCA_CURATED = [
    {"n_components": 5, "center": "feature"},
    {"n_components": 20, "center": "feature"},
    {"n_components": 50, "center": "feature"},
    {"n_components": 100, "center": "feature"},
    {"n_components": 200, "center": "feature"},
    {"n_components": 50, "center": "global"},
]

CURATED = {"vae": VAE_CURATED, "pca": PCA_CURATED}


# --------------------------------------------------------------------------- #
# Fabrication de la liste des runs
# --------------------------------------------------------------------------- #
def expand_ofat(baseline, sweeps):
    """Runs OFAT : la base, puis chaque valeur de chaque axe prise SEULE."""
    runs = [dict(baseline)]
    seen = {_key(baseline)}
    for axis, values in sweeps.items():
        for v in values:
            cand = dict(baseline)
            cand[axis] = v
            k = _key(cand)
            if k not in seen:
                seen.add(k)
                runs.append(cand)
    return runs


def expand_grid(baseline, grid):
    """Produit cartésien des listes de `grid`, chaque combo posé sur la base."""
    # On ignore les clés de commentaire (_comment…) : ce ne sont pas des axes.
    grid = {k: v for k, v in (grid or {}).items() if not k.startswith("_")}
    if not grid:
        return []
    axes = list(grid.keys())
    runs, seen = [], set()
    for combo in itertools.product(*(grid[a] for a in axes)):
        cand = dict(baseline)
        cand.update(dict(zip(axes, combo)))
        k = _key(cand)
        if k not in seen:
            seen.add(k)
            runs.append(cand)
    return runs


def _key(params):
    return tuple(sorted((k, _hashable(v)) for k, v in params.items()))


def _hashable(v):
    return tuple(v) if isinstance(v, list) else v


def build_runs(baseline, sweeps, grid, mode):
    """Liste des dicts de params pour un algo, selon le mode."""
    runs = []
    if mode in ("ofat", "both"):
        runs += expand_ofat(baseline, sweeps)
    if mode in ("grid", "both"):
        runs += expand_grid(baseline, grid)
    out, seen = [], set()
    for r in runs:
        k = _key(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def load_data(dataset, n_train, n_test, seed, quickdraw_per_class):
    """Charge le dataset via l'app, puis sous-échantillonne train/test.

    Le MÊME (X_train, X_test) sert à TOUS les runs de ce dataset : les MSE sont
    ainsi directement comparables. n_train/n_test = None -> tout garder.
    """
    kwargs = {}
    if dataset in ("quickdraw", "quickdraw10"):
        kwargs["max_per_class"] = quickdraw_per_class
        kwargs["seed"] = seed
    ds = load_dataset(dataset, **kwargs)

    rng = np.random.default_rng(seed)

    def subsample(X, y, n):
        if n is None or n >= len(X):
            return X, y
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx], y[idx]

    X_train, y_train = subsample(ds.X_train, ds.y_train, n_train)
    X_test, y_test = subsample(ds.X_test, ds.y_test, n_test)
    return ds, X_train, y_train, X_test, y_test


# --------------------------------------------------------------------------- #
# Reconstruction (pour la MSE test), spécifique à chaque algo
# --------------------------------------------------------------------------- #
def vae_reconstruct(weights, X):
    from vae.utils.model import decode_batch, encode_batch

    return decode_batch(encode_batch(X, weights), weights)


def pca_reconstruct(algo, weights, meta, X):
    from utils.pca.compression import compress, reconstruct

    pca = algo._pca_of(weights, meta)
    return reconstruct(pca, compress(pca, np.asarray(X, dtype=np.float64)))


def mse_per_pixel(X, X_hat):
    X = np.asarray(X, dtype=np.float64)
    X_hat = np.asarray(X_hat, dtype=np.float64)
    return float(np.mean((X - X_hat) ** 2))


# --------------------------------------------------------------------------- #
# Un run
# --------------------------------------------------------------------------- #
def run_one(algo_key, algo, params, X_train, X_test, dataset, n_train,
            save_models, used_names):
    """Entraîne, mesure, (sauvegarde) et renvoie (record plat, detail complet)."""
    p = dict(params)
    p["n_samples"] = n_train

    record = {"algo": algo_key, "dataset": dataset,
              **{k: _flat(v) for k, v in params.items()}}
    detail = {"algo": algo_key, "dataset": dataset, "params": params}

    try:
        algo.check(p, n_train)
    except Exception as e:  # noqa: BLE001 — on continue le balayage
        record["status"] = "skipped"
        record["error"] = str(e)
        return record, detail

    try:
        t0 = time.perf_counter()
        weights, meta = algo.train(X_train, p)
        train_seconds = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = str(e)
        return record, detail

    # Même enrichissement que l'app (app.py:454) : c'est ce sur quoi elle filtre.
    meta.update({"algo": algo_key, "dataset": dataset, "n_samples": n_train})

    k = int(meta["k"])
    train_mse = meta["inertia"] / (n_train * IMAGE_DIM)
    X_hat_test = (vae_reconstruct(weights, X_test) if algo_key == "vae"
                  else pca_reconstruct(algo, weights, meta, X_test))
    test_mse = mse_per_pixel(X_test, X_hat_test)

    record.update({
        "status": "ok",
        "k": k,
        "code_floats": k,
        "ratio_vs_uint8": IMAGE_DIM * 8 / (k * 32),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "inertia": float(meta["inertia"]),
        "train_seconds": round(train_seconds, 2),
    })

    if algo_key == "vae":
        kl_hist = meta.get("kl_history") or []
        total_hist = meta.get("total_history") or []
        record["kl_final"] = float(kl_hist[-1]) if kl_hist else float("nan")
        record["total_loss_final"] = float(total_hist[-1]) if total_hist else float("nan")
        std = meta.get("latent_std") or []
        record["latent_std_mean"] = float(np.mean(std)) if std else float("nan")
        detail["reconstruction_mse_history"] = [
            v / (n_train * IMAGE_DIM) for v in meta.get("inertia_history", [])
        ]
        detail["kl_history"] = kl_hist
        detail["total_history"] = total_hist
    else:
        record["explained_variance_kept"] = float(meta.get("explained_variance_kept", float("nan")))
        detail["inertia_history"] = meta.get("inertia_history", [])

    if save_models:
        # Même convention de nom que l'app : "{algo}_{auto_name}". auto_name
        # n'encode pas tous les axes balayés -> on désambiguë les collisions,
        # sinon deux runs (ex. optimizers différents) s'écraseraient.
        name = unique_name(f"{algo_key}_{algo.auto_name(dataset, p)}", used_names)
        try:
            algo.save(name, weights, meta)
            record["saved_as"] = name
        except Exception as e:  # noqa: BLE001
            record["save_error"] = str(e)

    return record, detail


def unique_name(base, used_names):
    """base, base_2, base_3… — garantit que chaque modèle a son propre fichier."""
    name = base
    i = 2
    while name in used_names:
        name = f"{base}_{i}"
        i += 1
    used_names.add(name)
    return name


def _flat(v):
    return "-".join(map(str, v)) if isinstance(v, (list, tuple)) else v


# --------------------------------------------------------------------------- #
# Écriture des résultats
# --------------------------------------------------------------------------- #
def write_csv(records, path):
    cols = []
    for r in records:
        for k in r:
            if k not in cols:
                cols.append(k)
    lines = [",".join(cols)]
    for r in records:
        lines.append(",".join(_csv_cell(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def _csv_cell(v):
    s = "" if v is None else str(v)
    if any(c in s for c in (",", '"', "\n")):
        s = '"' + s.replace('"', '""') + '"'
    return s


# --------------------------------------------------------------------------- #
# Figures de diagnostic (par dataset, PROPRES à un algo — jamais PCA vs VAE)
# --------------------------------------------------------------------------- #
def plot_vae_kl_tradeoff(records, baseline, out_dir, dataset):
    """Arbitrage du VAE : reconstruction et KL selon le poids KL (VAE seul)."""
    vae = [r for r in records if r.get("algo") == "vae"
           and r.get("dataset") == dataset and r.get("status") == "ok"]
    pts = sorted({(r["kl_weight"], r["train_mse"], r.get("kl_final", float("nan")))
                  for r in vae
                  if r.get("architecture") == baseline["architecture"]
                  and r.get("k") == baseline["latent_dim"]
                  and r.get("kl_weight") is not None})
    if len(pts) < 2:
        return None
    kls, mses, klf = zip(*pts)
    x = range(len(kls))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax1.plot(x, mses, "o-", color="tab:blue")
    ax1.set_xticks(list(x)); ax1.set_xticklabels([f"{v:g}" for v in kls])
    ax1.set_xlabel("poids KL"); ax1.set_ylabel("MSE reconstruction (train)")
    ax1.set_title("Le prix de la régularisation"); ax1.grid(True, alpha=0.3)
    ax2.plot(x, klf, "o-", color="tab:red")
    ax2.set_xticks(list(x)); ax2.set_xticklabels([f"{v:g}" for v in kls])
    ax2.set_xlabel("poids KL"); ax2.set_ylabel("KL finale (sommée sur les dims)")
    ax2.set_title("Rapprochement du latent vers N(0, 1)"); ax2.grid(True, alpha=0.3)
    fig.suptitle(f"VAE {dataset} — arbitrage reconstruction / KL "
                 f"(latent={baseline['latent_dim']})", fontweight="bold")
    fig.tight_layout()
    path = out_dir / f"vae_kl_tradeoff_{dataset}.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


def plot_vae_mse_vs_latent(records, out_dir, dataset):
    """MSE test vs dimension latente, une courbe par kl_weight (VAE seul)."""
    vae = [r for r in records if r.get("algo") == "vae"
           and r.get("dataset") == dataset and r.get("status") == "ok"
           and r.get("architecture") == "dense"]
    fig, ax = plt.subplots(figsize=(8, 5))
    drew = False
    for kl in sorted({r.get("kl_weight") for r in vae if r.get("kl_weight") is not None}):
        pts = sorted({(r["k"], r["test_mse"]) for r in vae if r.get("kl_weight") == kl})
        if len(pts) >= 2:
            ks, ms = zip(*pts)
            ax.plot(ks, ms, "o-", label=f"kl={kl:g}"); drew = True
    if not drew:
        plt.close(fig); return None
    ax.set_xlabel("dimension latente"); ax.set_ylabel("MSE reconstruction (test)")
    ax.set_title(f"VAE {dataset} — MSE vs dimension latente"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8); fig.tight_layout()
    path = out_dir / f"vae_mse_vs_latent_{dataset}.png"
    fig.savefig(path, dpi=130); plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Config / datasets
# --------------------------------------------------------------------------- #
def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def resolve_baseline(cfg, algo, default):
    return {**default, **(cfg or {}).get(algo, {}).get("baseline", {})}


def parse_datasets(spec):
    """"quickdraw:full,mnist:light" -> [("quickdraw","full"), ("mnist","light")]."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, intensity = chunk.partition(":")
        intensity = intensity or "full"
        if name not in DATASETS:
            raise SystemExit(f"Dataset inconnu : {name!r}. Attendu : {list(DATASETS)}")
        if intensity not in SWEEPS:
            raise SystemExit(f"Intensité inconnue : {intensity!r}. full ou light.")
        out.append((name, intensity))
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Banc d'expérimentation VAE + PCA (pilote les classes de l'app).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config", help="JSON de config (baseline/sweeps/grid).")
    ap.add_argument("--algos", default="vae,pca",
                    help="Algos testés, séparés par des virgules : vae, pca.")
    ap.add_argument("--datasets", default="quickdraw",
                    help="Datasets et intensités, ex. 'quickdraw:full,mnist:light'. "
                         "L'intensité est ignorée en mode curated.")
    ap.add_argument("--mode", default="curated",
                    choices=["curated", "ofat", "grid", "both"],
                    help="curated = 10 tests conçus main (plusieurs params changent "
                         "à chaque fois) ; ofat = un param à la fois ; grid = produit "
                         "cartésien.")
    ap.add_argument("--n-train", type=int, default=0,
                    help="Images d'entraînement (0 = TOUT le split, le défaut).")
    ap.add_argument("--n-test", type=int, default=0,
                    help="Images de test pour la MSE (0 = TOUT le split, le défaut).")
    ap.add_argument("--quickdraw-per-class", type=int, default=4000,
                    help="Dessins par classe pour quickdraw* avant sous-échantillon.")
    ap.add_argument("--data-seed", type=int, default=42,
                    help="Graine du tirage des données (fixe pour tous les runs).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Force le nombre d'époques du VAE (écrase la baseline).")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Plafonne le nombre de runs par dataset (0 = pas de limite).")
    ap.add_argument("--output", default=None,
                    help="Dossier de LOGS csv/json/png (défaut : results/<horodatage>/).")
    ap.add_argument("--no-save", action="store_true",
                    help="NE PAS ranger les modèles dans le registre de l'app.")
    ap.add_argument("--no-plots", action="store_true", help="Ne pas générer de figures.")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test : peu d'images, 3 époques.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Liste les runs planifiés et s'arrête.")
    args = ap.parse_args()

    algos_wanted = [a.strip() for a in args.algos.split(",") if a.strip()]
    n_train = None if args.n_train in (0, None) else args.n_train
    n_test = None if args.n_test in (0, None) else args.n_test
    save_models = not args.no_save

    cfg = load_config(args.config)
    vae_baseline = resolve_baseline(cfg, "vae", VAE_BASELINE)
    pca_baseline = resolve_baseline(cfg, "pca", PCA_BASELINE)
    if args.epochs is not None:
        vae_baseline["epochs"] = args.epochs

    datasets = parse_datasets(args.datasets)

    # Époques forcées : par --quick (3) ou --epochs. None = laisser chaque config
    # décider (utile en curated, où chaque test a ses propres époques).
    force_epochs = 3 if args.quick else args.epochs
    if args.quick:
        n_train = min(n_train or 600, 600)
        n_test = min(n_test or 400, 400)

    grids = {"vae": (cfg.get("vae", {}) or {}).get("grid", {}),
             "pca": (cfg.get("pca", {}) or {}).get("grid", {})}
    baselines = {"vae": vae_baseline, "pca": pca_baseline}

    def plan_for(dataset, intensity):
        plan = []
        for algo_key in algos_wanted:
            if args.mode == "curated":
                # Chaque test = baseline + ce qu'il change (plusieurs params).
                runs = [{**baselines[algo_key], **ov} for ov in CURATED[algo_key]]
            else:
                sweeps = SWEEPS[intensity][algo_key]
                runs = build_runs(baselines[algo_key], sweeps,
                                  grids[algo_key], args.mode)
            # Époques forcées (quick / --epochs) : ne touche que le VAE.
            if algo_key == "vae" and force_epochs is not None:
                runs = [{**r, "epochs": force_epochs} for r in runs]
            if args.max_runs:
                runs = runs[: args.max_runs]
            plan += [(algo_key, p) for p in runs]
        return plan

    # --- Dry-run : on liste et on sort, sans toucher à TensorFlow ni au disque.
    if args.dry_run:
        total = 0
        for dataset, intensity in datasets:
            plan = plan_for(dataset, intensity)
            total += len(plan)
            print(f"\n=== {dataset} [{intensity}] — {len(plan)} runs ===")
            for i, (algo_key, p) in enumerate(plan, 1):
                print(f"  {i:>3}. {algo_key:4} {p}")
        print(f"\n[plan] {total} runs au total, save={'off' if args.no_save else 'on'}")
        return

    # --- Dossier de logs.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else (SRC.parent / "results" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] logs -> {out_dir}")
    if save_models:
        print("[out] modèles -> registres de l'app (src/<algo>/models/), "
              "visibles dans `python app.py`")

    # --- Instances d'algos (import paresseux : VAE tire TensorFlow).
    instances = {}
    if "vae" in algos_wanted:
        from vae.algo import VAE
        instances["vae"] = VAE()
    if "pca" in algos_wanted:
        from pca.algo import PCAAlgo
        instances["pca"] = PCAAlgo()

    records, details = [], []
    t_start = time.perf_counter()

    for dataset, intensity in datasets:
        plan = plan_for(dataset, intensity)
        print(f"\n########## {dataset} [{intensity}] — {len(plan)} runs ##########")
        if dataset in ("quickdraw", "quickdraw10"):
            print("[data] Quick, Draw! : téléchargement des .npy au 1er appel "
                  "(mis en cache dans data/numpy_bitmap/).")
        ds, X_train, y_train, X_test, y_test = load_data(
            dataset, n_train, n_test, args.data_seed, args.quickdraw_per_class,
        )
        n_train_eff = len(X_train)
        print(f"[data] {ds.describe()}")
        print(f"[data] train={len(X_train)}  test={len(X_test)}")

        used_names = set()
        for i, (algo_key, params) in enumerate(plan, 1):
            label = ", ".join(f"{k}={_flat(v)}" for k, v in params.items())
            print(f"[{dataset} {i}/{len(plan)}] {algo_key}: {label}", flush=True)
            rec, det = run_one(
                algo_key, instances[algo_key], params, X_train, X_test,
                dataset, n_train_eff, save_models, used_names,
            )
            if rec.get("status") == "ok":
                extra = (f"test_mse={rec['test_mse']:.4f} "
                         f"train_mse={rec['train_mse']:.4f} ({rec['train_seconds']}s)")
                if rec.get("saved_as"):
                    extra += f"  saved: {rec['saved_as']}"
            else:
                extra = f"{rec.get('status')}: {rec.get('error', '')}"
            print(f"       -> {extra}")
            records.append(rec)
            details.append(det)

            # Écriture incrémentale : un crash tardif ne perd pas les runs faits.
            write_csv(records, out_dir / "results.csv")
            (out_dir / "details.json").write_text(
                json.dumps(details, indent=2, default=float), encoding="utf-8")

        # Figures de diagnostic de CE dataset (VAE seul, pas de comparaison).
        if not args.no_plots and "vae" in algos_wanted:
            for fn in (lambda: plot_vae_kl_tradeoff(records, vae_baseline, out_dir, dataset),
                       lambda: plot_vae_mse_vs_latent(records, out_dir, dataset)):
                path = fn()
                if path:
                    print(f"[plots] {path.name}")

    total = time.perf_counter() - t_start
    n_ok = sum(1 for r in records if r.get("status") == "ok")
    n_saved = sum(1 for r in records if r.get("saved_as"))
    print(f"\n[done] {n_ok}/{len(records)} runs OK en {total:.0f}s")
    if save_models:
        print(f"[done] {n_saved} modèles rangés dans les registres de l'app "
              f"— ouvre `python app.py` et choisis le dataset pour les voir.")

    (out_dir / "run_meta.json").write_text(json.dumps({
        "datasets": datasets, "mode": args.mode,
        "n_train": args.n_train, "n_test": args.n_test,
        "data_seed": args.data_seed, "algos": algos_wanted,
        "vae_baseline": vae_baseline, "pca_baseline": pca_baseline,
        "saved_models": save_models, "n_models_saved": n_saved,
        "total_seconds": round(total, 1), "timestamp": stamp,
    }, indent=2, default=float), encoding="utf-8")

    print(f"\nLogs : {out_dir}  (results.csv · details.json · run_meta.json · *.png)")


if __name__ == "__main__":
    main()
