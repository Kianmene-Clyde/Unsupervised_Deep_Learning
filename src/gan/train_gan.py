"""Entraîne un GAN et produit toutes ses figures — le pendant ligne de commande
de l'onglet Entraînement de l'app, et le script de démonstration du rendu final.

Usage, depuis src/ :

    python gan/train_gan.py                          # MNIST, réglages « qualité »
    python gan/train_gan.py --dataset quickdraw      # Quick, Draw! (cat/apple/car)
    python gan/train_gan.py --quick                  # réglages « rapides » (~30 s)

Le modèle est sauvegardé dans src/gan/models/ (même registre que l'app : il
apparaît dans le sélecteur au prochain lancement) et les figures dans
src/gan/figures/<nom_du_modèle>/.

Deux jeux d'hyperparamètres, comme le demande le protocole du projet :
    --quick    : 10 époques sur 5 000 images — vérifier que tout tourne ;
    (défaut)   : 60 époques sur 20 000 images — les figures du rapport.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# src/ doit être importable quel que soit le dossier de lancement.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data_import import load_dataset  # noqa: E402
from gan.algo import GAN  # noqa: E402

FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist",
                        choices=["mnist", "quickdraw", "quickdraw10"])
    parser.add_argument("--n-train", type=int, default=20000,
                        help="Images d'entraînement (0 = tout le split train).")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--g-hidden", default="256-512",
                        choices=["256-512", "128-256", "256-512-1024"])
    parser.add_argument("--d-hidden", default="512-256",
                        choices=["512-256", "256-128", "1024-512-256"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Réglages rapides : 10 époques, 5 000 images.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 10)
        args.n_train = min(args.n_train or 5000, 5000)

    ds = load_dataset(args.dataset)
    n = len(ds.X_train) if args.n_train == 0 else min(args.n_train, len(ds.X_train))
    X = ds.X_train[:n]
    print(f"Dataset : {ds.describe()}")
    print(f"Entraînement sur {n} images · {args.epochs} époques · seed {args.seed}")

    algo = GAN()
    params = {
        "latent_dim": args.latent_dim,
        "g_hidden": args.g_hidden,
        "d_hidden": args.d_hidden,
        "learning_rate": 2e-4,
        "label_smoothing": 0.9,
        "batch_size": 128,
        "epochs": args.epochs,
        "seed": args.seed,
        "n_samples": n,
        "progress": lambda frac, desc="": print(f"  {desc}", flush=True),
    }
    algo.check(params, n)

    weights, meta = algo.train(X, params)
    # Les mêmes clés que l'app ajoute après train (voir src/app.py) : le modèle
    # est indistinguable d'un modèle entraîné depuis l'interface.
    meta.update({"algo": "gan", "dataset": args.dataset, "n_samples": n})

    # Même convention de nommage que l'app : {algo}_{auto_name} (le dataset est
    # déjà dans auto_name, et dans les métadonnées pour le filtrage).
    name = f"{algo.key}_{algo.auto_name(args.dataset, params)}"
    path = algo.save(name, weights, meta)
    print(f"Modèle sauvegardé : {path}")

    # ------------------------------------------------------------- Figures
    out = FIGURES_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, fname):
        fig.savefig(out / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  figure : {out / fname}")

    n_viz = min(3000, len(ds.X_test))
    labels = algo.assign(ds.X_train[:n_viz], weights)

    extras = algo.extra_figures_with_data(
        ds.X_train[:n_viz], weights, meta, labels,
        ds.y_train[:n_viz], ds.class_names,
    )
    for i, (title, fig) in enumerate(extras, start=1):
        slug = (title.split("—")[0].strip().lower()
                .replace(" ", "_").replace("/", "-"))
        save(fig, f"{i:02d}_{slug}.png")

    fig, _ = algo.plot_dictionary(weights, meta, None, None, None)
    save(fig, "06_dictionnaire_generateur.png")

    fig, _ = algo.plot_curve(meta["inertia_history"], n)
    save(fig, "07_loss_generateur.png")

    fig = algo.plot_latent(ds.X_test[:n_viz], weights, meta,
                           ds.y_test[:n_viz], ds.class_names,
                           "Classe réelle", f"GAN — espace latent inversé ({args.dataset}, test)")
    save(fig, "08_espace_latent_test.png")

    # La « compression » : 10 images de test, leur inversion, la reconstruction.
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds.X_test), size=10, replace=False)
    originals = ds.X_test[idx]
    codes = [algo.encode(img, weights) for img in originals]
    recons = [algo.decode(c, weights) for c in codes]
    fig = algo.plot_reconstructions(
        originals, recons,
        top_labels=[ds.class_names[y] for y in ds.y_test[idx]],
        bottom_labels=[str(c) for c in codes],
        title="GAN comme codec — la démonstration par l'échec "
              "(inversion par banque, voir rapport)",
    )
    save(fig, "09_codec_inversion.png")

    mse = meta["inertia"] / (meta["n_samples"] * 784)
    print(f"\nTerminé. MSE d'inversion (banque) : {mse:.4f} — à comparer aux "
          f"codecs entraînés pour compresser (PCA, autoencodeur).")


if __name__ == "__main__":
    main()
