"""Figures du GAN côté app.

Les vues génériques — reconstructions côte à côte, nuage latent, distribution
des classes, grille d'images générées — sont réutilisées telles quelles de
src/autoencoder/utils/visualization.py, comme le VAE le fait déjà : un GAN se
REGARDE comme les autres, seul le récit change (voir algo.py).

S'y ajoutent trois vues propres au GAN :
- les courbes du duel G / D (deux losses + deux accuracies) — la lecture de
  santé d'un entraînement adversarial, qui ne ressemble à aucune courbe de
  descente classique : on n'y cherche pas une baisse, on y cherche un ÉQUILIBRE ;
- la maturation du générateur : les MÊMES 16 z décodés à ~5 moments de
  l'entraînement ;
- l'interpolation latente entre deux z — la preuve que G a appris une surface
  continue d'images, pas un dictionnaire par cœur.
"""

import matplotlib.pyplot as plt
import numpy as np

# Réutilisées telles quelles : les figures sont identiques à celles de
# l'autoencodeur, on ne les redéfinit pas.
from autoencoder.utils.visualization import (  # noqa: F401
    plot_dominant_distribution,
    plot_generated_grid,
    plot_latent_panels,
    show_reconstructions,
)

IMG_SIZE = 28


def _as_image(flat):
    return np.asarray(flat, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)


def plot_gan_losses(g_loss, d_loss, d_real_acc, d_fake_acc, suptitle=""):
    """Le duel G / D au fil des époques — losses à gauche, accuracies à droite.

    Lecture : les deux losses d'un GAN sain OSCILLENT sans converger vers 0 —
    chaque réseau annule les progrès de l'autre. Les accuracies de D (vraies
    reconnues vraies, générées reconnues générées) donnent l'arbitre : ~100 %
    = D écrase G (gradient mort pour G), ~50 % = D ne voit plus la différence
    (G a gagné, ou s'est effondré sur un mode que D ne teste plus).
    """
    epochs = range(1, len(g_loss) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, g_loss, marker="o", markersize=3, label="Générateur")
    ax_loss.plot(epochs, d_loss, marker="o", markersize=3, label="Discriminateur")
    ax_loss.set_xlabel("Époque")
    ax_loss.set_ylabel("BCE")
    ax_loss.set_title("Losses — l'équilibre plutôt que la descente")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, np.asarray(d_real_acc) * 100, marker="o", markersize=3,
                label="Vraies → « vraie »")
    ax_acc.plot(epochs, np.asarray(d_fake_acc) * 100, marker="o", markersize=3,
                label="Générées → « générée »")
    ax_acc.axhline(50, color="gray", linestyle="--", linewidth=1,
                   label="50 % — D aveugle")
    ax_acc.set_xlabel("Époque")
    ax_acc.set_ylabel("Accuracy de D (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.set_title("Le verdict du discriminateur")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    fig.suptitle(suptitle or "GAN — le duel générateur / discriminateur",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_snapshot_grid(snapshots, suptitle=""):
    """La maturation de G : une colonne par instantané, les MÊMES z partout.

    snapshots : [(époque, images (n, 784)), ...]. Chaque LIGNE est un z fixe :
    la lire de gauche à droite montre le bruit devenir image au fil des époques.
    """
    n_cols = len(snapshots)
    n_rows = min(8, len(snapshots[0][1]))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.35 * n_cols, 1.3 * n_rows), squeeze=False)
    for j, (epoch, images) in enumerate(snapshots):
        for i in range(n_rows):
            ax = axes[i][j]
            ax.imshow(_as_image(images[i]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(f"ép. {epoch}", fontsize=9)
    fig.suptitle(suptitle or "Maturation du générateur — mêmes z, époques croissantes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_interpolation(rows, suptitle=""):
    """Interpolations latentes : chaque ligne va de G(z_a) à G(z_b) pas à pas.

    rows : liste de (n_steps, 784). Si toutes les images intermédiaires sont
    plausibles, G a appris une SURFACE d'images — c'est ce qui le sépare d'un
    dictionnaire de prototypes, qui ne saurait rien mettre « entre » deux entrées.
    """
    n_rows = len(rows)
    n_steps = len(rows[0])
    fig, axes = plt.subplots(n_rows, n_steps,
                             figsize=(1.3 * n_steps, 1.35 * n_rows), squeeze=False)
    for i, row in enumerate(rows):
        for j in range(n_steps):
            ax = axes[i][j]
            ax.imshow(_as_image(row[j]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                if j == 0:
                    ax.set_title("z_a", fontsize=9)
                elif j == n_steps - 1:
                    ax.set_title("z_b", fontsize=9)
    fig.suptitle(suptitle or "Interpolation latente — G((1−t)·z_a + t·z_b)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
