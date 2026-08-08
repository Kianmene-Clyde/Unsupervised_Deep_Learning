"""Figures de l'autoencodeur côté app.

Les vues signature du notebook autoencodeur.py — grille de l'espace latent 2D,
génération par échantillonnage gaussien — sont reprises ici, mais en fonctions
qui RETOURNENT leur figure (le notebook fait plt.show() et sauve sur disque :
l'app, elle, veut l'objet Figure pour Gradio) et qui reçoivent des IMAGES déjà
décodées : tout ce qui touche aux modèles Keras reste dans utils/model.py,
ces fonctions ne font que du matplotlib.

S'y ajoutent les vues que l'interface Algo exige (reconstructions côte à côte,
projection latente, distribution des classes) et une vue propre au réseau :
la traversée du décodeur dimension par dimension — l'équivalent continu d'un
dictionnaire de prototypes.
"""

import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 28


def _as_image(flat):
    return np.asarray(flat, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)


def show_reconstructions(originals, reconstructions, top_labels=None,
                         bottom_labels=None, title=""):
    """Originaux en haut, reconstructions en dessous — le format de l'onglet codec.

    Même lecture que pour les autres algos : chaque colonne est une image, la
    lire de haut en bas montre ce que la compression a perdu (ici, ce que
    latent_dim flottants ne suffisent pas à retenir).
    """
    tops = [_as_image(img) for img in originals]
    bottoms = [_as_image(img) for img in reconstructions]
    n = len(tops)
    if n == 0:
        raise ValueError("Aucune image à afficher.")
    if len(bottoms) != n:
        raise ValueError(
            f"{n} originaux mais {len(bottoms)} reconstructions : il en faut autant."
        )

    fig, axes = plt.subplots(2, n, figsize=(n * 1.35, 3.6))
    axes = np.array(axes).reshape(2, n)

    for col in range(n):
        for row, (imgs, labels) in enumerate(
            ((tops, top_labels), (bottoms, bottom_labels))
        ):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if labels is not None:
                # fontsize 7 : un code « z=(+1.2, −0.4, …) » est plus long
                # qu'un « code 37 » de K-means.
                ax.set_title(str(labels[col]), fontsize=7)

    for row, name in ((0, "Original"), (1, "Reconstruit")):
        ax = axes[row, 0]
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_ylabel(name, fontsize=9, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_latent_panels(coords, y=None, class_names=None, labels=None,
                       title="", y_label="Classe réelle"):
    """z₁ × z₂ (et z₁ × z₂ × z₃) côte à côte, en UNE figure.

    C'est la vue « espace latent » de l'autoencodeur — et comme pour la PCA,
    ces axes SONT le code : les sorties brutes de l'encodeur, pas une
    projection d'illustration.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[:, None]
    if coords.shape[1] < 2:
        # latent_dim=1 : l'axe vertical n'existe pas, on le pose à 0.
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])

    has_3d = coords.shape[1] >= 3

    fig = plt.figure(figsize=(13, 6) if has_3d else (7, 6))
    ax2d = fig.add_subplot(1, 2, 1) if has_3d else fig.add_subplot(1, 1, 1)
    ax3d = fig.add_subplot(1, 2, 2, projection="3d") if has_3d else None

    if y is not None:
        y = np.asarray(y).astype(int).ravel()
        names = class_names or [str(i) for i in range(int(y.max()) + 1)]
        cmap = plt.get_cmap("tab20" if len(names) > 10 else "tab10")
        for i, cname in enumerate(names):
            m = y == i
            if not m.any():
                continue
            color = cmap(i % cmap.N)
            ax2d.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.4,
                         color=color, label=cname)
            if ax3d is not None:
                ax3d.scatter(coords[m, 0], coords[m, 1], zs=coords[m, 2],
                             s=6, alpha=0.4, color=color, label=cname)
        ax2d.legend(title=y_label, markerscale=3, fontsize=8)
    else:
        # Sans classes réelles, la dimension dominante colore le nuage.
        c = labels if labels is not None else "tab:blue"
        sc = ax2d.scatter(coords[:, 0], coords[:, 1], c=c, cmap="tab20",
                          s=6, alpha=0.4)
        if ax3d is not None:
            ax3d.scatter(coords[:, 0], coords[:, 1], zs=coords[:, 2], c=c,
                         cmap="tab20", s=6, alpha=0.4)
        if labels is not None:
            fig.colorbar(sc, ax=ax2d, label="Dimension dominante")

    ax2d.set_xlabel("z₁")
    ax2d.set_ylabel("z₂")
    ax2d.set_title("z₁ × z₂")
    if ax3d is not None:
        ax3d.set_xlabel("z₁")
        ax3d.set_ylabel("z₂")
        ax3d.set_zlabel("z₃")
        ax3d.set_title("z₁ × z₂ × z₃")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_dominant_distribution(codes, y_true, class_names, k, title="",
                               max_shown=40):
    """Classes réelles × dimension latente dominante, en carte de chaleur.

    L'équivalent autoencodeur de la « distribution des classes par cluster » :
    le réseau n'assignant pas de cluster, chaque image est rangée sous la
    dimension latente qui s'écarte le plus de sa moyenne (voir Algo.assign).
    """
    codes = np.asarray(codes).astype(int).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    if codes.shape != y_true.shape:
        raise ValueError(
            f"codes a {codes.size} éléments mais y_true {y_true.size} : "
            f"passe la même tranche."
        )

    n_classes = len(class_names)
    counts = np.zeros((n_classes, k), dtype=np.int64)
    np.add.at(counts, (y_true, codes), 1)

    purity = counts.max(axis=0).sum() / counts.sum() if counts.sum() else 0.0

    order = np.argsort(-counts.sum(axis=0))[:min(k, max_shown)]
    shown = np.sort(order)
    hidden = k - len(shown)

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.30 * len(shown) + 2.5), 0.45 * n_classes + 2.4)
    )
    im = ax.imshow(counts[:, shown], aspect="auto", cmap="viridis")
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(class_names)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"z{i + 1}" for i in shown], rotation=90, fontsize=7)
    ax.set_xlabel("Dimension latente dominante (|zᵢ − z̄ᵢ| maximal)")
    ax.set_ylabel("Classe réelle")
    fig.colorbar(im, ax=ax, label="Images")

    subtitle = f"{k} dimensions · pureté globale : {purity * 100:.1f} %"
    if hidden > 0:
        subtitle += f" · {hidden} dimension(s) peu peuplée(s) masquée(s)"
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=11)
    fig.tight_layout()
    return fig


def plot_traversal_grid(images, ts, dim_names, suptitle=""):
    """La traversée du décodeur : une ligne par dimension latente.

    images : (n_dims, n_ts, 784) — decode(z̄ + t·σᵢ·eᵢ) pour chaque dimension i
    et chaque pas t. La colonne t=0 est identique sur toutes les lignes (c'est
    l'image du latent moyen) : ce qui change le long d'une ligne est ce que
    CETTE dimension encode.
    """
    n_dims, n_ts = len(images), len(ts)
    fig, axes = plt.subplots(
        n_dims, n_ts, figsize=(1.3 * n_ts, 1.35 * n_dims), squeeze=False
    )
    for i in range(n_dims):
        for j in range(n_ts):
            ax = axes[i][j]
            ax.imshow(_as_image(images[i][j]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if i == 0:
                ax.set_title(f"{ts[j]:+.0f}σ" if ts[j] else "z̄", fontsize=9)
            if j == 0:
                ax.set_ylabel(dim_names[i], fontsize=9, fontweight="bold")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_manifold(canvas, z1_range, z2_range, suptitle=""):
    """La grille de l'espace latent 2D — la vue signature du notebook.

    canvas : la mosaïque (grid·28, grid·28) des images décodées, déjà
    assemblée ; chaque position (z₁, z₂) du plan a été décodée en image.
    """
    z1_min, z1_max = z1_range
    z2_min, z2_max = z2_range
    fig = plt.figure(figsize=(9, 9))
    plt.imshow(
        canvas, cmap="gray",
        extent=[z1_min, z1_max, z2_min, z2_max],
        origin="upper", aspect="auto",
    )
    plt.xlabel("Dimension latente z₁")
    plt.ylabel("Dimension latente z₂")
    plt.title(
        suptitle or "Visualisation directe de l'espace latent 2D\n"
                    "Chaque position (z₁, z₂) est décodée en image"
    )
    fig.tight_layout()
    return fig


def make_manifold_canvas(decoded, grid_size):
    """Assemble les grid² images décodées en une seule mosaïque."""
    canvas = np.zeros((IMG_SIZE * grid_size, IMG_SIZE * grid_size), dtype=np.float32)
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            canvas[
                row * IMG_SIZE:(row + 1) * IMG_SIZE,
                col * IMG_SIZE:(col + 1) * IMG_SIZE,
            ] = _as_image(decoded[index])
            index += 1
    return canvas


def plot_generated_grid(images, z_titles=None, suptitle=""):
    """Les images générées par échantillonnage du latent — reprise du notebook.

    L'autoencodeur simple n'étant pas régularisé comme un VAE, les z sont tirés
    de la gaussienne EMPIRIQUE des vrais codes (moyenne + covariance mesurées),
    pas d'une N(0, I) qui tomberait à côté du nuage.
    """
    n = len(images)
    n_cols = int(np.ceil(np.sqrt(n)))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.9 * n_cols, 2.0 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(_as_image(images[i]), cmap="gray", vmin=0, vmax=1)
            if z_titles is not None:
                ax.set_title(z_titles[i], fontsize=7)
    fig.suptitle(
        suptitle or "Images générées — gaussienne empirique de l'espace latent",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_loss_curve(mses, suptitle=""):
    """La MSE par pixel au fil des époques — la courbe d'entraînement du réseau."""
    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mses) + 1), mses, marker="o", markersize=3)
    plt.xlabel("Époque")
    plt.ylabel("MSE de reconstruction (par pixel)")
    plt.title(suptitle or "Autoencodeur — évolution de la loss")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
