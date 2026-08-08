"""Figures PCA côté app — capture des tracés notebook + vues exigées par l'app.

--- Capturer plutôt que réécrire ---

Les fonctions de tracé d'utils/pca (plot_spectrum, plot_eigenimages,
plot_generated_grid…) sont écrites pour les notebooks : elles appellent
plt.show() et ne RETOURNENT pas leur figure. L'app, elle, a besoin de l'objet
Figure pour le donner à Gradio. Plutôt que de modifier utils/pca (contrat :
on n'y touche pas), `captured()` relève les numéros de figures avant/après
l'appel et ressort la figure créée — plt.show() étant muet sous le backend
Agg que l'app impose, rien ne s'affiche au passage.

--- Le reste ---

Trois vues que l'interface Algo exige et qu'utils/pca n'a pas sous cette
forme : les reconstructions côte à côte (le codec), la projection 2D + 3D en
UNE figure (utils/pca en fait deux, l'app n'a qu'un emplacement par split), et
la distribution des classes par composante dominante (la PCA n'a pas de
clusters : on range chaque image sous la composante où |zᵢ| est maximal).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

IMG_SIZE = 28


def captured(plot_fn, *args, **kwargs):
    """Appelle une fonction de tracé notebook et capture la figure qu'elle crée.

    Repose sur le gestionnaire global de pyplot : toute figure créée pendant
    l'appel y est enregistrée, show() ou pas. Si la fonction en crée plusieurs,
    on garde la dernière et on ferme les autres (sinon elles fuient : personne
    d'autre n'en tient de référence).
    """
    before = set(plt.get_fignums())
    with warnings.catch_warnings():
        # Chaque plt.show() des tracés notebook avertit qu'Agg n'affiche rien —
        # c'est précisément ce qu'on attend de lui : silence.
        warnings.filterwarnings(
            "ignore", message="FigureCanvasAgg is non-interactive"
        )
        plot_fn(*args, **kwargs)
    new = [plt.figure(num) for num in plt.get_fignums() if num not in before]
    if not new:
        raise RuntimeError(f"{plot_fn.__name__} n'a créé aucune figure à capturer.")
    for fig in new[:-1]:
        plt.close(fig)
    return new[-1]


def show_reconstructions(originals, reconstructions, top_labels=None,
                         bottom_labels=None, title=""):
    """Originaux en haut, reconstructions PCA en dessous — le format de l'onglet codec.

    Même lecture que pour les autres algos : chaque colonne est une image, la
    lire de haut en bas montre ce que la compression a perdu (ici, la variance
    des composantes abandonnées).
    """
    tops = [np.asarray(img).reshape(IMG_SIZE, IMG_SIZE) for img in originals]
    bottoms = [np.asarray(img).reshape(IMG_SIZE, IMG_SIZE) for img in reconstructions]
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
                # fontsize 7 et non 8 : un code PCA « code z=(+1.2, −0.4, …) »
                # est plus long qu'un « code 37 » de K-means.
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


def plot_projection_panels(coords, y=None, class_names=None, labels=None,
                           ratios=None, title="", y_label="Classe réelle"):
    """PC1 × PC2 et PC1 × PC2 × PC3 côte à côte, en UNE figure.

    C'est la vue « espace latent » de la PCA — et contrairement aux autres
    algos, ces axes SONT le code : pas une projection d'illustration.

    Args:
        coords:      (n, k) les codes PCA (pca.transform).
        y:           (n,) classes réelles ; colore par classe si fourni.
        class_names: noms lisibles des classes.
        labels:      (n,) composante dominante ; colorage de repli quand y=None.
        ratios:      variance expliquée par composante — annotée sur les axes.
        y_label:     titre de la légende (« Chiffre réel » sur MNIST).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 1:
        coords = coords[:, None]
    if coords.shape[1] < 2:
        # k=1 : un seul axe appris ; l'axe vertical n'existe pas, on le pose à 0.
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        ratios = None

    has_3d = coords.shape[1] >= 3

    def axis_name(i):
        if ratios is not None and i < len(ratios):
            return f"PC{i + 1} ({ratios[i] * 100:.1f} % de la variance)"
        return f"PC{i + 1}"

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
        # Sans classes réelles, la composante dominante colore le nuage : on
        # voit quelles régions du plan chaque axe « possède ».
        c = labels if labels is not None else "tab:blue"
        sc = ax2d.scatter(coords[:, 0], coords[:, 1], c=c, cmap="tab20",
                          s=6, alpha=0.4)
        if ax3d is not None:
            ax3d.scatter(coords[:, 0], coords[:, 1], zs=coords[:, 2], c=c,
                         cmap="tab20", s=6, alpha=0.4)
        if labels is not None:
            fig.colorbar(sc, ax=ax2d, label="Composante dominante")

    ax2d.set_xlabel(axis_name(0))
    ax2d.set_ylabel(axis_name(1))
    ax2d.set_title("PC1 × PC2")
    if ax3d is not None:
        ax3d.set_xlabel("PC1")
        ax3d.set_ylabel("PC2")
        ax3d.set_zlabel("PC3")
        ax3d.set_title("PC1 × PC2 × PC3")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_dominant_distribution(codes, y_true, class_names, k, title="",
                               max_shown=40):
    """Classes réelles × composante dominante, en carte de chaleur.

    L'équivalent PCA de la « distribution des classes par cluster » : la PCA
    n'assignant pas de cluster, chaque image est rangée sous la composante où
    |zᵢ| est maximal. Une carte plutôt que des barres groupées : k peut valoir
    des centaines, et une ligne de 400 groupes de barres est illisible.
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

    # Pureté, au même sens que pour les clusters : part des images dont la
    # classe est majoritaire sous leur composante dominante.
    purity = counts.max(axis=0).sum() / counts.sum() if counts.sum() else 0.0

    # Seules les composantes les plus peuplées sont montrées : au-delà, des
    # colonnes vides. On les réordonne par indice pour garder la lecture PC1→PCk.
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
    ax.set_xticklabels([f"PC{i + 1}" for i in shown], rotation=90, fontsize=7)
    ax.set_xlabel("Composante dominante (|zᵢ| maximal)")
    ax.set_ylabel("Classe réelle")
    fig.colorbar(im, ax=ax, label="Images")

    subtitle = f"{k} composantes · pureté globale : {purity * 100:.1f} %"
    if hidden > 0:
        subtitle += f" · {hidden} composante(s) peu peuplée(s) masquée(s)"
    ax.set_title(f"{title}\n{subtitle}" if title else subtitle, fontsize=11)
    fig.tight_layout()
    return fig
