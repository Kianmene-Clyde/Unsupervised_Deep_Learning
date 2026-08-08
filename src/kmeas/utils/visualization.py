import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show_centroid(centroid, title="Centroïde reconstruit", show=True):
    image = np.asarray(tf.reshape(centroid, (28, 28)))

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def show_reconstructions(originals, reconstructions, top_labels=None,
                         bottom_labels=None, title="", figsize=None, show=True):
    """Compare des images à leur reconstruction : originaux en haut, codecs en bas.

    Chaque colonne est une image ; la lire de haut en bas montre ce que le codec
    a perdu. Une seule image ne dit rien de la qualité d'un codec — c'est en en
    alignant une dizaine qu'on voit s'il tient sur des cas variés.

    Args:
        originals:      (n, 784) ou (n, 28, 28) — les images d'entrée.
        reconstructions:(n, ...) — leurs reconstructions, même ordre.
        top_labels:     titre de chaque original (p. ex. la classe réelle).
        bottom_labels:  titre de chaque reconstruction (p. ex. le code transmis).
        title:          bandeau au-dessus de la figure (p. ex. « TRAIN »).
        show:           True -> plt.show() (notebook) ; False -> retourne la figure.
    """
    tops = [np.asarray(tf.reshape(img, (28, 28))) for img in originals]
    bottoms = [np.asarray(tf.reshape(img, (28, 28))) for img in reconstructions]
    n = len(tops)
    if n == 0:
        raise ValueError("Aucune image à afficher.")
    if len(bottoms) != n:
        raise ValueError(
            f"{n} originaux mais {len(bottoms)} reconstructions : il en faut autant."
        )

    fig, axes = plt.subplots(2, n, figsize=figsize or (n * 1.35, 3.6))
    axes = np.array(axes).reshape(2, n)

    for col in range(n):
        for row, (imgs, labels) in enumerate(
            ((tops, top_labels), (bottoms, bottom_labels))
        ):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if labels is not None:
                ax.set_title(str(labels[col]), fontsize=8)

    # Les libellés de gauche repèrent les deux lignes sans coûter une colonne.
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruit")
    for row, name in ((0, "Original"), (1, "Reconstruit")):
        axes[row, 0].axis("on")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for side in axes[row, 0].spines.values():
            side.set_visible(False)
        axes[row, 0].set_ylabel(name, fontsize=9, fontweight="bold")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_class_distribution(cluster_labels, y_true, class_names=None, k=None,
                            clusters_per_chart=20, normalize=False,
                            title="Distribution des classes réelles par cluster",
                            figsize=None, show=True):
    """Bar chart groupé : de quelles classes réelles chaque cluster est composé.

    Chaque cluster reçoit un groupe de barres fines, une par classe réelle, côte
    à côte. Un groupe dominé par une seule barre = cluster pur ; plusieurs barres
    de hauteur voisine = cluster qui mélange des classes. Les barres étant
    juxtaposées et non empilées, leurs hauteurs se comparent directement.

    C'est la vue qui répond à « mes clusters retrouvent-ils mes vraies classes ? »,
    ce que l'inertie ne dit pas : elle ne mesure que la compression.

    K pouvant valoir des centaines, les clusters sont répartis sur plusieurs
    sous-graphiques : au-delà d'une vingtaine de groupes, les barres deviennent
    trop fines pour être lues.

    Args:
        cluster_labels:     (n,) cluster attribué à chaque image.
        y_true:             (n,) classe réelle de chaque image.
        class_names:        noms lisibles des classes ; indices si None.
        k:                  nombre de clusters ; déduit des labels si None.
        clusters_per_chart: groupes de barres par sous-graphique.
        normalize:          True -> hauteurs en % du cluster (composition pure,
                            tailles masquées) ; False -> hauteur = nb d'images.
        show:               True -> plt.show() ; False -> retourne la figure.
    """
    cluster_labels = np.asarray(cluster_labels).astype(int).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()
    if cluster_labels.shape != y_true.shape:
        raise ValueError(
            f"cluster_labels a {cluster_labels.size} éléments mais y_true "
            f"{y_true.size} : passe la même tranche."
        )

    k = int(k if k is not None else cluster_labels.max() + 1)
    n_classes = int(y_true.max() + 1)
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    n_classes = max(n_classes, len(class_names))

    # counts[classe, cluster] : effectif de chaque classe dans chaque cluster.
    counts = np.zeros((n_classes, k), dtype=np.int64)
    np.add.at(counts, (y_true, cluster_labels), 1)

    sizes = counts.sum(axis=0)
    # Pureté : part des images tombant dans un cluster où leur classe est
    # majoritaire. 1.0 = clusters parfaitement alignés sur les vraies classes.
    purity = counts.max(axis=0).sum() / counts.sum() if counts.sum() else 0.0

    heights = counts.astype(float)
    if normalize:
        # where= évite la division par zéro des clusters vides (0/0 -> 0).
        heights = np.divide(heights, np.where(sizes > 0, sizes, 1) * 1.0) * 100

    per_chart = max(1, int(clusters_per_chart))
    n_charts = int(np.ceil(k / per_chart))
    colors = plt.get_cmap("tab20" if n_classes > 10 else "tab10")

    # Barres juxtaposées : chaque cluster reçoit une case de largeur 1, partagée
    # entre ses n_classes barres. Le facteur 0.75 les affine et laisse respirer
    # les groupes voisins, pour qu'on voie où finit un cluster.
    group_width = 0.75
    bar_width = group_width / n_classes

    # La largeur doit suivre le nombre TOTAL de barres : 20 clusters x 10 classes
    # font 200 barres, illisibles dans la largeur qui suffisait à 20 barres.
    n_bars = min(per_chart, k) * n_classes
    width = float(np.clip(n_bars * 0.085 + 2.5, 7.0, 26.0))

    # Hauteur du bandeau titre + légende, en POUCES. Le raisonner en fraction de
    # la figure donne une réserve qui enfle avec le nombre de sous-graphiques :
    # 20 % suffisent à 3 sous-graphiques mais ouvrent un gouffre blanc à 25.
    legend_rows = int(np.ceil(n_classes / 10))
    header_in = 0.95 + 0.30 * legend_rows
    body_in = 2.9 * n_charts
    fig_height = body_in + header_in

    fig, axes = plt.subplots(
        n_charts, 1,
        figsize=figsize or (width, fig_height),
        squeeze=False,
    )
    axes = axes.ravel()

    for chart, ax in enumerate(axes):
        lo = chart * per_chart
        hi = min(lo + per_chart, k)
        idx = np.arange(lo, hi)

        for cls in range(n_classes):
            # Décalage de chaque classe autour du centre de sa case.
            offset = (cls - (n_classes - 1) / 2) * bar_width
            ax.bar(idx + offset, heights[cls, lo:hi],
                   width=bar_width * 0.88,   # gap fin entre barres voisines
                   color=colors(cls % colors.N),
                   label=class_names[cls] if chart == 0 else None)

        ax.set_xticks(idx)
        ax.set_xticklabels(idx, fontsize=8, rotation=90 if hi - lo > 14 else 0)
        ax.set_xlim(lo - 0.6, hi - 0.4)
        ax.set_ylabel("Part du cluster (%)" if normalize else "Images")
        ax.set_title(f"Clusters {lo} à {hi - 1}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Séparateurs entre clusters : sans eux, les groupes de barres fines se
        # confondent visuellement avec leurs voisins.
        for boundary in idx[:-1]:
            ax.axvline(boundary + 0.5, color="0.85", linewidth=0.6, zorder=0)

        empty = int((sizes[lo:hi] == 0).sum())
        if empty:
            ax.text(0.998, 0.96, f"{empty} cluster(s) vide(s)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="tab:red")

    axes[-1].set_xlabel("Cluster (numéro = code du codec)")

    # Les axes s'arrêtent sous le bandeau ; titre et légende l'occupent. Tout est
    # positionné en pouces convertis en fraction, pour que l'écart reste constant
    # quel que soit le nombre de sous-graphiques.
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig_height))

    fig.suptitle(
        f"{title}\n{k} clusters · pureté globale : {purity * 100:.1f} %",
        fontsize=12, y=1 - 0.06 / fig_height, va="top",
    )

    # Légende hors des axes : posée dans le graphique, elle recouvrait les barres
    # les plus hautes. Une seule pour toute la figure, la répéter serait du bruit.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Classe réelle", fontsize=8,
               ncol=min(n_classes, 10), loc="upper center",
               bbox_to_anchor=(0.5, 1 - 0.62 / fig_height), framealpha=0.9)

    if show:
        plt.show()
    return fig


def show_images(images, titles=None, n_rows=1, n_cols=None, figsize=None, show=True):
    """Affiche plusieurs images 28x28 dans une seule figure (grille).

    Args:
        images:  liste/tableau d'images, chacune (784,) ou (28, 28).
        titles:  liste de titres, un par image (optionnel).
        n_rows:  nombre de lignes de la grille.
        n_cols:  nombre de colonnes ; déduit de n_rows si None.
        figsize: taille de la figure ; déduite de la grille si None.
        show:    True -> plt.show() (notebook) ; False -> retourne la figure (app).

    Exemple (4 images en colonne) :
        show_images([img1, img2, img3, img4], titles=[...], n_rows=4, n_cols=1)
    """
    # Convertit chaque image en (28, 28) numpy, quel que soit le format d'entrée
    imgs = [np.asarray(tf.reshape(img, (28, 28))) for img in images]
    n = len(imgs)

    if n_cols is None:
        n_cols = int(np.ceil(n / n_rows))
    if figsize is None:
        figsize = (n_cols * 3, n_rows * 3)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)  # aplatit, gère le cas d'un seul axe

    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i])
        ax.axis("off")  # masque aussi les cases vides éventuelles

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_centroid_map(centroids, cluster_labels=None, y_true=None, n_cols=None,
                      sort_by_size=True, cell_size=0.6,
                      title="Cartographie des centroïdes", show=True):
    """Affiche tous les centroïdes en grille : la carte du dictionnaire appris.

    Args:
        centroids:      (k, 784) centroïdes du K-means.
        cluster_labels: (n,) index de cluster de chaque point ; nécessaire pour
                        trier par taille de cluster et pour déduire le chiffre
                        dominant.
        y_true:         (n,) labels réels ; annote chaque case du chiffre
                        majoritaire de son cluster (nécessite cluster_labels).
        n_cols:         nombre de colonnes ; déduit en grille ~carrée si None.
        sort_by_size:   True -> centroïdes triés du cluster le plus peuplé au
                        moins peuplé (les prototypes utiles d'abord, les
                        clusters vides à la fin).
        cell_size:      taille d'une case en pouces.
        show:           True -> plt.show() (notebook) ; False -> retourne la figure (app).
    """
    C = np.asarray(tf.reshape(centroids, (-1, 28, 28)))
    k = C.shape[0]

    sizes = None
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels).astype(int).ravel()
        sizes = np.bincount(cluster_labels, minlength=k)

    # Ordre d'affichage
    order = np.arange(k)
    if sort_by_size:
        if sizes is None:
            raise ValueError("sort_by_size=True nécessite cluster_labels.")
        order = np.argsort(-sizes)

    # Chiffre dominant de chaque cluster
    dominant = None
    if y_true is not None:
        if cluster_labels is None:
            raise ValueError("y_true nécessite cluster_labels pour être exploité.")
        y_true = np.asarray(y_true).astype(int).ravel()
        if y_true.shape[0] != cluster_labels.shape[0]:
            raise ValueError(
                f"y_true a {y_true.shape[0]} éléments mais cluster_labels en a "
                f"{cluster_labels.shape[0]}. Passe la même tranche."
            )
        dominant = np.full(k, -1)
        for c in range(k):
            members = y_true[cluster_labels == c]
            if members.size:
                dominant[c] = np.bincount(members, minlength=10).argmax()

    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(k)))
    n_rows = int(np.ceil(k / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_size, n_rows * cell_size))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= k:
            continue
        c = order[i]
        empty = sizes is not None and sizes[c] == 0
        # Un cluster vide garde le centroïde de son initialisation : on le
        # grise pour ne pas le lire comme un prototype appris.
        ax.imshow(C[c], cmap="gray", vmin=0, vmax=1, alpha=0.25 if empty else 1.0)
        if dominant is not None and not empty:
            ax.text(0.5, -0.02, str(dominant[c]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=cell_size * 11, color="tab:blue")

    subtitle = f"{k} centroïdes"
    if sizes is not None:
        subtitle += f" — {int((sizes == 0).sum())} cluster(s) vide(s)"
        if sort_by_size:
            subtitle += ", triés par taille décroissante"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)
    # tight_layout réserve la place des titres de chaque case et disperse la
    # grille : on règle l'espacement à la main pour garder une carte dense.
    top = 1 - 0.6 / (n_rows * cell_size)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=top,
                        wspace=0.05, hspace=0.35 if dominant is not None else 0.05)

    if show:
        plt.show()
    return fig


def plot_inertia(history, n_samples=None, image_dim=784,
                 title="Courbe d'entraînement — inertie par itération",
                 ylabel="Inertie (somme des distances²)", figsize=(8, 4.5),
                 show=True):
    """Trace l'inertie au fil des itérations : la courbe d'entraînement du K-means.

    L'inertie est ce que K-means minimise — c'est sa loss. Elle ne peut que
    décroître : les deux étapes de l'algorithme la réduisent tour à tour. Une
    courbe qui remonte trahit donc un bug, pas un mauvais réglage.

    Args:
        history:   liste des inerties, une par itération
                   (fit_kmeans(..., return_history=True)).
        n_samples: nombre d'images d'entraînement. Fourni, un second axe à droite
                   gradue la même courbe en MSE de reconstruction.

                   MSE et inertie ne diffèrent que d'un facteur (n · d) — voir
                   utils/metrics.py : la reconstruction d'une image étant son
                   centroïde, l'inertie mesure déjà l'erreur de reconstruction.
                   D'où un second axe plutôt qu'une seconde courbe, qui serait
                   rigoureusement superposée à la première.
        image_dim: pixels par image (784 = 28x28).
        show:      True -> plt.show() (notebook) ; False -> retourne la figure (app).
    """
    history = np.asarray(history, dtype=np.float64).ravel()
    n = history.size
    if n == 0:
        raise ValueError("history est vide : rien à tracer.")

    iterations = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(iterations, history, marker="o", markersize=3, linewidth=1.5,
            color="tab:blue")

    start, end = history[0], history[-1]
    ax.annotate(f"départ : {start:,.0f}".replace(",", " "),
                xy=(1, start), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.annotate(f"final : {end:,.0f}".replace(",", " "),
                xy=(n, end), xytext=(-6, 10), textcoords="offset points",
                fontsize=9, ha="right", color="tab:green")
    ax.scatter([n], [end], color="tab:green", zorder=5, s=40)

    drop = (1 - end / start) * 100 if start else 0.0
    subtitle = f"{n} itération(s) — inertie réduite de {drop:.1f} %"

    ax.set_xlabel("Itération")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if n > 1:
        ax.set_xlim(0.5, n + 0.5)

    if n_samples:
        # Même courbe, seconde graduation : on convertit les bornes de l'axe de
        # gauche au lieu de retracer la série. Les deux axes restent ainsi
        # forcément alignés, quoi que fasse matplotlib sur l'échelle.
        scale = int(n_samples) * int(image_dim)
        ax_mse = ax.twinx()
        ax_mse.set_ylim(*(np.array(ax.get_ylim()) / scale))
        ax_mse.set_ylabel("MSE de reconstruction (par pixel)", color="tab:red")
        ax_mse.tick_params(axis="y", labelcolor="tab:red")
        subtitle += f" · MSE finale {end / scale:.4f}"

    # La boucle s'arrête dès que les centroïdes ne bougent plus : le nombre
    # d'itérations affiché EST le point de convergence, sauf si max_iter l'a
    # coupée avant.
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    fig.tight_layout()
    if show:
        plt.show()
    return fig