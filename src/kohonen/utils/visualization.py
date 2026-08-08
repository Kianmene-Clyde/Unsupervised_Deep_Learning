"""Visualisations de la carte de Kohonen.

L'espace latent du SOM est un entier discret (le neurone gagnant) : il ne se
projette pas, il se DÉCRIT. La carte des prototypes montre le dictionnaire à sa
place sur la grille, et les distributions par neurone montrent quelles classes
réelles chaque code regroupe.
"""

import numpy as np
import matplotlib.pyplot as plt

from .kohonen import grid_coords


def show_prototype(weight, title="Feature vector du neurone", show=True):
    image = np.asarray(weight, dtype=np.float32).reshape(28, 28)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

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
        show:    True -> plt.show() (notebook) ; False -> retourne la figure.
    """
    imgs = [np.asarray(img, dtype=np.float32).reshape(28, 28) for img in images]
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
    tops = [np.asarray(img, dtype=np.float32).reshape(28, 28) for img in originals]
    bottoms = [np.asarray(img, dtype=np.float32).reshape(28, 28) for img in reconstructions]
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


def neighbor_distance(weights, rows, cols):
    """(k,) distance moyenne de chaque neurone à ses voisins immédiats — l'U-matrix.

    « Voisin immédiat » = à distance 1 dans l'espace TOPOLOGIQUE : les 6 voisins
    d'un hexagone. Aucun décalage n'est codé à la main — on relit les coordonnées
    qui ont servi à l'entraînement.

    Grande valeur = le neurone ne ressemble pas à ses voisins = on est sur une
    frontière entre deux groupes. C'est l'information que ni la carte des
    prototypes ni celle des classes ne donnent : la grille est régulière, les
    distances réelles entre voisins ne le sont pas.
    """
    coords = grid_coords(rows, cols)
    W = np.asarray(weights, dtype=np.float32)

    d_grid = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    adjacent = np.abs(d_grid - 1.0) < 1e-3          # (k, k) booléen

    out = np.zeros(len(W), dtype=np.float32)
    for i in range(len(W)):
        neighbors = W[adjacent[i]]
        out[i] = np.linalg.norm(neighbors - W[i], axis=1).mean() if len(neighbors) else 0.0
    return out


def plot_heatmap(weights, rows, cols, cluster_labels=None, y_true=None,
                 class_names=None, color_by="umatrix",
                 annotate=True, cmap=None, title=None, figsize=None, show=True):
    """Heatmap de la carte de Kohonen : une case = un neurone, à sa place.

    Args:
        weights:        (k, 784) feature vectors, k = rows · cols.
        rows, cols:     forme de la grille.
        color_by:       ce que la couleur encode —
                        "umatrix" (défaut) : distance moyenne aux voisins. La
                            heatmap canonique d'un SOM. Clair = frontière entre
                            groupes, sombre = intérieur d'un groupe.
                        "density" : nombre d'images gagnées (demande
                            cluster_labels). Les cases à 0 sont les neurones morts.
                        "class"   : classe réelle majoritaire (demande
                            cluster_labels + y_true).
        annotate:       écrit la valeur dans chaque case.
        cmap:           colormap ; un défaut adapté à color_by si None.

                        Éviter "jet" et "rainbow", tentantes mais trompeuses :
                        leur luminosité n'est pas monotone, elles inventent des
                        crêtes jaunes là où la distance ne fait que passer et
                        aplatissent de vrais écarts dans le vert. Les défauts
                        ci-dessous sont perceptuellement uniformes : un écart de
                        couleur y vaut partout le même écart de valeur.
        show:           True -> plt.show() (notebook) ; False -> retourne la figure.

    Note : la heatmap est une grille rectangulaire. Les voisinages sont bien
    calculés en hexagonal (6 voisins), mais l'AFFICHAGE ne décale pas les lignes
    impaires d'une demi-case — deux cases mitoyennes à l'écran sur deux lignes
    différentes ne sont donc pas forcément voisines. La carte des feature
    vectors, elle, respecte le décalage.
    """
    rows, cols = int(rows), int(cols)
    k = rows * cols
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape[0] != k:
        raise ValueError(
            f"{weights.shape[0]} feature vectors pour une grille {rows}x{cols} = {k} cases."
        )

    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels).astype(int).ravel()

    ticks = None
    tick_labels = None

    if color_by == "umatrix":
        values = neighbor_distance(weights, rows, cols)
        # inferno : du noir au jaune en passant par le pourpre et l'orange. La
        # luminosité y croît avec la valeur, donc « clair = frontière » reste
        # vrai — la couleur ne fait qu'écarter davantage les paliers voisins.
        cmap = plt.get_cmap(cmap or "inferno")
        cbar_label = "Distance moyenne aux voisins"
        fmt = "{:.1f}"
        default_title = "Carte de Kohonen — U-matrix"

    elif color_by == "density":
        if cluster_labels is None:
            raise ValueError("color_by='density' demande cluster_labels.")
        values = np.bincount(cluster_labels, minlength=k).astype(np.float32)
        cmap = plt.get_cmap(cmap or "viridis")
        cbar_label = "Images gagnées"
        fmt = "{:.0f}"
        default_title = "Carte de Kohonen — densité"

    elif color_by == "class":
        if cluster_labels is None or y_true is None:
            raise ValueError("color_by='class' demande cluster_labels ET y_true.")
        y_true = np.asarray(y_true).astype(int).ravel()
        if y_true.shape[0] != cluster_labels.shape[0]:
            raise ValueError(
                f"y_true a {y_true.shape[0]} éléments mais cluster_labels en a "
                f"{cluster_labels.shape[0]}. Passe la même tranche."
            )
        n_classes = int(y_true.max() + 1)
        if class_names is not None:
            n_classes = max(n_classes, len(class_names))

        values = np.full(k, np.nan, dtype=np.float32)
        for c in range(k):
            members = y_true[cluster_labels == c]
            if members.size:
                values[c] = np.bincount(members, minlength=n_classes).argmax()

        # Classes = catégories sans ordre : il faut une palette qualitative, pas
        # un dégradé — sinon « 8 » paraîtrait plus proche de « 9 » que de « 1 ».
        cmap = plt.get_cmap(cmap or ("tab10" if n_classes <= 10 else "tab20"),
                            n_classes).copy()
        # Un neurone mort n'a aucune classe : on le laisse en NaN plutôt que de
        # lui en attribuer une qu'il n'a jamais vue.
        cmap.set_bad("0.9")
        cbar_label = "Classe majoritaire"
        fmt = "{:.0f}"
        default_title = "Carte de Kohonen — classe majoritaire"
        ticks = range(n_classes)
        tick_labels = (list(class_names) if class_names is not None
                       else [str(i) for i in range(n_classes)])
    else:
        raise ValueError(
            f"color_by inconnu : {color_by!r}. Attendu : 'umatrix', 'density' ou 'class'."
        )

    grid = values.reshape(rows, cols)

    if figsize is None:
        figsize = (cols * 0.62 + 3.0, rows * 0.62 + 1.4)
    fig, ax = plt.subplots(figsize=figsize)

    if color_by == "class":
        image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
                          vmin=-0.5, vmax=len(tick_labels) - 0.5)
    else:
        image = ax.imshow(grid, cmap=cmap)

    if annotate:
        # Texte sombre sur fond clair et inversement, sinon les valeurs
        # disparaissent sur les cases extrêmes.
        rgba = image.cmap(image.norm(grid))
        for r in range(rows):
            for c in range(cols):
                v = grid[r, c]
                if np.isnan(v):
                    continue
                text = (tick_labels[int(v)] if color_by == "class"
                        else fmt.format(v))
                luma = (0.299 * rgba[r, c, 0] + 0.587 * rgba[r, c, 1]
                        + 0.114 * rgba[r, c, 2])
                ax.text(c, r, text, ha="center", va="center", fontsize=6.5,
                        color="black" if luma > 0.5 else "white")

    ax.set_xlabel("Colonne de la grille")
    ax.set_ylabel("Ligne de la grille")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(labelsize=7)

    cbar = fig.colorbar(image, ax=ax, label=cbar_label, fraction=0.046)
    if ticks is not None:
        cbar.set_ticks(list(ticks))
        cbar.ax.set_yticklabels(tick_labels)

    subtitle = f"grille {rows}x{cols} = {k} neurones (hexagonale)"
    if cluster_labels is not None:
        dead = int((np.bincount(cluster_labels, minlength=k) == 0).sum())
        subtitle += f" — {dead} neurone(s) mort(s)"
    ax.set_title(f"{title or default_title}\n{subtitle}", fontsize=11)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_prototype_map(weights, rows, cols, cluster_labels=None, y_true=None,
                       class_names=None, cell_size=0.62,
                       title="Carte de Kohonen — feature vectors", show=True):
    """Affiche les k feature vectors à leur place sur la grille.

    C'est la vue signature du SOM : chaque case montre ce que le neurone a
    appris, et les cases voisines se ressemblent — c'est exactement ce que la
    règle de voisinage impose. La même vue pour un K-means est une grille
    d'affichage arbitraire : ses centroïdes n'ont pas de voisins, on les y range
    par taille de cluster faute de mieux.

    L'affichage est une grille RECTANGULAIRE bien alignée : les voisinages
    hexagonaux valent pour l'entraînement, pas pour la lecture — comme les
    heatmaps, on ne décale pas les lignes impaires.

    Args:
        weights:        (k, 784) feature vectors, k = rows · cols.
        rows, cols:     forme de la grille.
        cluster_labels: (n,) neurone gagnant de chaque image ; grise les neurones
                        morts (jamais gagnants) et permet d'annoter les classes.
        y_true:         (n,) labels réels ; annote chaque case de sa classe
                        majoritaire (nécessite cluster_labels).
        class_names:    noms lisibles des classes ; indices si None.
        show:           True -> plt.show() (notebook) ; False -> retourne la figure.
    """
    rows, cols = int(rows), int(cols)
    W = np.asarray(weights, dtype=np.float32).reshape(-1, 28, 28)
    k = W.shape[0]
    if k != rows * cols:
        raise ValueError(
            f"{k} feature vectors pour une grille {rows}x{cols} = {rows * cols} cases."
        )

    sizes = None
    if cluster_labels is not None:
        cluster_labels = np.asarray(cluster_labels).astype(int).ravel()
        sizes = np.bincount(cluster_labels, minlength=k)

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
        n_classes = int(y_true.max() + 1)
        if class_names is not None:
            n_classes = max(n_classes, len(class_names))
        dominant = np.full(k, -1)
        for c in range(k):
            members = y_true[cluster_labels == c]
            if members.size:
                dominant[c] = np.bincount(members, minlength=n_classes).argmax()

    # Une seule paire d'axes, et chaque prototype posé à ses coordonnées via
    # `extent` : case (ligne, colonne) = position (colonne, ligne), grille
    # rectangulaire bien alignée.
    fig, ax = plt.subplots(figsize=(cols * cell_size + 2.2,
                                    (rows - 1) * cell_size + 1.4))

    # Vignettes CARRÉES, demi-côté < 0.5 : un fin liseré sépare les cases
    # voisines sans les faire se chevaucher.
    half = 0.46

    for i in range(k):
        x, y = float(i % cols), float(i // cols)
        dead = sizes is not None and sizes[i] == 0
        # extent = (gauche, droite, bas, haut) : « bas » vaut y + half car
        # l'axe est inversé plus bas, donc les grands y s'affichent en bas.
        ax.imshow(W[i], cmap="gray", vmin=0, vmax=1,
                  alpha=0.25 if dead else 1.0,
                  extent=(x - half, x + half, y + half, y - half),
                  zorder=2)
        if dominant is not None and not dead and dominant[i] >= 0:
            name = (class_names[dominant[i]] if class_names is not None
                    else str(dominant[i]))
            # Étiquette DANS le coin de la vignette : les lignes sont trop
            # resserrées pour loger du texte entre elles. Le fond des prototypes
            # étant noir, le cyan reste lisible.
            ax.text(x - half * 0.92, y + half * 0.92, str(name),
                    ha="left", va="bottom", fontsize=cell_size * 9,
                    color="cyan", zorder=3)

    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.set_aspect("equal")
    # Ligne 0 en haut, comme toutes les autres vues de la grille.
    ax.invert_yaxis()
    ax.axis("off")

    subtitle = f"grille {rows}x{cols} = {k} neurones (hexagonale)"
    if sizes is not None:
        subtitle += f" — {int((sizes == 0).sum())} neurone(s) mort(s)"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_class_distribution(cluster_labels, y_true, class_names=None, k=None,
                            clusters_per_chart=20, normalize=False,
                            title="Distribution des classes réelles par neurone",
                            figsize=None, show=True):
    """Bar chart groupé : de quelles classes réelles chaque neurone est composé.

    Chaque neurone reçoit un groupe de barres fines, une par classe réelle, côte
    à côte. Un groupe dominé par une seule barre = neurone pur ; plusieurs barres
    de hauteur voisine = neurone qui mélange des classes. Les barres étant
    juxtaposées et non empilées, leurs hauteurs se comparent directement.

    C'est la vue qui répond à « mes neurones retrouvent-ils mes vraies classes ? »,
    ce que l'inertie ne dit pas : elle ne mesure que la compression.

    k pouvant valoir des centaines, les neurones sont répartis sur plusieurs
    sous-graphiques : au-delà d'une vingtaine de groupes, les barres deviennent
    trop fines pour être lues.

    Args:
        cluster_labels:     (n,) neurone gagnant de chaque image.
        y_true:             (n,) classe réelle de chaque image.
        class_names:        noms lisibles des classes ; indices si None.
        k:                  nombre de neurones ; déduit des labels si None.
        clusters_per_chart: groupes de barres par sous-graphique.
        normalize:          True -> hauteurs en % du neurone (composition pure,
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

    # counts[classe, neurone] : effectif de chaque classe dans chaque neurone.
    counts = np.zeros((n_classes, k), dtype=np.int64)
    np.add.at(counts, (y_true, cluster_labels), 1)

    sizes = counts.sum(axis=0)
    # Pureté : part des images tombant dans un neurone où leur classe est
    # majoritaire. 1.0 = neurones parfaitement alignés sur les vraies classes.
    purity = counts.max(axis=0).sum() / counts.sum() if counts.sum() else 0.0

    heights = counts.astype(float)
    if normalize:
        # where= évite la division par zéro des neurones morts (0/0 -> 0).
        heights = np.divide(heights, np.where(sizes > 0, sizes, 1) * 1.0) * 100

    per_chart = max(1, int(clusters_per_chart))
    n_charts = int(np.ceil(k / per_chart))
    colors = plt.get_cmap("tab20" if n_classes > 10 else "tab10")

    # Barres juxtaposées : chaque neurone reçoit une case de largeur 1, partagée
    # entre ses n_classes barres. Le facteur 0.75 les affine et laisse respirer
    # les groupes voisins, pour qu'on voie où finit un neurone.
    group_width = 0.75
    bar_width = group_width / n_classes

    # La largeur doit suivre le nombre TOTAL de barres : 20 neurones x 10 classes
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
        ax.set_ylabel("Part du neurone (%)" if normalize else "Images")
        ax.set_title(f"Neurones {lo} à {hi - 1}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        # Séparateurs entre neurones : sans eux, les groupes de barres fines se
        # confondent visuellement avec leurs voisins.
        for boundary in idx[:-1]:
            ax.axvline(boundary + 0.5, color="0.85", linewidth=0.6, zorder=0)

        empty = int((sizes[lo:hi] == 0).sum())
        if empty:
            ax.text(0.998, 0.96, f"{empty} neurone(s) mort(s)", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="tab:red")

    axes[-1].set_xlabel("Neurone (numéro = code du codec)")

    # Les axes s'arrêtent sous le bandeau ; titre et légende l'occupent. Tout est
    # positionné en pouces convertis en fraction, pour que l'écart reste constant
    # quel que soit le nombre de sous-graphiques.
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig_height))

    fig.suptitle(
        f"{title}\n{k} neurones · pureté globale : {purity * 100:.1f} %",
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


def plot_inertia(history, n_samples=None, image_dim=784,
                 title="Courbe d'entraînement — inertie par époque",
                 ylabel="Inertie (somme des distances²)", figsize=(8, 4.5),
                 show=True):
    """Trace l'inertie au fil des époques : la courbe d'entraînement du SOM.

    Attention à la différence avec le K-means : là-bas l'inertie EST la loss, et
    ne peut que décroître — une remontée trahit un bug. Ici, le SOM ne minimise
    pas l'inertie mais un compromis entre elle et la contrainte de voisinage. La
    courbe peut donc osciller un peu, surtout à alpha élevé, sans que rien ne
    soit cassé. C'est la tendance générale qui compte, pas la monotonie.

    Args:
        history:   inerties de fit_kohonen(..., return_history=True) : l'état
                   initial en premier, puis une valeur par époque. L'axe des x
                   démarre donc à 0 = avant entraînement.
        n_samples: nombre d'images d'entraînement. Fourni, un second axe à droite
                   gradue la même courbe en MSE de reconstruction.

                   MSE et inertie ne diffèrent que d'un facteur (n · d) — voir
                   utils/metrics.py : la reconstruction d'une image étant le
                   feature vector de son neurone, l'inertie mesure déjà l'erreur
                   de reconstruction. D'où un second axe plutôt qu'une seconde
                   courbe, qui serait rigoureusement superposée à la première.
        image_dim: pixels par image (784 = 28x28).
        show:      True -> plt.show() (notebook) ; False -> retourne la figure.
    """
    history = np.asarray(history, dtype=np.float64).ravel()
    n = history.size
    if n == 0:
        raise ValueError("history est vide : rien à tracer.")

    # history[0] est l'état initial : l'axe démarre à 0, et il y a n - 1 époques
    # pour n points.
    epochs = np.arange(n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(epochs, history, marker="o", markersize=3, linewidth=1.5,
            color="tab:blue")

    start, end = history[0], history[-1]
    ax.annotate(f"avant entraînement : {start:,.0f}".replace(",", " "),
                xy=(0, start), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.annotate(f"final : {end:,.0f}".replace(",", " "),
                xy=(n - 1, end), xytext=(-6, 10), textcoords="offset points",
                fontsize=9, ha="right", color="tab:green")
    ax.scatter([n - 1], [end], color="tab:green", zorder=5, s=40)

    drop = (1 - end / start) * 100 if start else 0.0
    subtitle = f"{n - 1} époque(s) — inertie réduite de {drop:.1f} %"

    ax.set_xlabel("Époque (0 = avant entraînement)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if n > 1:
        ax.set_xlim(-0.5, n - 0.5)

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

    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    fig.tight_layout()
    if show:
        plt.show()
    return fig
