"""App Gradio — explorer les algos de compression du projet comme des codecs.

Lancement, depuis src/ :
    python app.py

Deux sélecteurs commandent tout : l'**algo** (K-means, Kohonen…) et le
**dataset** (MNIST, Quick, Draw!). Tous les onglets suivent.

--- Cette app ne connaît aucun algo ---

Elle vivait dans src/kmeas/ et ne savait faire que du K-means. Elle ne parle
désormais qu'à l'interface `Algo` (src/algo_base.py) : entraîner, encoder,
décoder, tracer. Elle ignore ce qu'est un centroïde ou un neurone.

Conséquence : ajouter un algo ne demande pas de rouvrir ce fichier. On écrit une
sous-classe d'Algo, on l'ajoute à ALGOS, et l'app se reconfigure — y compris ses
champs d'hyperparamètres, construits à partir de `algo.params`.

--- Les modèles ne se mélangent jamais ---

Chaque algo a son propre registre (src/<algo>/models/), et les modèles portent
leur dataset dans leurs métadonnées. Le sélecteur ne propose donc que les modèles
de l'algo ET du dataset courants. Sans ce filtre, un K-means MNIST et un K-means
Quick, Draw! — tous deux des tableaux (k, 784) — se croiseraient silencieusement
en produisant n'importe quoi.
"""

import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib

# Backend non-interactif : le serveur rend les figures en PNG sans jamais ouvrir
# de fenêtre. À poser avant d'importer pyplot, sinon matplotlib choisit un
# backend GUI et plante hors du thread principal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import gradio as gr

# src/ doit être importable quel que soit le dossier de lancement : c'est là que
# vivent data_import, algos/, kmeas/ et kohonen/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from algo_base import history_of  # noqa: E402
from data_import import DATASETS, load_dataset  # noqa: E402
from autoencoder.algo import Autoencoder  # noqa: E402
from gan.algo import GAN  # noqa: E402
from kmeas.algo import KMeans  # noqa: E402
from kohonen.algo import Kohonen  # noqa: E402
from pca.algo import PCAAlgo  # noqa: E402
from utils.model_registry import validate_model_name  # noqa: E402
from vae.algo import VAE  # noqa: E402

# Le catalogue des algos — la SEULE chose que ce fichier sait d'eux. Chaque algo
# vit chez lui (src/kmeas/algo.py, src/kohonen/algo.py) et présente l'interface
# Algo (algo_base.py) ; ajouter un algo se limite donc à cette ligne.
# L'ordre fixe celui du sélecteur ; la première entrée est l'algo par défaut.
ALGOS = {algo.key: algo for algo in (KMeans(), Kohonen(), PCAAlgo(), Autoencoder(), VAE(), GAN())}
DEFAULT_ALGO = next(iter(ALGOS))

IMAGE_DIM = 784
DEFAULT_DATASET = "mnist"

# Plafond des figures de compression : au-delà, les vignettes deviennent illisibles.
# Rien à voir avec le coût de calcul — c'est la largeur de la figure qui borne.
MAX_CODEC_IMAGES = 30

# Réglages par ligne dans l'onglet Entraînement. Le SOM en a 7 : sur une seule
# ligne, chacun se réduit à une bande trop étroite pour lire son libellé.
PARAMS_PER_ROW = 4

# Emplacements de figures dans l'onglet « Vues de l'algo ». K-means en montre 1,
# le SOM 2, la PCA jusqu'à 9 (spectre, heatmaps, centrage, génération…) : on crée
# le maximum d'emplacements et run_extra masque ceux qui restent vides.
MAX_EXTRA_FIGURES = 10

NO_MODEL = "⚠️ Sélectionne ou entraîne un modèle d'abord."

# Les datasets restent en mémoire une fois chargés : MNIST coûte ~10 s (import
# TensorFlow), et on ne modifie jamais ces tableaux.
_LOADED = {}


def get_dataset(key, progress=None):
    """Charge un dataset (ou le ressort du cache). Entrée unique de l'app."""
    if key not in _LOADED:
        _LOADED[key] = load_dataset(key, progress=progress)
    return _LOADED[key]


def split_of(ds, split):
    """(X, y) du split demandé — évite de répéter le ternaire partout."""
    return (ds.X_train, ds.y_train) if split == "Train" else (ds.X_test, ds.y_test)


def label_name(ds, y):
    """Nom lisible d'un label : « 7 » pour MNIST, « cat » pour Quick, Draw!."""
    return ds.class_names[int(y)]


def y_label_of(ds_key):
    """MNIST étiquette des chiffres, Quick, Draw! des objets : le libellé suit."""
    return "Chiffre réel" if ds_key == "mnist" else "Classe réelle"


# ------------------------------------------------------------------ Helpers


def format_bytes(n_bytes):
    if n_bytes >= 1024 ** 3:
        return f"{n_bytes / 1024 ** 3:.1f} Go"
    return f"{n_bytes / 1024 ** 2:.0f} Mo"


def memory_note(n, k):
    """Estime la RAM du calcul de distances — le poste dominant de l'entraînement.

    Les deux algos passent par l'identité ||x-c||² et n'allouent plus que (n, k) :
    le produit matriciel somme les 784 pixels au passage. Le coût est donc devenu
    modeste, mais reste le produit n×k — et ni n ni k ne sont plafonnés dans l'UI,
    d'où ce garde-fou.
    """
    n, k = int(n or 0), int(k or 0)
    cost = n * k * 4
    msg = (
        f"Matrice de distances `({n}, {k})` : **~{format_bytes(cost)}** de RAM "
        f"(+ {format_bytes(n * IMAGE_DIM * 4)} pour les images elles-mêmes)."
    )

    if cost > 4 * 1024 ** 3:
        return f"🔴 {msg}\n\nn×k est énorme : risque de saturer la RAM. Baisse k ou les images."
    if cost > 1024 ** 3:
        return f"🟠 {msg}\n\nAu-delà du Go : surveille ta RAM."
    if cost > 256 * 1024 ** 2:
        return f"🟡 {msg}"
    return f"⚪ {msg}"


def released(fig):
    """Retire la figure du gestionnaire global de pyplot avant de la rendre.

    pyplot garde une référence sur chaque figure créée : sans ça, aucune n'est
    jamais collectée et la RAM grimpe à chaque interaction. La figure reste
    parfaitement affichable par Gradio après fermeture.
    """
    plt.close(fig)
    return fig


# ------------------------------------------------- Hyperparamètres dynamiques

# L'app ne code en dur aucun hyperparamètre : elle construit un widget par
# `Param` déclaré par chaque algo, et n'affiche que le groupe de l'algo actif.
#
# Gradio veut une liste d'entrées FIXE à la construction. On empile donc les
# widgets de tous les algos, et PARAM_SLOTS retient à qui appartient chacun :
# les handlers reçoivent tout et ne gardent que ce qui concerne l'algo courant.
PARAM_SLOTS = []          # [(algo_key, param_name), ...] — même ordre que les widgets


def build_param_widget(param):
    """Le widget d'un Param.

    `step` n'est pas cosmétique. Les valeurs légales d'un gr.Number sont
    `minimum + n · step`, et son step vaut 1 PAR DÉFAUT : un champ flottant qui
    ne le précise pas n'accepte donc que des valeurs espacées de 1. Alpha (0 à 1)
    n'y avait plus que deux valeurs possibles, et son propre défaut n'en faisait
    pas partie — le champ était inéditable. Param.__post_init__ verrouille
    maintenant cette cohérence au démarrage.
    """
    if param.kind == "choice":
        return gr.Dropdown(
            choices=param.choices, value=param.default,
            label=param.label, info=param.info,
        )

    common = dict(
        value=param.default, label=param.label, info=param.info,
        minimum=param.minimum, maximum=param.maximum,
    )
    if param.kind == "float":
        return gr.Number(step=float(param.step or 0.01), **common)
    return gr.Number(precision=0, step=int(param.step or 1), **common)


def collect_params(algo_key, values, n_samples=None):
    """Reconstruit le dict d'hyperparamètres de l'algo actif à partir de tout le lot."""
    params = {
        name: value
        for (key, name), value in zip(PARAM_SLOTS, values)
        if key == algo_key
    }
    if n_samples is not None:
        params["n_samples"] = int(n_samples or 0)
    return params


# ------------------------------------------------------------------ Modèles


# Lire les métadonnées d'un modèle coûte la décompression du .npz entier — poids
# compris. models_for() les relit pour TOUT le registre à chaque changement d'algo
# ou de dataset, et chaque vue vérifie son modèle avant de tracer : sans cache, on
# décompresserait des dizaines de Mo pour lire deux clés. Les registres ne bougent
# que par cette app, qui invalide ce qu'elle écrit ; un rechargement de page repart
# de toute façon à zéro.
_META_CACHE = {}          # (algo_key, nom) -> meta


def model_meta(algo_key, name):
    """Métadonnées du modèle, ou None s'il n'existe pas / est illisible."""
    cached = _META_CACHE.get((algo_key, name))
    if cached is not None:
        return cached
    try:
        _, meta = ALGOS[algo_key].load(name)
    except Exception:
        return None  # absent, .npz corrompu ou d'un format plus ancien
    _META_CACHE[(algo_key, name)] = meta
    return meta


def models_for(algo_key, ds_key):
    """Modèles de cet algo entraînés sur ce dataset.

    Le registre d'un algo ne contient que ses modèles : reste à filtrer sur le
    dataset, que les métadonnées portent.
    """
    out = []
    for name in ALGOS[algo_key].list_models():
        meta = model_meta(algo_key, name)
        # Les modèles d'avant ces clés n'en ont aucune : ils ont tous été
        # entraînés sur MNIST, d'où cette valeur par défaut.
        if meta is not None and meta.get("dataset", "mnist") == ds_key:
            out.append(name)
    return out


def active_model(algo_key, ds_key, name):
    """Le nom demandé s'il désigne vraiment un modèle de cet algo ET ce dataset.

    Le navigateur envoie la valeur que le sélecteur avait au moment du clic. Un
    changement d'algo repeuple ce sélecteur, mais les vues déjà parties gardent
    l'ancien nom — un K-means demandé au registre Kohonen. Le rejeter donne une
    vue vide le temps que le sélecteur se cale, plutôt qu'une exception.
    """
    return name if name and name in models_for(algo_key, ds_key) else None


def model_choices(algo_key, ds_key, selected=None):
    """(update du sélecteur, nom retenu). Le libellé porte l'algo : deux updates
    séparés sur le même composant s'écraseraient l'un l'autre."""
    models = models_for(algo_key, ds_key)
    if selected is None or selected not in models:
        selected = models[0] if models else None
    update = gr.update(
        choices=models, value=selected,
        label=f"Modèle {ALGOS[algo_key].label} (filtré par dataset)",
    )
    return update, selected


def load_active(algo_key, name):
    """(algo, weights, meta) du modèle sélectionné."""
    algo = ALGOS[algo_key]
    weights, meta = algo.load(name)
    return algo, np.asarray(weights, dtype=np.float32), meta


def describe_model(algo_key, name):
    algo = ALGOS[algo_key]
    meta = model_meta(algo_key, name) if name else None
    if meta is None:
        return (
            "ℹ️ Aucun modèle pour cet algo et ce dataset. Entraînes-en un dans "
            "l'onglet **Entraînement**."
        )
    rows = "\n".join(f"| **{k}** | {v} |" for k, v in algo.describe_rows(meta))
    return (
        f"### `{name}`\n"
        f"| | |\n|---|---|\n"
        f"| **Algo** | {algo.label} |\n"
        f"| **Dataset** | {meta.get('dataset', 'mnist')} |\n"
        f"{rows}\n"
    )


def model_rows(algo_key, name):
    """Métadonnées en lignes (libellé, valeur), pour le tableau exporté."""
    algo, _, meta = load_active(algo_key, name)
    rows = [("Algo", algo.label), ("Dataset", meta.get("dataset", "mnist"))]
    rows += algo.describe_rows(meta)
    # La MSE plutôt que l'inertie brute : c'est la seule des deux qui se compare
    # d'un modèle à l'autre, l'inertie dépendant du nombre d'images.
    rows.append((
        "MSE de reconstruction",
        f"{meta['inertia'] / (meta['n_samples'] * IMAGE_DIM):.4f}",
    ))
    return rows


def plot_metadata_table(algo_key, name, figsize=None):
    """Rend le tableau de métadonnées en image, collable dans un rapport."""
    rows = model_rows(algo_key, name)
    height = 0.42 * len(rows) + 0.9
    fig, ax = plt.subplots(figsize=figsize or (6.4, height))
    ax.axis("off")

    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colWidths=[0.58, 0.42],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if col == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#f5f5f5")

    ax.set_title(name, fontsize=12, fontweight="bold", pad=14, family="monospace")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------- Événements


def max_train_images(ds_key):
    """Nombre max d'images d'entraînement du dataset actif."""
    return len(get_dataset(ds_key).X_train)


def refresh_models(algo_key, ds_key, name):
    """Relit le registre depuis le disque et repeuple le sélecteur.

    Gradio fige les `choices` d'un Dropdown dans la config envoyée au navigateur au
    DÉMARRAGE du serveur. Un gr.update() les corrige dans la page ouverte, mais un
    F5 recharge cette config d'origine : les modèles entraînés depuis semblent
    perdus. Ils sont bien sur le disque — c'est la liste qui datait. D'où ce
    rappel à chaque chargement de page, seul moment où le cache peut aussi se
    vider sans risque.
    """
    _META_CACHE.clear()
    update, selected = model_choices(algo_key, ds_key, name)
    return update, describe_model(algo_key, selected)


def cleared_views():
    """Les sorties des onglets, vidées. Un changement d'algo ou de dataset rend
    les figures affichées caduques : mieux vaut du vide qu'un faux repère.

    Déduit des composants plutôt qu'écrit à la main : un Plot se vide avec None,
    un Markdown avec "", et une liste de 15 littéraux tenue en parallèle de
    VIEW_OUTPUTS finit par se décaler. Quand ça arrive, Gradio rejette la réponse
    ENTIÈRE — y compris la mise à jour du sélecteur de modèle, qui semble alors ne
    plus filtrer par algo. VIEW_OUTPUTS n'existe qu'après la construction de l'UI,
    mais cette fonction n'est appelée qu'au moment d'une requête : c'est assez tôt.
    """
    return tuple("" if isinstance(c, gr.Markdown) else None for c in VIEW_OUTPUTS)


def switch_algo(algo_key, ds_key):
    """Change l'algo actif : bascule les hyperparamètres, refiltre les modèles."""
    algo = ALGOS[algo_key]
    update, selected = model_choices(algo_key, ds_key)
    return (
        # Un seul groupe d'hyperparamètres visible : celui de l'algo actif.
        *[gr.update(visible=(key == algo_key)) for key in ALGOS],
        update,
        describe_model(algo_key, selected),
        f"## {algo.title}",
        f"## {algo.dict_label} — le dictionnaire du codec",
        f"### Vues propres à {algo.label}",
        *cleared_views(),
    )


def switch_dataset(algo_key, ds_key, n_cur, nviz_cur, export_nviz_cur, progress=gr.Progress()):
    """Change le dataset actif : recharge, refiltre les modèles, réajuste l'UI."""
    ds = get_dataset(ds_key, progress=progress)
    n_train, n_test = len(ds.X_train), len(ds.X_test)

    update, selected = model_choices(algo_key, ds_key)
    banner = (
        f"**{DATASETS[ds_key]}**  \n{n_train} images d'entraînement · {n_test} de test · "
        f"{len(ds.class_names)} classes : {', '.join(ds.class_names)}"
    )

    # Relever le plafond ne suffit pas : 60 000 saisi sur MNIST resterait affiché
    # sur Quick, Draw! (25 715 max) et ne planterait qu'au clic sur Entraîner.
    # On ramène donc aussi les valeurs dans les nouvelles bornes.
    clamp = lambda v, hi: min(int(v or 1), hi)

    return (
        banner,
        update,
        describe_model(algo_key, selected),
        gr.update(maximum=n_train, value=clamp(n_cur, n_train),
                  label=f"Images d'entraînement (max {n_train})"),
        gr.update(maximum=n_train, value=clamp(nviz_cur, n_train),
                  label=f"Images à projeter (max {n_train})"),
        gr.update(maximum=n_train, value=clamp(export_nviz_cur, n_train),
                  label=f"Images à projeter (max {n_train})"),
        *cleared_views(),
    )


def sync_name(algo_key, ds_key, n_samples, current, previous_auto, *values):
    """Le nom suit algo/dataset/hyperparamètres, mais n'écrase jamais une saisie
    manuelle."""
    params = collect_params(algo_key, values, n_samples)
    auto = f"{algo_key}_{ALGOS[algo_key].auto_name(ds_key, params)}"
    keep_current = current and current.strip() and current != previous_auto
    return (current if keep_current else auto), auto


def sync_memory(algo_key, n_samples, *values):
    params = collect_params(algo_key, values, n_samples)
    return memory_note(n_samples, ALGOS[algo_key].k_of_params(params))


def train(algo_key, ds_key, n_samples, name, *values):
    name = (name or "").strip()
    try:
        validate_model_name(name)
    except ValueError as err:
        raise gr.Error(str(err))
    if not name:
        raise gr.Error("Donne un nom au modèle avant d'entraîner.")

    algo = ALGOS[algo_key]
    ds = get_dataset(ds_key)
    n_samples = int(n_samples or 0)

    if not 1 <= n_samples <= len(ds.X_train):
        raise gr.Error(
            f"Images d'entraînement : entre 1 et {len(ds.X_train)} "
            f"(taille du split train de {ds_key})."
        )

    params = collect_params(algo_key, values, n_samples)
    # Chaque algo valide ses propres réglages : l'app n'a pas à savoir que K
    # ne peut pas dépasser n, ni que gamma doit être positif.
    try:
        algo.check(params, n_samples)
    except ValueError as err:
        raise gr.Error(str(err))

    X_fit = ds.X_train[:n_samples]
    weights, meta = algo.train(X_fit, params)
    meta.update({"algo": algo_key, "dataset": ds_key, "n_samples": n_samples})
    try:
        algo.save(name, weights, meta)
    except (OSError, ValueError) as err:
        raise gr.Error(f"Impossible de sauvegarder le modèle : {err}")
    # Réentraîner sous un nom existant écrase le .npz : le cache doit suivre.
    _META_CACHE.pop((algo_key, name), None)

    history = history_of(meta)
    status = (
        f"✅ **{name}** ({algo.label}) entraîné sur {ds_key} et sauvegardé — inertie "
        f"finale : {meta['inertia']:.1f}"
        + (f" après {len(history)} point(s) de courbe." if history else ".")
        + " Il est sélectionné ci-dessus."
    )
    update, selected = model_choices(algo_key, ds_key, name)
    return update, describe_model(algo_key, selected), status


def remove_model(algo_key, ds_key, name):
    if not active_model(algo_key, ds_key, name):
        raise gr.Error("Aucun modèle sélectionné.")
    ALGOS[algo_key].delete(name)
    _META_CACHE.pop((algo_key, name), None)
    update, selected = model_choices(algo_key, ds_key)
    return update, describe_model(algo_key, selected), f"🗑️ Modèle **{name}** supprimé."


def reroll_seed():
    """Nouvelle graine de tirage, pour re-piocher d'autres images."""
    return int(np.random.default_rng().integers(0, 1_000_000))


def run_codec(algo_key, ds_key, name, n_images, seed):
    """Encode/décode n images tirées au hasard, une figure par split."""
    name = active_model(algo_key, ds_key, name)
    if not name:
        return None, None, NO_MODEL

    ds = get_dataset(ds_key)
    algo, weights, meta = load_active(algo_key, name)
    n_images = int(np.clip(int(n_images or 1), 1, MAX_CODEC_IMAGES))

    figures, stats = [], {}
    for split in ("Train", "Test"):
        X_split, y_split = split_of(ds, split)

        # Même graine pour les deux splits, mais tirage indépendant : on veut des
        # images variées, pas le même index des deux côtés.
        rng = np.random.default_rng(int(seed or 0) + (0 if split == "Train" else 1))
        idx = rng.choice(len(X_split), size=min(n_images, len(X_split)), replace=False)

        originals = X_split[idx]
        codes = [algo.encode(img, weights) for img in originals]
        reconstructions = [algo.decode(code, weights) for code in codes]

        stats[split] = len(set(codes))
        figures.append(released(algo.plot_reconstructions(
            originals, reconstructions,
            [label_name(ds, y) for y in y_split[idx]],
            [f"code {c}" for c in codes],
            f"{split.upper()} — {len(idx)} images tirées au hasard",
        )))

    # Le récit de la compression appartient à l'algo : K-means transmet UN
    # entier, la PCA k flottants — un texte unique ici mentirait pour l'un
    # des deux. Le défaut (algo_base) reprend l'ancien texte mot pour mot.
    info = algo.codec_note(meta, ds_key, n_images, int(seed or 0), stats)
    return figures[0], figures[1], info


def run_latent(algo_key, ds_key, name, n_viz, show_true):
    """Projette les deux splits d'un coup : nuage + distribution des classes."""
    name = active_model(algo_key, ds_key, name)
    if not name:
        return None, None, None, None, NO_MODEL

    ds = get_dataset(ds_key)
    algo, weights, meta = load_active(algo_key, name)

    n_viz = int(n_viz or 0)
    if n_viz < 1:
        raise gr.Error("Images à projeter : au moins 1.")
    if n_viz > len(ds.X_train):
        raise gr.Error(
            f"Images à projeter : au plus {len(ds.X_train)} (split train de {ds_key})."
        )

    y_label = y_label_of(ds_key)
    figures, distributions, counts = [], [], {}

    for split in ("Train", "Test"):
        X_all, y_all = split_of(ds, split)
        # Test est plus petit : on prend ce qui existe plutôt que de refuser.
        n = min(n_viz, len(X_all))
        counts[split] = n
        X_viz, y_viz = X_all[:n], y_all[:n]

        # None pour les codecs à code discret (K-means, SOM) : leur latent est
        # un entier, il n'y a rien à projeter — le nuage reste vide, seules les
        # distributions parlent. PCA et autoencodeur, eux, rendent une figure.
        fig = algo.plot_latent(
            X_viz, weights, meta,
            y_viz if show_true else None,
            ds.class_names, y_label,
            f"{split.upper()} — {n} images",
        )
        figures.append(released(fig) if fig is not None else None)
        distributions.append(released(algo.plot_distribution(
            algo.assign(X_viz, weights), y_viz,
            ds.class_names, int(meta["k"]),
            f"{split.upper()} — distribution des classes réelles",
        )))

    info = (
        f"**{ds_key} — Train : {counts['Train']} images · Test : {counts['Test']} "
        f"images.** Le modèle, lui, a été entraîné sur {meta['n_samples']} images de "
        f"train.\n\n"
    )
    if counts["Test"] < counts["Train"]:
        info += f"ℹ️ Test plafonne à {len(ds.X_test)} images : c'est toute sa taille.\n\n"
    # Le paragraphe « qu'est-ce que cet espace latent » vient de l'algo : discret
    # pour K-means/SOM (le défaut d'algo_base), continu pour la PCA.
    info += algo.latent_note(meta) + "\n\n" + memory_note(n_viz, int(meta["k"]))
    return figures[0], figures[1], distributions[0], distributions[1], info


def run_dictionary(algo_key, ds_key, name, n_viz=2000):
    """Le dictionnaire appris — centroïdes, feature vectors… selon l'algo."""
    name = active_model(algo_key, ds_key, name)
    if not name:
        return None, NO_MODEL

    ds = get_dataset(ds_key)
    algo, weights, meta = load_active(algo_key, name)

    n = min(int(n_viz or 1), len(ds.X_train))
    labels = algo.assign(ds.X_train[:n], weights)
    fig, note = algo.plot_dictionary(weights, meta, labels, ds.y_train[:n], ds.class_names)
    return released(fig), note


def run_extra(algo_key, ds_key, name, n_viz=2000):
    """Les vues propres à l'algo — les emplacements vides restent masqués."""
    name = active_model(algo_key, ds_key, name)
    if not name:
        return (*[gr.update(visible=False)] * MAX_EXTRA_FIGURES, NO_MODEL)

    ds = get_dataset(ds_key)
    algo, weights, meta = load_active(algo_key, name)

    n = min(int(n_viz or 1), len(ds.X_train))
    X, y = ds.X_train[:n], ds.y_train[:n]
    labels = algo.assign(X, weights)
    # La variante « avec données » : les vues de la PCA (génération,
    # interpolation, centrage) ont besoin des images ; pour K-means et Kohonen
    # elle délègue simplement à extra_figures.
    figures = algo.extra_figures_with_data(X, weights, meta, labels, y, ds.class_names)

    slots = []
    for i in range(MAX_EXTRA_FIGURES):
        if i < len(figures):
            title, fig = figures[i]
            slots.append(gr.update(value=released(fig), label=title, visible=True))
        else:
            slots.append(gr.update(value=None, visible=False))

    if not figures:
        note = f"ℹ️ **{algo.label}** n'expose pas de vue supplémentaire."
    else:
        note = (
            f"Vues propres à **{algo.label}**, calculées sur {n} images de train : "
            + " · ".join(f"**{t}**" for t, _ in figures)
        )
    return (*slots, note)


def run_curve(algo_key, ds_key, name):
    """Courbe d'entraînement du modèle sélectionné."""
    name = active_model(algo_key, ds_key, name)
    if not name:
        return None, NO_MODEL

    algo, _, meta = load_active(algo_key, name)
    history = history_of(meta)
    if not history:
        # Les modèles entraînés avant que l'historique soit persisté n'en ont
        # pas : il est perdu, seule l'inertie finale a survécu.
        return None, (
            f"⚠️ **{name}** a été entraîné avant l'ajout des courbes : son historique "
            f"n'a pas été sauvegardé. Seule son inertie finale est connue "
            f"({meta['inertia']:.1f}). Réentraîne-le pour obtenir sa courbe."
        )

    fig, note = algo.plot_curve(history, meta["n_samples"])
    return released(fig), note


def export_all(algo_key, ds_key, name, n_viz, n_images, seed, progress=gr.Progress()):
    """Génère toutes les figures du modèle d'un coup, prêtes pour le rapport.

    Chaque figure part en PNG 150 dpi, plus une archive ZIP pour tout récupérer
    en un clic. Les figures sont écrites sur disque plutôt que rendues en base64 :
    c'est ce qui permet le téléchargement et le clic droit « copier l'image ».
    """
    if not active_model(algo_key, ds_key, name):
        raise gr.Error("Sélectionne ou entraîne un modèle d'abord.")

    ds = get_dataset(ds_key)
    algo, weights, meta = load_active(algo_key, name)

    n_viz = int(np.clip(int(n_viz or 1), 1, len(ds.X_train)))
    n_images = int(np.clip(int(n_images or 10), 1, MAX_CODEC_IMAGES))
    seed = int(seed or 0)

    out_dir = Path(tempfile.mkdtemp(prefix="rapport_"))
    saved, skipped = [], []

    def emit(filename, fig):
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    progress(0.05, desc="Tableau des métadonnées…")
    emit("01_metadata.png", plot_metadata_table(algo_key, name))

    progress(0.15, desc="Courbe d'entraînement…")
    history = history_of(meta)
    if history:
        fig, _ = algo.plot_curve(history, meta["n_samples"])
        emit("02_courbe_entrainement.png", fig)
    else:
        skipped.append(
            "**la courbe d'entraînement** — ce modèle est antérieur à sa sauvegarde, "
            "réentraîne-le pour l'obtenir"
        )

    progress(0.3, desc="Compression / décompression…")
    codec_train, codec_test, _ = run_codec(algo_key, ds_key, name, n_images, seed)
    emit("03_codec_train.png", codec_train)
    emit("04_codec_test.png", codec_test)

    progress(0.5, desc="Espaces latents et distributions…")
    lat_train, lat_test, dist_train, dist_test, _ = run_latent(
        algo_key, ds_key, name, n_viz, True
    )
    # Pas de projection pour les codecs à code discret : plot_latent rend None.
    if lat_train is not None:
        emit("05_latent_train.png", lat_train)
    if lat_test is not None:
        emit("06_latent_test.png", lat_test)
    if lat_train is None and lat_test is None:
        skipped.append(
            "**les nuages de l'espace latent** — le code de cet algo est un entier "
            "discret, il n'y a rien à projeter (voir les distributions)"
        )
    emit("07_distribution_train.png", dist_train)
    emit("08_distribution_test.png", dist_test)

    progress(0.75, desc=f"{algo.dict_label}…")
    dict_fig, _ = run_dictionary(algo_key, ds_key, name, n_viz)
    emit("09_dictionnaire.png", dict_fig)

    progress(0.85, desc=f"Vues propres à {algo.label}…")
    labels = algo.assign(ds.X_train[:n_viz], weights)
    extras = algo.extra_figures_with_data(
        ds.X_train[:n_viz], weights, meta, labels, ds.y_train[:n_viz], ds.class_names
    )
    for i, (title, fig) in enumerate(extras):
        emit(f"{10 + i}_{title.split('—')[0].strip().lower().replace(' ', '_')}.png", fig)

    progress(0.95, desc="Archive ZIP…")
    zip_path = out_dir / f"{name}_rapport.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in saved:
            archive.write(path, path.name)

    info = (
        f"### {len(saved)} figures générées pour `{name}` ({algo.label})\n\n"
        f"Télécharge le **ZIP** pour tout récupérer d'un coup, ou fais un clic droit → "
        f"« Copier l'image » sur une vignette de l'aperçu.\n\n"
        f"| Fichier | Contenu |\n|---|---|\n"
        f"| `01_metadata.png` | le tableau du modèle |\n"
        + ("| `02_courbe_entrainement.png` | la courbe d'entraînement |\n" if history else "")
        + f"| `03_codec_train.png` · `04_codec_test.png` | {n_images} images "
        f"compressées/décompressées par split (graine {seed}) |\n"
        f"| `05_latent_train.png` · `06_latent_test.png` | espaces latents, {n_viz} images |\n"
        f"| `07_distribution_train.png` · `08_distribution_test.png` | distribution des "
        f"classes réelles |\n"
        f"| `09_dictionnaire.png` | les {meta['k']} {algo.dict_label.lower()} |\n"
        + "".join(f"| `{10 + i}_….png` | {t} |\n" for i, (t, _) in enumerate(extras))
    )
    if ds_key != "mnist":
        info += (
            f"\nℹ️ Les annotations de classe sont des **indices** : "
            f"{', '.join(f'{i} = {c}' for i, c in enumerate(ds.class_names))}.\n"
        )
    if skipped:
        info += "\n⚠️ Non généré : " + " ; ".join(skipped) + ".\n"

    return [str(p) for p in saved], str(zip_path), info


# ---------------------------------------------------------------------- UI

with gr.Blocks(title="Codecs non supervisés") as demo:
    gr.Markdown("# Compression non supervisée — K-means, Kohonen…")

    _algo = ALGOS[DEFAULT_ALGO]
    _ds = get_dataset(DEFAULT_DATASET)
    _n_train, _n_test = len(_ds.X_train), len(_ds.X_test)
    _models = models_for(DEFAULT_ALGO, DEFAULT_DATASET)

    _default_params = {p.name: p.default for p in _algo.params}
    _default_params["n_samples"] = 1000
    DEFAULT_NAME = f"{DEFAULT_ALGO}_{_algo.auto_name(DEFAULT_DATASET, _default_params)}"

    # Doit refléter la valeur initiale de name_in : sync_name compare le nom
    # courant à ce témoin pour distinguer « auto » d'« édité à la main ». Les
    # désynchroniser ferait passer le nom par défaut pour une saisie utilisateur,
    # et il ne suivrait plus jamais l'algo et les hyperparamètres.
    auto_state = gr.State(DEFAULT_NAME)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Algo")
            algo_dd = gr.Dropdown(
                choices=[(a.label, key) for key, a in ALGOS.items()],
                value=DEFAULT_ALGO,
                label="Algo actif (vaut pour tous les onglets)",
                interactive=True,
            )

            gr.Markdown("### Dataset")
            dataset_dd = gr.Dropdown(
                choices=[(label, key) for key, label in DATASETS.items()],
                value=DEFAULT_DATASET,
                label="Dataset actif (vaut pour tous les onglets)",
                interactive=True,
            )
            dataset_md = gr.Markdown(
                f"**{DATASETS[DEFAULT_DATASET]}**  \n{_n_train} images d'entraînement · "
                f"{_n_test} de test · {len(_ds.class_names)} classes : "
                f"{', '.join(_ds.class_names)}"
            )

            gr.Markdown("### Modèle actif")
            model_dd = gr.Dropdown(
                choices=_models,
                value=_models[0] if _models else None,
                label=f"Modèle {_algo.label} (filtré par dataset)",
                interactive=True,
                # Changer d'algo repeuple ce sélecteur, mais les vues déjà parties
                # portent encore l'ancien nom — un K-means dans une liste Kohonen.
                # Sans ça, Gradio refuse cette valeur AVANT d'appeler le handler
                # (« … is not in the list of choices ») et l'onglet part en erreur
                # au lieu de se vider. active_model() la rejette proprement, lui.
                allow_custom_value=True,
            )
            delete_btn = gr.Button("Supprimer ce modèle", variant="stop", size="sm")
            model_md = gr.Markdown(
                describe_model(DEFAULT_ALGO, _models[0] if _models else None)
            )
            status_md = gr.Markdown()

        with gr.Column(scale=3):
            with gr.Tabs():
                # ----------------------------------------- Entraînement
                with gr.Tab("Entraînement"):
                    train_title_md = gr.Markdown(f"## {_algo.title}")

                    with gr.Row():
                        with gr.Column(min_width=170):
                            n_in = gr.Number(
                                value=1000, precision=0, minimum=1, maximum=_n_train,
                                label=f"Images d'entraînement (max {_n_train})",
                            )
                            max_train_btn = gr.Button("Tout le split train", size="sm")

                    # Un groupe d'hyperparamètres par algo, construit depuis
                    # `algo.params` : ce fichier n'en connaît aucun.
                    param_groups, param_widgets = {}, []
                    for key, algo in ALGOS.items():
                        with gr.Group(visible=(key == DEFAULT_ALGO)) as group:
                            gr.Markdown(f"**Hyperparamètres — {algo.label}**")
                            # Par lignes de PARAMS_PER_ROW : les 7 réglages du SOM
                            # tenaient sur une seule ligne, chacun réduit à une
                            # bande trop étroite pour lire son libellé.
                            for start in range(0, len(algo.params), PARAMS_PER_ROW):
                                with gr.Row():
                                    for param in algo.params[start:start + PARAMS_PER_ROW]:
                                        param_widgets.append(build_param_widget(param))
                                        PARAM_SLOTS.append((key, param.name))
                        param_groups[key] = group

                    mem_md = gr.Markdown(
                        memory_note(1000, _algo.k_of_params(_default_params))
                    )
                    name_in = gr.Textbox(value=DEFAULT_NAME, label="Nom du modèle")
                    train_btn = gr.Button("Entraîner", variant="primary")

                # ------------------------- Compression / Décompression
                with gr.Tab("Compression / Décompression") as tab_codec:
                    gr.Markdown("## Compression / Décompression")
                    gr.Markdown(
                        "Des images tirées **au hasard**, encodées puis décodées, sur les "
                        "**deux splits à la fois**. Originaux en haut de chaque figure, "
                        "reconstructions en dessous."
                    )

                    with gr.Row():
                        with gr.Column(min_width=170):
                            codec_n_in = gr.Number(
                                value=10, precision=0, minimum=1, maximum=MAX_CODEC_IMAGES,
                                label=f"Images par split (max {MAX_CODEC_IMAGES})",
                            )
                            codec_max_btn = gr.Button(f"Max ({MAX_CODEC_IMAGES})", size="sm")
                        codec_seed_in = gr.Number(value=0, precision=0, label="Graine du tirage")
                        codec_reroll_btn = gr.Button("Nouveau tirage aléatoire")

                    codec_plot_train = gr.Plot(label="TRAIN")
                    codec_plot_test = gr.Plot(label="TEST")
                    codec_md = gr.Markdown()

                # ----------------------------------------- Espace latent
                with gr.Tab("Espace latent") as tab_latent:
                    gr.Markdown("## Espace latent")
                    gr.Markdown(
                        "Les **deux splits traités en une fois**. Test étant plus petit, "
                        "il plafonne à sa taille. Les algos à latent **continu** (PCA, "
                        "autoencodeur) affichent leur projection 2D ; les codecs à code "
                        "**discret** (K-means, SOM) n'ont rien à projeter — leurs "
                        "distributions suffisent."
                    )

                    with gr.Row():
                        with gr.Column(min_width=170):
                            nviz_in = gr.Number(
                                value=1000, precision=0, minimum=1, maximum=_n_train,
                                label=f"Images à projeter (max {_n_train})",
                            )
                            max_viz_btn = gr.Button("Tout le split train", size="sm")
                        true_in = gr.Checkbox(value=True, label="Comparer aux classes réelles")

                    latent_btn = gr.Button("Projeter train et test", variant="primary")
                    latent_plot_train = gr.Plot(label="TRAIN")
                    latent_plot_test = gr.Plot(label="TEST")
                    gr.Markdown("### Distribution des classes réelles")
                    latent_dist_train = gr.Plot(label="TRAIN — distribution")
                    latent_dist_test = gr.Plot(label="TEST — distribution")
                    latent_md = gr.Markdown()

                # ------------------------------------------ Dictionnaire
                with gr.Tab("Dictionnaire") as tab_dict:
                    dict_title_md = gr.Markdown(
                        f"## {_algo.dict_label} — le dictionnaire du codec"
                    )
                    dict_plot = gr.Plot(label="Dictionnaire")
                    dict_md = gr.Markdown()

                # --------------------------------------- Vues de l'algo
                with gr.Tab("Vues de l'algo") as tab_extra:
                    extra_title_md = gr.Markdown(f"### Vues propres à {_algo.label}")
                    gr.Markdown(
                        "Ce que cet algo est seul à pouvoir montrer : l'U-matrix d'un SOM "
                        "n'a aucun sens pour un K-means, et réciproquement."
                    )
                    # Autant d'emplacements que l'algo le plus bavard (la PCA en
                    # remplit 9) : run_extra masque ceux qui restent vides.
                    extra_plots = [
                        gr.Plot(visible=False) for _ in range(MAX_EXTRA_FIGURES)
                    ]
                    extra_md = gr.Markdown()

                # ------------------------------------ Courbe d'entraînement
                with gr.Tab("Courbe d'entraînement") as tab_curve:
                    gr.Markdown("## Courbe d'entraînement")
                    curve_plot = gr.Plot(label="Inertie")
                    curve_md = gr.Markdown()

                # -------------------------------------------- Export rapport
                with gr.Tab("Export / Rapport"):
                    gr.Markdown("## Export pour le rapport")
                    gr.Markdown(
                        "Génère **toutes** les figures du modèle en une fois, y compris "
                        "les vues propres à l'algo."
                    )

                    with gr.Row():
                        with gr.Column(min_width=170):
                            export_n_images = gr.Number(
                                value=10, precision=0, minimum=1, maximum=MAX_CODEC_IMAGES,
                                label="Images par split (compression)",
                            )
                            export_max_images_btn = gr.Button(
                                f"Max ({MAX_CODEC_IMAGES})", size="sm"
                            )
                        with gr.Column(min_width=170):
                            export_seed = gr.Number(
                                value=0, precision=0, label="Graine du tirage"
                            )
                            export_reroll_btn = gr.Button("Nouveau tirage", size="sm")
                        with gr.Column(min_width=170):
                            export_nviz = gr.Number(
                                value=1000, precision=0, minimum=1, maximum=_n_train,
                                label=f"Images à projeter (max {_n_train})",
                            )
                            export_max_viz_btn = gr.Button("Tout le split train", size="sm")

                    export_btn = gr.Button("Générer toutes les figures", variant="primary")
                    export_zip = gr.File(label="Archive ZIP — toutes les images")
                    export_gallery = gr.Gallery(
                        label="Aperçu — clic droit sur une image pour la copier",
                        columns=3, height=420,
                    )
                    export_md = gr.Markdown()

    # ------------------------------------------------------------ Câblage

    VIEW_OUTPUTS = [
        codec_plot_train, codec_plot_test, codec_md,
        latent_plot_train, latent_plot_test, latent_dist_train, latent_dist_test, latent_md,
        dict_plot, dict_md,
        *extra_plots, extra_md,
        curve_plot, curve_md,
    ]

    # Les boutons « Tout le split train » remplissent le champ avec le maximum du
    # dataset actif — 60 000 sur MNIST, 25 715 sur Quick, Draw!.
    max_train_btn.click(max_train_images, dataset_dd, n_in)
    max_viz_btn.click(max_train_images, dataset_dd, nviz_in)
    export_max_viz_btn.click(max_train_images, dataset_dd, export_nviz)

    # Les plafonds constants n'ont pas besoin du serveur pour être connus.
    codec_max_btn.click(lambda: MAX_CODEC_IMAGES, None, codec_n_in)
    export_max_images_btn.click(lambda: MAX_CODEC_IMAGES, None, export_n_images)
    export_reroll_btn.click(reroll_seed, None, export_seed)

    # Repeuple le sélecteur à chaque ouverture ou rechargement de page : les
    # choices figées dans la config datent du démarrage du serveur.
    demo.load(refresh_models, [algo_dd, dataset_dd, model_dd], [model_dd, model_md])

    algo_dd.change(
        switch_algo,
        [algo_dd, dataset_dd],
        [*param_groups.values(), model_dd, model_md,
         train_title_md, dict_title_md, extra_title_md, *VIEW_OUTPUTS],
    )
    algo_dd.change(
        sync_name,
        [algo_dd, dataset_dd, n_in, name_in, auto_state, *param_widgets],
        [name_in, auto_state],
    )
    algo_dd.change(sync_memory, [algo_dd, n_in, *param_widgets], mem_md)

    dataset_dd.change(
        switch_dataset,
        [algo_dd, dataset_dd, n_in, nviz_in, export_nviz],
        [dataset_md, model_dd, model_md, n_in, nviz_in, export_nviz, *VIEW_OUTPUTS],
    )
    dataset_dd.change(
        sync_name,
        [algo_dd, dataset_dd, n_in, name_in, auto_state, *param_widgets],
        [name_in, auto_state],
    )

    for widget in (n_in, *param_widgets):
        widget.change(sync_memory, [algo_dd, n_in, *param_widgets], mem_md)
        widget.change(
            sync_name,
            [algo_dd, dataset_dd, n_in, name_in, auto_state, *param_widgets],
            [name_in, auto_state],
        )

    train_btn.click(
        train,
        [algo_dd, dataset_dd, n_in, name_in, *param_widgets],
        [model_dd, model_md, status_md],
    )
    delete_btn.click(
        remove_model, [algo_dd, dataset_dd, model_dd], [model_dd, model_md, status_md]
    )
    model_dd.change(describe_model, [algo_dd, model_dd], model_md)

    # On ne recalcule que la vue regardée, jamais les autres.
    codec_inputs = [algo_dd, dataset_dd, model_dd, codec_n_in, codec_seed_in]
    codec_outputs = [codec_plot_train, codec_plot_test, codec_md]
    tab_codec.select(run_codec, codec_inputs, codec_outputs)
    for widget in (codec_n_in, codec_seed_in, model_dd):
        widget.change(run_codec, codec_inputs, codec_outputs)
    # Le bouton ne fait que changer la graine : le .change ci-dessus redessine.
    codec_reroll_btn.click(reroll_seed, None, codec_seed_in)

    # La projection latente n'est pas plafonnée et traite les deux splits : elle
    # peut coûter des minutes sur 60 000 images. Elle ne part donc QUE sur clic
    # explicite — un rendu auto à l'ouverture figerait l'app.
    latent_btn.click(
        run_latent,
        [algo_dd, dataset_dd, model_dd, nviz_in, true_in],
        [latent_plot_train, latent_plot_test, latent_dist_train, latent_dist_test, latent_md],
    )

    dict_outputs = [dict_plot, dict_md]
    tab_dict.select(run_dictionary, [algo_dd, dataset_dd, model_dd], dict_outputs)
    model_dd.change(run_dictionary, [algo_dd, dataset_dd, model_dd], dict_outputs)

    extra_outputs = [*extra_plots, extra_md]
    tab_extra.select(run_extra, [algo_dd, dataset_dd, model_dd], extra_outputs)
    model_dd.change(run_extra, [algo_dd, dataset_dd, model_dd], extra_outputs)

    # La courbe est déjà calculée et stockée dans le .npz : la tracer ne coûte
    # rien, elle peut donc se rendre à l'ouverture de l'onglet.
    curve_outputs = [curve_plot, curve_md]
    tab_curve.select(run_curve, [algo_dd, dataset_dd, model_dd], curve_outputs)
    model_dd.change(run_curve, [algo_dd, dataset_dd, model_dd], curve_outputs)

    # L'export refait toutes les vues d'un coup : uniquement sur clic explicite.
    export_btn.click(
        export_all,
        [algo_dd, dataset_dd, model_dd, export_nviz, export_n_images, export_seed],
        [export_gallery, export_zip, export_md],
    )


if __name__ == "__main__":
    # Gradio 6 : le thème se passe à launch(), plus au constructeur de Blocks.
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
