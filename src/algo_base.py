"""L'interface que tout algo doit présenter à l'app.

--- Pourquoi cette couche existe ---

L'app ne doit rien savoir des algos hormis leur existence. Sans ça, chaque
nouvel algo obligerait à parsemer app.py de `if algo == "kmeans"` — c'est
exactement ce qu'on cherche à éviter en sortant l'app de src/kmeas/.

Chaque algo reste chez lui : `src/<algo>/algo.py` contient son adaptateur.
Ce fichier ne contient que le contrat commun aux six implémentations.

L'app ne manipule donc que des objets `Algo` : elle sait entraîner, encoder,
décoder et tracer sans jamais savoir ce qu'est un centroïde ou un neurone.

--- Ajouter un algo ---

    1. Écrire une sous-classe d'Algo dans src/<mon_algo>/algo.py.
    2. L'enregistrer dans ALGOS (src/app.py) — une ligne.

Rien d'autre. L'app découvre ses hyperparamètres via `params` et construit les
widgets toute seule.

--- Le contrat ---

Un algo expose un entraînement, un codec `encode/decode`, une assignation utile
aux vues et un registre de modèles. Son code peut être discret (K-means/SOM),
continu (PCA/AE/VAE) ou obtenu par inversion approximative (GAN).

Les poids voyagent toujours sous la forme d'un tableau NumPy, mais leur forme
est propre à l'algorithme : prototypes 2D, base PCA ou vecteur plat de réseaux.
Les métadonnées persistées décrivent l'architecture nécessaire au décodage.
"""

import math
from dataclasses import dataclass, field
from typing import Any

IMAGE_DIM = 784


def history_of(meta):
    """Courbe d'entraînement du modèle (liste vide si absente).

    Vit ici plutôt que dans un algo : l'app en a besoin et ne doit importer aucun
    algo en particulier.

    Une parenthèse du K-means a stocké un dict {"train": [...], "test": [...]} :
    on en ressort la série de train plutôt que de rejeter ces modèles.
    """
    raw = meta.get("inertia_history")
    if not raw:
        return []
    if isinstance(raw, dict):
        return raw.get("train", [])
    return raw


@dataclass
class Param:
    """Un hyperparamètre, décrit pour que l'app construise son widget seule.

    C'est ce qui permet à l'app d'ignorer les algos : elle lit cette description
    plutôt que de coder en dur « K-means a un K et une tolérance ».
    """

    name: str                       # clé dans le dict passé à train()
    label: str                      # libellé du widget
    kind: str = "int"               # "int" | "float" | "choice"
    default: Any = 0
    minimum: Any = None
    maximum: Any = None
    choices: Any = None             # [(libellé, valeur), ...] pour kind="choice"
    info: str = ""                  # aide sous le widget
    step: Any = None                # granularité ; 1 (int) ou 0.01 (float) si None

    def __post_init__(self):
        """Refuse un défaut que le widget lui-même rejetterait.

        Les valeurs légales d'un champ numérique sont `minimum + n · step` — et
        RIEN d'autre. Un défaut hors de cette grille produit un champ né invalide,
        que le navigateur refuse de modifier : le paramètre devient inatteignable
        sans le moindre message d'erreur. C'est exactement ce qui est arrivé à
        alpha (défaut 0,1 avec minimum=0,0001 et step=1 : légal = 0,0001 / 1,0001 /
        2,0001…). Le piège est silencieux, donc il se re-tendra : autant qu'il
        casse ici, au démarrage, plutôt que dans l'UI.
        """
        if self.kind not in ("int", "float", "choice"):
            raise ValueError(f"{self.name} : kind inconnu {self.kind!r}.")
        if self.kind == "choice":
            values = [v for _, v in (self.choices or [])]
            if self.default not in values:
                raise ValueError(
                    f"{self.name} : défaut {self.default!r} absent des choix {values}."
                )
            return

        step = self.step if self.step is not None else (1 if self.kind == "int" else 0.01)
        if step <= 0:
            raise ValueError(f"{self.name} : step doit être strictement positif.")

        base = self.minimum if self.minimum is not None else 0
        offsets = (self.default - base) / step
        if abs(offsets - round(offsets)) > 1e-6:
            raise ValueError(
                f"{self.name} : défaut {self.default} inatteignable — les valeurs "
                f"légales sont {base} + n × {step}. Aligne le défaut, le minimum "
                f"ou le step."
            )
        if self.maximum is not None and self.default > self.maximum:
            raise ValueError(f"{self.name} : défaut {self.default} > maximum {self.maximum}.")


class Algo:
    """Classe de base. Voir kmeans.py / kohonen.py pour des implémentations."""

    key = ""                        # identifiant technique, ex. "kmeans"
    label = ""                      # libellé affiché
    title = ""                      # titre de l'onglet d'entraînement

    # Vocabulaire de l'algo : l'app l'utilise pour ses libellés, plutôt que de
    # dire « centroïde » à un SOM ou « neurone » à un K-means.
    dict_label = "Prototypes"       # nom du dictionnaire
    code_label = "prototype"        # nom d'une entrée du dictionnaire
    k_label = "k — nombre de prototypes"

    params: list = field(default_factory=list)

    # ------------------------------------------------------------ Entraînement

    def train(self, X, p):
        """Entraîne sur X (n, 784) avec le dict d'hyperparamètres p.

        Retourne (weights (k, 784), meta) où meta contient au moins :
            "k"               : taille du dictionnaire
            "inertia"         : inertie finale
            "inertia_history" : l'inertie au fil de l'entraînement
        plus les hyperparamètres propres à l'algo.
        """
        raise NotImplementedError

    def k_of_params(self, p):
        """Taille du dictionnaire pour ces hyperparamètres, AVANT d'entraîner.

        Sert à l'estimation mémoire et aux garde-fous de l'UI.
        """
        raise NotImplementedError

    def check(self, p, n_samples):
        """Lève ValueError si ces hyperparamètres sont invalides. Message pour l'UI."""
        return None

    def auto_name(self, ds_key, p):
        """Nom de modèle proposé par défaut."""
        raise NotImplementedError

    # ------------------------------------------------------------------ Codec

    def assign(self, X, weights):
        """(n,) indice du prototype le plus proche de chaque image."""
        raise NotImplementedError

    def encode(self, image, weights):
        """Code d'une image — un int."""
        raise NotImplementedError

    def decode(self, code, weights):
        """Prototype de ce code — (784,) numpy."""
        raise NotImplementedError

    # --------------------------------------------------------------- Registre

    def list_models(self):
        raise NotImplementedError

    def load(self, name):
        """(weights, meta)."""
        raise NotImplementedError

    def save(self, name, weights, meta):
        raise NotImplementedError

    def delete(self, name):
        raise NotImplementedError

    # ------------------------------------------------------------------- Vues

    def plot_reconstructions(self, originals, reconstructions, top_labels,
                             bottom_labels, title):
        """Originaux en haut, reconstructions en bas.

        Cette vue ne regarde que des images : elle serait identique pour tous les
        algos. Elle passe quand même par l'interface, pour que chaque algo reste
        servi par SON package — sinon il faudrait un module de figures partagées
        à côté, et l'app devrait choisir laquelle appeler selon l'algo.
        """
        raise NotImplementedError

    def plot_distribution(self, codes, y_true, class_names, k, title):
        """Classes réelles par code. Générique elle aussi — voir ci-dessus."""
        raise NotImplementedError

    def plot_latent(self, X, weights, meta, y_true, class_names, y_label, title):
        """Projection 2D de l'espace latent, ou None s'il n'y en a pas.

        Le défaut est None : le latent d'un codec discret (K-means, SOM) est un
        entier, il n'y a rien à projeter — l'app laisse alors les nuages vides
        et ne montre que les distributions. Seuls les algos à latent continu
        (PCA, autoencodeur) fournissent une figure.
        """
        return None

    def plot_dictionary(self, weights, meta, labels, y_true, class_names):
        """Le dictionnaire appris, en une figure. Retourne (figure, note markdown)."""
        raise NotImplementedError

    def plot_curve(self, history, n_samples):
        """Courbe d'entraînement. Retourne (figure, note markdown)."""
        raise NotImplementedError

    def extra_figures(self, weights, meta, labels, y_true, class_names):
        """Vues propres à l'algo — [(titre, figure), ...].

        C'est l'échappatoire de l'interface : l'U-matrix n'a de sens que pour un
        SOM, la cartographie triée par taille que pour un K-means. Plutôt que de
        forcer tous les algos dans les mêmes vues, chacun ajoute les siennes.
        L'app expose un nombre fixe d'emplacements (MAX_EXTRA_FIGURES) et masque
        ceux qui restent vides.
        """
        return []

    def extra_figures_with_data(self, X, weights, meta, labels, y_true, class_names):
        """Comme extra_figures, mais avec les images en plus.

        Certains algos n'ont besoin que de leurs poids pour leurs vues propres
        (U-matrix, cartographie) — d'autres ont besoin des DONNÉES : la PCA
        refait des décompositions sur X (comparaison des centrages), ajuste des
        gaussiennes latentes (génération) ou interpole entre deux images.

        Le défaut délègue à extra_figures : K-means et Kohonen n'ont rien à
        changer, seuls les algos gourmands en données surchargent celle-ci.
        """
        return self.extra_figures(weights, meta, labels, y_true, class_names)

    # ------------------------------------------------------------------ Récits

    # Les textes sous les figures dépendent du CODEC, pas seulement des nombres :
    # K-means transmet UN entier, la PCA k flottants — un texte unique dans l'app
    # mentirait forcément pour l'un des deux. Les défauts ci-dessous racontent le
    # codec à code discret (K-means, SOM) ; la PCA les remplace entièrement.

    def codec_note(self, meta, ds_key, n_images, seed, stats):
        """Texte de l'onglet Compression / Décompression.

        stats : {"Train": codes distincts, "Test": codes distincts} sur le tirage.
        """
        k = int(meta["k"])
        bits = math.ceil(math.log2(k)) if k > 1 else 0
        ratio = float("inf") if bits == 0 else IMAGE_DIM * 8 / bits
        mse = meta["inertia"] / (meta["n_samples"] * IMAGE_DIM)
        return (
            f"**{n_images} images tirées au hasard** dans chaque split de **{ds_key}**, "
            f"encodées par **{self.label}** (graine {seed} — relance le tirage "
            f"pour en voir d'autres).\n\n"
            f"Chaque image (784 pixels) est transmise sous la forme d'**un seul entier**, "
            f"soit {bits} bits pour k={k} — un taux de compression de **{ratio:.0f}:1** "
            f"face aux {IMAGE_DIM * 8} bits de l'image brute (le dictionnaire des "
            f"{self.dict_label.lower()} n'étant transmis qu'une fois).\n\n"
            f"Le décodeur renvoie le {self.code_label} portant ce numéro : deux images "
            f"partageant un code se reconstruisent **à l'identique** — visible dès que deux "
            f"colonnes ont le même code.\n\n"
            f"Le prix de cette compression est la **MSE : {mse:.4f}** par pixel (mesurée sur "
            f"le train à l'entraînement). C'est l'écart entre la ligne du haut et celle du "
            f"bas.\n\n"
            f"Codes distincts sur ce tirage : {stats['Train']}/{n_images} en train, "
            f"{stats['Test']}/{n_images} en test."
        )

    def latent_note(self, meta):
        """Paragraphe « qu'est-ce que cet espace latent » de l'onglet Espace latent."""
        return (
            f"L'espace latent de ce codec est **un entier discret** dans "
            f"{{0, …, {int(meta['k']) - 1}}} : il n'y a rien à projeter en 2D, ce sont "
            f"donc les **distributions** qui le décrivent.\n\n"
            "**Distribution** : un groupe de barres par code, une barre par classe réelle. "
            "Un groupe dominé par une seule barre = code pur ; plusieurs barres de hauteur "
            "voisine = code qui mélange des classes. La pureté indiquée est la part des "
            "images tombant dans un groupe où leur classe est majoritaire — c'est ce que "
            "l'inertie ne mesure **pas**."
        )

    def describe_rows(self, meta):
        """Métadonnées du modèle en lignes (libellé, valeur)."""
        raise NotImplementedError
