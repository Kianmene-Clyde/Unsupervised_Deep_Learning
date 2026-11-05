"""Construction, entraînement et (dé)sérialisation des GAN — en NumPy pur.

--- Pourquoi NumPy ici, quand l'autoencodeur et le VAE parlent Keras ---

Le GAN est le dernier algo du projet : c'est l'occasion de montrer qu'on sait
écrire soi-même TOUT ce que Keras cachait — la passe avant, la rétro-propagation
couche par couche, et Adam. Deux MLP se font face :

    Générateur    G : z (latent_dim) → couches denses LeakyReLU → 784, sigmoïde
    Discriminateur D : 784 → couches denses LeakyReLU → 1 logit (pas de sigmoïde)

et s'entraînent en alternance sur le jeu minimax de Goodfellow (2014) :

    D maximise   log D(x) + log(1 − D(G(z)))     (distinguer vrai / généré)
    G maximise   log D(G(z))                     (variante « non saturante » :
                                                  le -log(1-D) du papier s'écrase
                                                  quand D gagne, celle-ci non)

Trois stabilisateurs classiques, chacun justifié :
- D produit un LOGIT et la BCE se calcule dessus (softplus) : composer
  sigmoïde puis log exploserait numériquement quand D devient sûr de lui ;
  le gradient de la BCE par rapport au logit est alors σ(logit) − étiquette,
  d'une propreté qui justifie à elle seule ce choix.
- lissage d'étiquettes : les vraies images valent 0,9 et non 1,0, pour que D
  ne devienne pas trop confiant et continue d'envoyer du gradient à G.
- Adam avec β₁ = 0,5 (au lieu de 0,9) : la recommandation DCGAN — le momentum
  fort oscille sur un jeu à deux joueurs dont l'objectif bouge sans cesse.

--- Les poids voyagent comme pour l'autoencodeur ---

Un vecteur float32 unique (tous les tenseurs de G puis de D, aplatis),
l'architecture dans les métadonnées. Même contrat que utils/model.py de
l'autoencodeur : l'app fait circuler des tableaux numpy, rien d'autre.

--- L'« encodeur » qui n'existe pas : l'inversion par banque ---

Un GAN n'a PAS d'encodeur : rien ne va de l'image vers z. Pour que le GAN
reste un citoyen complet de l'app (onglets compression et espace latent), on
approxime l'inversion : une banque de BANK_SIZE couples (z, G(z)) est générée
déterministe depuis la graine du modèle, et encoder une image = prendre le z
de l'image générée la plus proche (plus proche voisin en distance L2). C'est
une vraie technique (« GAN inversion »), assumée comme grossière : la MSE
élevée qui en sort est PRÉCISÉMENT la démonstration que le GAN n'est pas un
codec — voir la discussion des 3 tâches dans algo.py.
"""

import numpy as np

from utils.array_cache import ArrayFingerprints
from utils.neural_config import positive_integer, validate_training_config

IMG_SIZE = 28
IMAGE_DIM = IMG_SIZE * IMG_SIZE
LEAKY_RELU_SLOPE = 0.2

# Taille de la banque d'inversion (couples (z, G(z)) — voir docstring).
BANK_SIZE = 2048

# vecteur.tobytes() -> cfg : de quoi redécouper le vecteur plat (léger).
_KNOWN = {}
_KNOWN_MAX = 64
# empreinte -> paramètres G/D déjà découpés (évite de recopier les poids à chaque génération)
_PARAMS = {}
_PARAMS_MAX = 4
# vecteur.tobytes() -> (Zb, Xb) : la banque d'inversion, régénérable mais coûteuse.
_BANKS = {}
_BANKS_MAX = 4
_FINGERPRINTS = ArrayFingerprints()


# ----------------------------------------------------------- Architecture


def layer_dims(cfg):
    """Les dimensions de chaque couche de G et de D, depuis la config.

    G : latent_dim → g_hidden… → 784      D : 784 → d_hidden… → 1
    C'est la SEULE source de vérité des shapes : params_from_flat et
    init_params s'appuient dessus, donc le redécoupage est déterministe.
    """
    positive_integer(cfg["latent_dim"], "latent_dim")
    for name in ("g_hidden", "d_hidden"):
        for units in cfg[name]:
            positive_integer(units, name)
    g_dims = [int(cfg["latent_dim"])] + [int(d) for d in cfg["g_hidden"]] + [IMAGE_DIM]
    d_dims = [IMAGE_DIM] + [int(d) for d in cfg["d_hidden"]] + [1]
    return g_dims, d_dims


def init_params(cfg, rng):
    """Poids initiaux (He pour les couches LeakyReLU) de G et D.

    Retourne deux listes de couples [W (in, out), b (out,)] — le format interne
    de tout ce module. He plutôt que Xavier : les couches cachées sont des
    LeakyReLU, la variance 2/n_in garde le signal debout à l'initialisation.
    """
    g_dims, d_dims = layer_dims(cfg)

    def net(dims):
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)
            b = np.zeros(n_out, dtype=np.float32)
            layers.append([W, b])
        return layers

    return net(g_dims), net(d_dims)


# ------------------------------------------------- Passes avant / arrière


def _leaky(x):
    return np.where(x > 0, x, LEAKY_RELU_SLOPE * x)


def _leaky_grad(x):
    return np.where(x > 0, 1.0, LEAKY_RELU_SLOPE).astype(np.float32)


def _sigmoid(x):
    # Version stable des deux côtés : exp() ne reçoit jamais un grand positif.
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def mlp_forward(x, layers, out_activation):
    """Passe avant d'un MLP LeakyReLU. → (sortie, caches pour le backward).

    out_activation : "sigmoid" (le générateur — des pixels dans [0, 1]) ou
    "linear" (le discriminateur — un LOGIT, la BCE se charge du sigmoïde).
    caches : [(entrée de la couche, pré-activation), ...] — tout ce qu'il faut
    pour dérouler la chaîne de dérivation dans mlp_backward.
    """
    caches = []
    h = x
    last = len(layers) - 1
    for i, (W, b) in enumerate(layers):
        z = h @ W + b
        caches.append((h, z))
        if i < last:
            h = _leaky(z)
        elif out_activation == "sigmoid":
            h = _sigmoid(z)
        else:
            h = z
    return h, caches


def mlp_backward(d_out, layers, caches, out_activation, d_input_needed=False):
    """Rétro-propagation du MLP. → (gradients [dW, db], gradient d'entrée).

    d_out : dérivée de la loss par rapport à la SORTIE du réseau — sauf pour
    out_activation="bce_logit", où d_out est directement σ(logit) − étiquette,
    la dérivée par rapport au LOGIT (le raccourci qui rend la BCE stable).
    d_input_needed : True quand le gradient doit continuer à travers l'entrée —
    c'est le canal par lequel le générateur reçoit la leçon du discriminateur.
    """
    grads = [None] * len(layers)
    last = len(layers) - 1
    delta = d_out

    for i in range(last, -1, -1):
        W, _ = layers[i]
        h_in, z = caches[i]
        if i == last:
            if out_activation == "sigmoid":
                s = _sigmoid(z)
                delta = delta * s * (1.0 - s)
            # "bce_logit" : delta est déjà par rapport au logit — rien à faire.
        else:
            delta = delta * _leaky_grad(z)
        dW = h_in.T @ delta
        db = delta.sum(axis=0)
        grads[i] = [dW.astype(np.float32), db.astype(np.float32)]
        if i > 0 or d_input_needed:
            delta = delta @ W.T
    return grads, (delta if d_input_needed else None)


# --------------------------------------------------------------------- Adam


class Adam:
    """Adam écrit à la main — un état (m, v) par tenseur, β₁ = 0,5 (DCGAN)."""

    def __init__(self, layers, lr, beta1=0.5, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
        self.v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]

    def step(self, layers, grads):
        self.t += 1
        b1t = 1.0 - self.b1 ** self.t
        b2t = 1.0 - self.b2 ** self.t
        for i, (params, gs) in enumerate(zip(layers, grads)):
            for j, (p, g) in enumerate(zip(params, gs)):
                self.m[i][j] = self.b1 * self.m[i][j] + (1 - self.b1) * g
                self.v[i][j] = self.b2 * self.v[i][j] + (1 - self.b2) * g * g
                m_hat = self.m[i][j] / b1t
                v_hat = self.v[i][j] / b2t
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ------------------------------------------------------------ Entraînement


def _bce_with_logits(logits, target):
    """BCE stable calculée sur les logits : max(x,0) − x·y + log(1 + e^−|x|)."""
    x = logits
    return float(np.mean(np.maximum(x, 0) - x * target + np.log1p(np.exp(-np.abs(x)))))


def batch_ranges(n, batch_size):
    """Return half-open batches that cover every sample exactly once."""
    positive_integer(n, "n")
    positive_integer(batch_size, "batch_size")
    return [
        (start, min(start + int(batch_size), int(n)))
        for start in range(0, int(n), int(batch_size))
    ]


def train_gan(X, cfg, progress=None):
    """Entraîne un GAN neuf sur X (n, 784). → (g_layers, d_layers, history, snapshots).

    history : dict de listes (une valeur par époque) — "g_loss", "d_loss",
    "d_real_acc", "d_fake_acc". Les deux accuracies disent qui gagne : un D
    bloqué à 100 % n'envoie plus de gradient utile, un D à 50 % est aveugle ;
    l'équilibre sain oscille entre les deux.

    snapshots : [(époque, images (16, 784)), ...] — les MÊMES 16 z décodés à
    ~5 moments de l'entraînement : la maturation du générateur se voit d'un
    coup d'œil, colonne par colonne.
    """
    validate_training_config(cfg)
    real_label = float(cfg.get("label_smoothing", 0.9))
    if not np.isfinite(real_label) or not 0 < real_label <= 1:
        raise ValueError("label_smoothing doit être fini et compris dans ]0, 1].")
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)

    latent_dim = int(cfg["latent_dim"])
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != IMAGE_DIM:
        raise ValueError(f"X doit avoir la forme (n, {IMAGE_DIM}), reçu : {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("X contient des valeurs NaN ou infinies.")
    if np.any(X < 0) or np.any(X > 1):
        raise ValueError("Les images du GAN doivent être normalisées dans [0, 1].")
    n = len(X)
    batch_size = min(batch_size, n)

    g_layers, d_layers = init_params(cfg, rng)
    opt_g = Adam(g_layers, lr=float(cfg["learning_rate"]))
    opt_d = Adam(d_layers, lr=float(cfg["learning_rate"]))

    # Les mêmes z à chaque instantané : seule l'évolution de G change l'image.
    z_snapshot = rng.standard_normal((16, latent_dim)).astype(np.float32)
    checkpoints = sorted({1, max(1, epochs // 4), max(1, epochs // 2),
                          max(1, (3 * epochs) // 4), epochs})
    snapshots = []

    history = {"g_loss": [], "d_loss": [], "d_real_acc": [], "d_fake_acc": []}
    ranges = batch_ranges(n, batch_size)
    steps = len(ranges)

    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        g_sum = d_sum = racc_sum = facc_sum = 0.0
        seen = 0

        for start, stop in ranges:
            x_real = X[perm[start:stop]]
            m = len(x_real)

            # ---- Pas de D : vraies images à real_label, générées à 0.
            z = rng.standard_normal((m, latent_dim)).astype(np.float32)
            x_fake, _ = mlp_forward(z, g_layers, "sigmoid")

            logits_r, caches_r = mlp_forward(x_real, d_layers, "linear")
            logits_f, caches_f = mlp_forward(x_fake, d_layers, "linear")

            d_loss = 0.5 * (_bce_with_logits(logits_r, real_label)
                            + _bce_with_logits(logits_f, 0.0))
            # dBCE/dlogit = σ(logit) − étiquette ; le ½ moyenne les deux moitiés.
            delta_r = (_sigmoid(logits_r) - real_label) / (2.0 * m)
            delta_f = _sigmoid(logits_f) / (2.0 * m)
            grads_r, _ = mlp_backward(delta_r, d_layers, caches_r, "bce_logit")
            grads_f, _ = mlp_backward(delta_f, d_layers, caches_f, "bce_logit")
            grads_d = [[gr[0] + gf[0], gr[1] + gf[1]]
                       for gr, gf in zip(grads_r, grads_f)]
            # Both discriminator accuracies describe the same pre-update batch.
            real_accuracy = float(np.mean(logits_r > 0))
            fake_accuracy = float(np.mean(logits_f <= 0))
            opt_d.step(d_layers, grads_d)

            # ---- Pas de G (non saturant) : faire dire « vrai » à D.
            z = rng.standard_normal((m, latent_dim)).astype(np.float32)
            x_fake, caches_g = mlp_forward(z, g_layers, "sigmoid")
            logits_f, caches_d = mlp_forward(x_fake, d_layers, "linear")

            g_loss = _bce_with_logits(logits_f, 1.0)
            delta = (_sigmoid(logits_f) - 1.0) / m
            # Le gradient traverse D (sans le modifier) puis remonte dans G.
            _, d_x_fake = mlp_backward(delta, d_layers, caches_d, "bce_logit",
                                       d_input_needed=True)
            grads_g, _ = mlp_backward(d_x_fake, g_layers, caches_g, "sigmoid")
            opt_g.step(g_layers, grads_g)

            if not np.isfinite(g_loss) or not np.isfinite(d_loss):
                raise FloatingPointError("Loss GAN non finie : vérifier données et learning_rate.")
            g_sum += g_loss * m
            d_sum += d_loss * m
            racc_sum += real_accuracy * m
            facc_sum += fake_accuracy * m
            seen += m

        history["g_loss"].append(g_sum / seen)
        history["d_loss"].append(d_sum / seen)
        history["d_real_acc"].append(racc_sum / seen)
        history["d_fake_acc"].append(facc_sum / seen)

        if epoch in checkpoints:
            imgs, _ = mlp_forward(z_snapshot, g_layers, "sigmoid")
            snapshots.append((epoch, imgs.astype(np.float32)))

        if progress is not None:
            progress(epoch / epochs,
                     desc=f"Époque {epoch}/{epochs} — G {history['g_loss'][-1]:.3f} "
                          f"· D {history['d_loss'][-1]:.3f}")

    return g_layers, d_layers, history, snapshots


# --------------------------------------------------- Poids ↔ vecteur plat


def flat_weights(g_layers, d_layers):
    """Tous les tenseurs de G puis de D, aplatis en UN vecteur float32."""
    parts = [p for layers in (g_layers, d_layers) for W, b in layers for p in (W, b)]
    return np.concatenate([p.ravel() for p in parts]).astype(np.float32)


def params_from_flat(flat, cfg):
    """Reconstruit (g_layers, d_layers) depuis le vecteur plat et la config.

    Les shapes ne sont pas stockées : layer_dims(cfg) les redonne dans l'ordre
    exact où flat_weights les a couchées — même principe que l'autoencodeur.
    """
    flat = np.asarray(flat, dtype=np.float32).ravel()
    if flat.size == 0 or not np.isfinite(flat).all():
        raise ValueError("Le vecteur de poids doit être non vide et fini.")
    g_dims, d_dims = layer_dims(cfg)
    expected = sum(
        n_in * n_out + n_out
        for dims in (g_dims, d_dims)
        for n_in, n_out in zip(dims[:-1], dims[1:])
    )
    if flat.size != expected:
        raise ValueError(
            f"Poids incompatibles avec cette architecture : {flat.size} valeurs "
            f"pour {expected} attendues. Le modèle vient-il d'une autre config ?"
        )
    offset = 0
    nets = []
    for dims in (g_dims, d_dims):
        layers = []
        for n_in, n_out in zip(dims[:-1], dims[1:]):
            W = flat[offset:offset + n_in * n_out].reshape(n_in, n_out).copy()
            offset += n_in * n_out
            b = flat[offset:offset + n_out].copy()
            offset += n_out
            layers.append([W, b])
        nets.append(layers)
    if offset != flat.size:
        raise ValueError(
            f"Poids incompatibles avec cette architecture : {flat.size} valeurs "
            f"pour {offset} attendues. Le modèle vient-il d'une autre config ?"
        )
    return nets[0], nets[1]


# ------------------------------------------------------------------ Cache


def _key_of(weights):
    return _FINGERPRINTS.key(weights)


def remember(weights, cfg, g_layers=None, d_layers=None):
    """Associe un vecteur de poids à sa config (appelé par load/train)."""
    key = _FINGERPRINTS.bind(weights)
    _KNOWN[key] = cfg
    while len(_KNOWN) > _KNOWN_MAX:
        _KNOWN.pop(next(iter(_KNOWN)))
    if g_layers is not None and d_layers is not None:
        _PARAMS[key] = (g_layers, d_layers)
        while len(_PARAMS) > _PARAMS_MAX:
            _PARAMS.pop(next(iter(_PARAMS)))


def config_of(weights, key=None):
    cfg = _KNOWN.get(key if key is not None else _key_of(weights))
    if cfg is None:
        raise RuntimeError(
            "Poids inconnus du cache : recharge le modèle (algo.load) avant "
            "d'encoder — l'architecture vit dans ses métadonnées."
        )
    return cfg


def params_of(weights, key=None):
    """Return parsed generator/discriminator parameters, cached by fingerprint."""
    key = key if key is not None else _key_of(weights)
    if key not in _PARAMS:
        _PARAMS[key] = params_from_flat(weights, config_of(weights, key))
        while len(_PARAMS) > _PARAMS_MAX:
            _PARAMS.pop(next(iter(_PARAMS)))
    return _PARAMS[key]


# --------------------------------------------------- Génération / inversion


def generate_batch(Z, weights):
    """(m, latent_dim) → (m, 784) dans [0, 1] : G(z), la raison d'être du GAN."""
    key = _key_of(weights)
    cfg = config_of(weights, key)
    g_layers, _ = params_of(weights, key)
    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim == 1:
        Z = Z[None, :]
    if (Z.ndim != 2 or Z.shape[0] == 0
            or Z.shape[1] != int(cfg["latent_dim"]) or not np.isfinite(Z).all()):
        raise ValueError(
            f"Z doit être une matrice finie (n, {int(cfg['latent_dim'])}), reçu : {Z.shape}."
        )
    out, _ = mlp_forward(Z, g_layers, "sigmoid")
    return np.clip(out.reshape(len(Z), IMAGE_DIM), 0.0, 1.0)


def _bank_of(weights):
    """La banque d'inversion (Zb, Xb) de ce modèle — régénérée au besoin.

    Déterministe : les z sortent d'un default_rng(seed du modèle), donc deux
    chargements du même .npz donnent exactement la même banque — encoder est
    reproductible sans rien stocker de plus dans le fichier.
    """
    key = _key_of(weights)
    if key not in _BANKS:
        cfg = config_of(weights, key)
        rng = np.random.default_rng(int(cfg["seed"]) + 1)  # ≠ graine d'entraînement
        Zb = rng.standard_normal((BANK_SIZE, int(cfg["latent_dim"]))).astype(np.float32)
        g_layers, _ = params_of(weights, key)
        Xb, _ = mlp_forward(Zb, g_layers, "sigmoid")
        Xb = np.clip(Xb.reshape(len(Zb), IMAGE_DIM), 0.0, 1.0)
        _BANKS[key] = (Zb, Xb)
        while len(_BANKS) > _BANKS_MAX:
            _BANKS.pop(next(iter(_BANKS)))
    return _BANKS[key]


def encode_batch(X, weights):
    """(n, 784) → (n, latent_dim) par inversion approximative (banque de G).

    Le z d'une image est celui de l'image GÉNÉRÉE la plus proche (L2) parmi
    BANK_SIZE candidates. ||x||² étant constant par requête, argmin ||x − b||²
    = argmin (||b||² − 2 x·b) : on évite de matérialiser les différences.
    """
    Zb, Xb = _bank_of(weights)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if (X.ndim != 2 or X.shape[0] == 0 or X.shape[1] != IMAGE_DIM
            or not np.isfinite(X).all()):
        raise ValueError(f"X doit être une matrice finie (n, {IMAGE_DIM}), reçu : {X.shape}.")

    sq_b = np.sum(Xb * Xb, axis=1)          # (BANK_SIZE,)
    out = np.empty((len(X), Zb.shape[1]), dtype=np.float32)
    step = 1024                              # bornes mémoire : (1024, BANK_SIZE)
    for i in range(0, len(X), step):
        chunk = X[i:i + step]
        d2 = sq_b[None, :] - 2.0 * chunk @ Xb.T
        out[i:i + len(chunk)] = Zb[np.argmin(d2, axis=1)]
    return out


def decode_batch(Z, weights):
    """Alias de generate_batch — le « décodeur » du GAN EST son générateur."""
    return generate_batch(Z, weights)
