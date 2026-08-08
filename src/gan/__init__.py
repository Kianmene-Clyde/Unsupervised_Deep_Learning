"""GAN — Generative Adversarial Network, écrit from scratch en NumPy.

Deux réseaux s'affrontent (Goodfellow et al., 2014) : un GÉNÉRATEUR apprend
z ~ N(0, I) → image, un DISCRIMINATEUR apprend à distinguer ses productions
des vraies images ; chacun s'améliore contre l'autre jusqu'à ce que les
générations deviennent indistinguables.

Contrairement à l'autoencodeur et au VAE (moteur Keras), TOUT est ici écrit à
la main en NumPy — passe avant, rétro-propagation couche par couche, Adam —
pour montrer ce que les frameworks cachent (utils/model.py).

Le GAN est le seul algo du projet SANS encodeur : il n'apprend que le sens
z → image. Sa place dans les 3 tâches est donc inversée — excellent en
génération (sa raison d'être), volontairement mauvais en compression (le code
d'une image est une inversion approximative par banque de générations), et sa
projection latente se lit avec des pincettes (voir algo.py).

Structure calquée sur src/vae/ :
    algo.py                 l'adaptateur vers l'interface Algo (algo_base)
    utils/model.py          le GAN NumPy : forward/backward/Adam + poids ↔ vecteur plat
    utils/registry.py       sauvegarde/chargement des modèles (src/gan/models/)
    utils/codec.py          le « code » d'une image : son z inversé, hashable/affichable
    utils/visualization.py  les vues : duel G/D, maturation, interpolation…
"""
