"""VAE — Variational Autoencoder, l'adaptateur app et son moteur Keras.

Le VAE reprend la structure encodeur → latent → décodeur de src/autoencoder/,
avec trois différences que ce package fait vivre :

- l'encodeur ne prédit plus un point latent mais les PARAMÈTRES d'une gaussienne
  (z_mean, z_log_var) ; le reparameterization trick tire z = μ + σ·ε à
  l'entraînement (utils/model.py) ;
- la loss ajoute la KL divergence vers N(0, 1), pondérée par kl_weight :
  Loss = reconstruction + kl_weight · KL ;
- le latent étant tiré vers N(0, 1), la génération échantillonne z ~ N(0, I)
  DIRECTEMENT — là où l'autoencodeur simple devait passer par la gaussienne
  empirique de ses codes (voir algo.py).

Structure calquée sur src/autoencoder/ :
    algo.py                 l'adaptateur vers l'interface Algo (algo_base)
    utils/model.py          construction/entraînement Keras + poids ↔ vecteur plat
    utils/registry.py       sauvegarde/chargement des modèles (src/vae/models/)
    utils/codec.py          le code latent (z_mean) d'une image, hashable/affichable
    utils/visualization.py  les vues, réutilisées de src/autoencoder/
"""
