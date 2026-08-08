"""Autoencodeur — le travail notebook (autoencodeur.py) et l'adaptateur app.

autoencodeur.py est le script/notebook d'origine (dense + conv, entraînement
MNIST, figures) ; il N'EST PAS modifié — et jamais importé : exécuté tel quel,
il télécharge MNIST et lance 200 époques d'entraînement au premier import.

Ce package y ajoute la façade pour l'app Gradio, sur le modèle de src/pca/ :
    algo.py             l'adaptateur vers l'interface Algo (algo_base)
    utils/model.py      construction/entraînement Keras + poids ↔ vecteur plat
    utils/registry.py   sauvegarde/chargement des modèles (src/autoencoder/models/)
    utils/codec.py      le code latent d'une image, hashable et affichable
    utils/visualization.py  les vues : manifold 2D, traversées, génération…
"""
