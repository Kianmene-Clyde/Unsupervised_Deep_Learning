"""PCA — les notebooks (compression, dim_reduction, generation) et l'adaptateur app.

Le calcul et les tracés vivent dans src/utils/pca/ (le terrain des notebooks de
ce dossier). Ce package n'y ajoute que la façade pour l'app Gradio :
    algo.py             l'adaptateur vers l'interface Algo (algo_base)
    utils/registry.py   sauvegarde/chargement des modèles (src/pca/models/)
    utils/codec.py      le code PCA d'une image, hashable et affichable
    utils/visualization.py  capture des figures notebook + vues exigées par l'app
"""
