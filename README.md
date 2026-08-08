# Unsupervised Deep Learning — MNIST & Quick, Draw!

*Unsupervised Deep Learning*. Une boîte à outils de
**six algorithmes d'apprentissage non supervisé**, chacun discuté sur
**trois tâches**, sur les datasets **MNIST** et **Quick, Draw!**:

| Algo | Cœur | Tâche 1 · Projection | Tâche 2 · Compression | Tâche 3 · Génération |
|---|---|---|---|---|
| K-means | TensorFlow (Lloyd écrit main) | distributions par cluster | ✅ codec 1 entier | mélange de gaussiennes sur les clusters |
| Kohonen (SOM) | NumPy pur | carte topologique + U-matrix | ✅ codec 1 entier | interpolation entre neurones voisins |
| PCA | NumPy pur | ✅ nuages PC1×PC2×PC3 | ✅ codec k flottants | gaussienne latente par classe |
| AutoEncoder | Keras (archi + entraînement maison) | ✅ nuages z₁×z₂ + manifold | ✅ codec z flottants | gaussienne empirique du latent |
| VAE | Keras + boucle GradientTape maison | ✅ latent régularisé N(0,1) | ✅ codec z_mean | ✅ z ~ N(0, I) direct |
| **GAN** | **NumPy pur (forward/backward/Adam à la main)** | z inversés (banque) | ❌ volontairement (pas d'encodeur) | ✅ z ~ N(0, I) — sa raison d'être |

Le tableau comparatif chiffré (MSE/PSNR/SSIM, pureté, silhouette, FID
simplifiée) est produit par `src/experiments/compare_all.py` et analysé dans
`documentation/Rapport_final/`.

## Structure du projet

```
Unsupervised_Deep_Learning/
├── data/
│   └── numpy_bitmap/          # .npy Quick, Draw! (auto-téléchargés au 1er usage)
├── documentation/
│   ├── Rapport_final/         # LE rapport consolidé du rendu final + figures
│   ├── autoencoder/ pca/ …    # rapports d'expériences par algo
│   └── quickdraw-dataset-master/  # doc officielle du dataset (catégories…)
├── src/
│   ├── algo_base.py           # l'interface commune Algo (contrat de l'app)
│   ├── data_import.py         # point d'entrée unique des datasets (MNIST/QuickDraw)
│   ├── app.py                 # l'app Gradio — les 6 algos, tous les onglets
│   ├── kmeas/  kohonen/  pca/  autoencoder/  vae/  gan/
│   │   ├── algo.py            # l'adaptateur vers l'interface Algo
│   │   ├── utils/             # le cœur : modèle, codec, registre, figures
│   │   └── models/            # les .npz entraînés (poids + métadonnées)
│   ├── gan/train_gan.py       # entraîne un GAN + produit toutes ses figures
│   ├── utils/
│   │   ├── metrics_common.py  # métriques COMMUNES (MSE/PSNR/SSIM, pureté, FID…)
│   │   └── pca/               # le cœur NumPy de la PCA (notebooks d'origine)
│   └── experiments/
│       ├── run_experiments.py # sweeps VAE + PCA (curated / ofat / grid)
│       └── compare_all.py     # LE banc final : 6 algos × 3 tâches × 2 datasets
├── cleanup_relics.py          # range les reliquats de dev dans _to_delete/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/t0r3l/Unsupervised_Deep_Learning.git
cd Unsupervised_Deep_Learning

python3 -m venv myenv
source myenv/bin/activate          # Windows : myenv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

# Optionnel : notebooks / exploration Quick Draw vectorielle
pip install -r requirements-notebooks.txt

# Optionnel : suite de tests
pip install -r requirements-dev.txt

# Vérification
python -c "import tensorflow as tf, keras; print('TF', tf.__version__, '| Keras', keras.__version__)"
python -m pytest -q
```

Environnement validé avec **Python 3.12**. Prévoir ~1 Go de disque pour les données (chaque `.npy`
Quick, Draw! fait ~100 Mo ; MNIST est téléchargé par Keras).

## Données

Tout passe par `src/data_import.py` — **aucun téléchargement manuel** :

- `load_dataset("mnist")` — 60 000 / 10 000, via Keras ;
- `load_dataset("quickdraw")` — cat, apple, car (10 000/classe par défaut) ;
- `load_dataset("quickdraw10")` — 10 classes × 6 000 (le « MNIST-large »).

Les `.npy` officiels sont téléchargés une fois dans `data/numpy_bitmap/` puis
mappés (on ne charge que les lignes utiles). Split test = 1/7 (ratio MNIST),
mélange et tirage seedés (42) : **tout le monde voit les mêmes splits**.

## L'app interactive

```bash
cd src
python app.py        # http://127.0.0.1:7860
```

Deux sélecteurs (algo × dataset) commandent tous les onglets : Entraînement
(widgets construits depuis `algo.params`), Compression/Décompression, Espace
latent, Dictionnaire, Vues de l'algo (U-matrix du SOM, duel G/D du GAN,
manifold du VAE…), Courbe d'entraînement, Export ZIP des figures. Les modèles
sont rangés par algo dans `src/<algo>/models/` et filtrés par dataset.

## Reproduire les résultats du rapport

Chaque chiffre et chaque figure du rapport final sort d'un de ces scripts,
**seeds fixes** (42) partout :

```bash
cd src

# 1. Les GAN (NumPy pur) — modèles + toutes leurs figures (~5-10 min chacun)
python gan/train_gan.py --dataset mnist
python gan/train_gan.py --dataset quickdraw --n-train 0
#    (--quick : 10 époques / 5 000 images pour vérifier que tout tourne)

# 2. LE tableau comparatif final — 6 algos × 3 tâches × 2 datasets (~15 min)
python experiments/compare_all.py
#    -> src/experiments/results/comparison/results_<ds>.{csv,md}
#    -> panneaux panel_generation_<ds>.png / panel_reconstruction_<ds>.png

# Validation rapide et bornée (échantillons équilibrés par classe)
python experiments/compare_all.py --quick --datasets mnist

# Cibler quelques algorithmes ou fixer exactement les volumes
python experiments/compare_all.py --datasets mnist --algorithms kmeans pca \
  --n-train 1000 --n-test 256 --n-gen 256

# Réutiliser explicitement un GAN du registre
# (sinon il est entraîné dans le même protocole que les autres algorithmes)
python experiments/compare_all.py --quick --datasets mnist --reuse-gan

# 3. Les sweeps d'hyperparamètres VAE + PCA (protocole détaillé)
python experiments/run_experiments.py --quick     # ou sans --quick : complet
```

Les modèles AE/VAE/GAN pré-entraînés du rendu sont déjà dans
`src/<algo>/models/` : l'app les propose sans rien réentraîner.

## Ménage

`python cleanup_relics.py --apply` range dans `_to_delete/` les reliquats du
développement (doublon de doc, fichiers vides, un .npz d'autoencodeur égaré
dans le registre VAE) — vérifier puis supprimer le dossier.

## Licence des données

Quick, Draw! est publié par Google sous CC BY 4.0
(`documentation/quickdraw-dataset-master/LICENSE`). MNIST : CC BY-SA 3.0
(Yann LeCun & Corinna Cortes).
