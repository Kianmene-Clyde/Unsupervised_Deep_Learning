# Unsupervised Deep Learning — MNIST et Quick, Draw!

Boîte à outils scientifique comparant **six méthodes**, de K-means au GAN, sur la projection, la reconstruction/compression et la génération d’images 28 × 28. Les implémentations pédagogiques maison sont conservées ; l’application Gradio et les expériences utilisent les mêmes composants.

| Méthode | Implémentation | Reconstruction | Génération |
|---|---|---|---|
| K-means | Lloyd écrit à la main avec TensorFlow | Index de centroïde | Mélange autour des clusters appris |
| SOM / Kohonen | NumPy, grille hexagonale | Index de prototype | Interpolation entre voisins hexagonaux |
| PCA | NumPy, décomposition spectrale | Coefficients linéaires | Gaussienne latente globale dans le comparatif |
| AutoEncoder | Architecture et entraînement Keras | Code latent | Gaussienne empirique du latent |
| VAE | Keras, reparamétrisation et GradientTape | Moyenne latente | Échantillonnage du prior |
| GAN | NumPy : forward, gradients et Adam maison | Inversion approximative diagnostique | Générateur appris |

Le GAN ne possède pas d’encodeur natif. Les démonstrations PCA par classe utilisent les labels et sont explicitement conditionnelles ; elles sont distinctes de la génération non supervisée du comparatif.

## Installation reproductible

Environnement vérifié localement sur **Windows, CPU, Python 3.12.11**, avec TensorFlow 2.21.0, Keras 3.15.1 et Gradio 6.31.0. Le fichier `.python-version` indique Python 3.12.

Depuis la racine, dans PowerShell avec Python 3.12 installé :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip check
python -m pytest -q
```

Sous Linux/macOS : créer l’environnement avec `python3.12 -m venv .venv`, puis l’activer avec `source .venv/bin/activate`. Ces plateformes n’ont pas été validées pendant cette intervention ; les mêmes versions ne garantissent pas des résultats bit à bit identiques.

`requirements-lock.txt` fige les dépendances de l’application **et des tests** de l’environnement vérifié. `requirements.txt` conserve les plages de compatibilité ; `requirements-notebooks.txt` contient les outils exploratoires optionnels, non inclus dans cette validation. Pour renouveler le verrouillage, installer dans un environnement neuf et refaire les vérifications.

## Application

```powershell
python src/app.py
```

Ouvrir l’adresse locale affichée, généralement `http://127.0.0.1:7860`. Choisir un algorithme et un dataset, puis entraîner un premier modèle ou charger un modèle enregistré. L’import Python ne charge aucun dataset ; la construction explicite `app.build_app()` initialise l’interface.

Onglets : entraînement, compression/décompression, espace latent, dictionnaire, vues propres à la méthode, courbes et export ZIP. Les exports incluent des noms de fichiers portables et un `manifest.json` décrivant le modèle et les paramètres de visualisation. Un nom de modèle déjà utilisé est conservé ; un suffixe est attribué à la nouvelle version.

**Aucun poids préentraîné n’était fourni dans le ZIP original.** Les modèles créés dans l’application sont enregistrés dans `src/<algorithme>/models/`, avec algorithme, dataset et configuration. Les modèles synthétiques de validation ne constituent pas des modèles de démonstration MNIST/Quick, Draw!.

## Données et séparation des partitions

Point d’entrée : `src/data_import.py`.

- `mnist` : partitions officielles 60 000 / 10 000 via Keras.
- `quickdraw` : catégories cat, apple et car, 10 000 dessins par classe par défaut.
- `quickdraw10` : dix catégories, 6 000 dessins par classe par défaut.

Les bitmaps Quick, Draw! officiels sont téléchargés dans `data/numpy_bitmap/`, validés puis renommés atomiquement. Prévoir plusieurs centaines de Mo pour les trois catégories ; le nombre de lignes utilisées ne réduit pas le téléchargement initial. Un cache invalide produit une erreur explicite. Les données brutes sont ignorées par Git.

Quick, Draw! reçoit un découpage stratifié train/test. Les expériences réservent ensuite 20 % du train pour la validation, sans modifier le test. Les labels servent à la stratification et à l’évaluation, jamais aux cibles d’apprentissage des modèles du comparatif. Les empreintes des partitions sont enregistrées.

## Comparaison des six méthodes

Vérification rapide, sur validation, sans recherche d’hyperparamètres :

```powershell
python src/experiments/compare_all.py --quick --epochs 1 --datasets mnist --n-train 200 --n-test 64 --n-gen 64 --seeds 42 43 --evaluation-split validation
```

La même commande avec `--datasets mnist quickdraw` vérifie les deux jeux de données. `--n-test` désigne le nombre d’images du **split d’évaluation choisi**, validation ou test.

Après gel des configurations, exemple de protocole final borné, plus coûteux et non exécuté ici :

```powershell
python src/experiments/compare_all.py --datasets mnist quickdraw --n-train 10000 --n-test 2000 --n-gen 2000 --epochs 20 --seeds 42 43 44 --evaluation-split test
```

Les sorties sont horodatées dans `src/experiments/results/comparison_runs/` : résultats détaillés, moyenne/écart type, protocole et versions, empreintes sources/données, PCA d’évaluation sauvegardée et panneaux visuels. Un dossier de résultats existants n’est pas écrasé. `--reuse-gan` est une option exploratoire : la provenance d’un modèle externe n’est pas contrôlée par ce protocole.

Métriques et interprétation :

- MSE, PSNR et SSIM : reconstruction sur la partition annoncée.
- Pureté et silhouette clusters/pixels : organisation des clusters dans les pixels.
- Silhouette classes/latent : séparation externe des classes dans les représentations continues ; ce n’est pas la même mesure que la silhouette précédente.
- **FD-PCA** : distance de Fréchet dans une PCA apprise sur le train, partagée entre méthodes et gelée. Ce n’est pas la FID Inception standard.
- Compression : budget théorique du code séparé des octets NPZ réellement sérialisés, avec coût du modèle amorti sur le lot évalué. Ce coût dépend du format et du volume d’amortissement.
- Diversité et voisins proches : diagnostics bornés, qui ne prouvent pas l’absence de mémorisation ni la qualité perceptuelle.

Les capacités et budgets de calcul des méthodes diffèrent. Les temps incluent des diagnostics et des initialisations ; ils ne mesurent pas une inférence à chaud.

## Sélection d’hyperparamètres PCA / VAE

```powershell
python src/experiments/run_experiments.py --datasets mnist --algos pca,vae --quick --epochs 1 --max-runs 2 --n-train 200 --n-validation 64 --n-test 64 --no-save
```

Chaque configuration est évaluée **sur validation**. Ajouter `--final-test` évalue seulement le gagnant de validation de chaque algorithme sur le test, sans réentraînement. La sélection utilise la MSE de reconstruction ; elle ne prétend pas optimiser la génération. Éviter de relancer cette étape pour orienter des choix à partir du test.

`--max-runs` s’applique par algorithme et par dataset. `--epochs` explicite est prioritaire sur le mode rapide. Les modes `curated`, `ofat` et `grid` sont disponibles ; `--dry-run` affiche le plan sans entraîner. Les graphiques de configurations hétérogènes sont descriptifs et n’isolent pas un effet causal.

## Validation et preuves

```powershell
python -m pytest -q
python scripts/smoke_algorithms.py --output tmp/smoke
```

Le second outil vérifie les six entraînements minimaux, les codecs et les sauvegardes/rechargements sur des **entrées synthétiques**, dans des registres isolés.

Les preuves de l’intervention sont dans `output/validation/`. Lire le [bilan scientifique et technique](documentation/VALIDATION_ET_METHODOLOGIE.md) pour les corrections avant/après, les essais réels et les limites. Le workflow `.github/workflows/tests.yml` prépare l’exécution automatique des tests et du smoke synthétique sur Windows/Python 3.12 ; son exécution distante reste à vérifier après publication du dépôt.

## Organisation et documents historiques

- `src/algo_base.py` : contrat commun et validation des paramètres.
- `src/<algorithme>/` : adaptateur, cœur, codec, visualisations et registre.
- `src/utils/` : métriques, stockage partagé et utilitaires.
- `src/experiments/` : comparaison et sélection sur validation.
- `tests/` : tests numériques, données, registres, interface et protocole.
- `documentation/Rapport_final/` : rendu initial conservé, marqué historique.

Les notebooks, anciennes figures, anciens tableaux et slides sont conservés comme historique exploratoire. Ils n’ont pas tous été réexécutés et ne remplacent pas le nouveau protocole. Aucun gain de qualité prédictive ou de vitesse n’est revendiqué sans expérience dédiée.

Quick, Draw! : source et licence CC BY 4.0 dans `documentation/quickdraw-dataset-master/`. MNIST : source via Keras et page officielle du dataset ; conserver les attributions applicables lors de sa redistribution.
