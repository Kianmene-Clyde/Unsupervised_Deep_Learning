# Expériences reproductibles

Les scripts se lancent depuis la racine du dépôt, avec l’environnement activé. Les méthodes et les cœurs sont ceux de l’application ; les sorties historiques restent dans `results/comparison/`.

## Comparaison à configurations fixées

```powershell
python src/experiments/compare_all.py --quick --epochs 1 --datasets mnist quickdraw --n-train 200 --n-test 64 --n-gen 64 --seeds 42 43 --evaluation-split validation
```

Les partitions et le tirage équilibré sont identiques pour toutes les méthodes. `--data-seed` fixe les données ; `--seeds` répète les initialisations et les générations. Une validation stratifiée est réservée dans le train. Les paramètres du script sont fixés à l’avance ; ce script ne sélectionne pas d’hyperparamètres.

Pour un bilan final, geler le protocole avant de remplacer `validation` par `test`. `--n-test` plafonne la partition choisie. Le mode rapide et une seule époque vérifient le fonctionnement, pas la convergence.

Options principales :

| Option | Rôle |
|---|---|
| `--datasets mnist quickdraw` | Jeux de données ; `quickdraw10` également disponible |
| `--algorithms kmeans pca` | Restreindre les méthodes |
| `--n-train`, `--n-test`, `--n-gen` | Volumes explicites, au moins huit images |
| `--epochs` | Budget AE/VAE/GAN/SOM ; sans effet sur K-means/PCA |
| `--seeds 42 43 44` | Répétitions ; moyenne et écart type d’échantillon |
| `--validation-fraction .2` | Validation réservée dans le train |
| `--evaluation-split validation` | Exploration ; `test` est réservé au bilan final |
| `--output-dir <nouveau_dossier>` | Sortie explicite ; protège les résultats existants |
| `--reuse-gan` | Exploration avec modèle externe, provenance non contrôlée |

Sortie par défaut : `src/experiments/results/comparison_runs/<horodatage UTC>/`.

Chaque dataset produit :

- `results_<dataset>.csv` et `.md` : mesures détaillées par méthode et graine.
- `summary_<dataset>.csv` : moyenne, écart type et nombre de valeurs finies ; les quantités inapplicables restent indéfinies.
- `protocol_<dataset>.json` : options, configurations, versions, empreintes sources/données et limites.
- `fd_pca_features_<dataset>.npz` : moyenne et composantes de la PCA d’évaluation apprise sur le train.
- Deux panneaux PNG de génération et de reconstruction/inversion.

**FD-PCA n’est pas la FID Inception.** Les deux silhouettes mesurent des objets différents. Les octets NPZ incluent en-têtes et modèle amorti sur le lot évalué ; les bits théoriques supposent un code idéal. Le GAN est présenté avec une inversion diagnostique, pas un encodeur natif. Les diagnostics de diversité sont bornés et ne démontrent pas l’absence de mémorisation.

`train_s` inclut l’ajustement et les statistiques nécessaires à la génération. `evaluation_s` inclut le reste du runner : inférence, métriques, sérialisation, imports paresseux et initialisations éventuelles. Ces durées ne sont pas des benchmarks d’inférence à chaud.

## Sélection PCA / VAE sur validation

```powershell
python src/experiments/run_experiments.py --datasets mnist --algos pca,vae --quick --epochs 1 --max-runs 2 --n-train 200 --n-validation 64 --n-test 64 --no-save
```

Chaque configuration est mesurée sur la même validation, jamais sur le test. La meilleure MSE de validation sélectionne un candidat par algorithme. Ajouter `--final-test` évalue seulement ces candidats sur le test ; leurs poids sont conservés, sans réentraînement sur validation. Cette sélection optimise la reconstruction, pas une qualité générative globale.

| Option | Rôle |
|---|---|
| `--mode curated` | Configurations conçues à la main ; mode par défaut |
| `--mode ofat` | Un facteur modifié autour d’une baseline |
| `--mode grid` ou `both` | Produit cartésien du JSON, ou OFAT + grille |
| `--config configs/default.json` | Chemin à adapter depuis la racine : `src/experiments/configs/default.json` |
| `--datasets quickdraw:full,mnist:light` | Intensité des balayages hors mode curated |
| `--n-validation` | Nombre de lignes de validation par configuration ; zéro = toute la validation |
| `--max-runs` | Limite par algorithme et par dataset ; zéro = tous les essais |
| `--epochs` | Prioritaire même avec `--quick` |
| `--no-save` | Aucun modèle dans le registre de l’application |
| `--dry-run` | Plan sans import TensorFlow, données ou entraînement |
| `--no-plots` | Désactiver les figures |
| `--output` | Dossier vide explicite ; sinon `results/<horodatage>/` |

Les sorties incluent `results.csv` avec `validation_mse`, `details.json`, une sélection par dataset, `run_meta.json` et, uniquement avec `--final-test`, `selected_test_results.json`. Le manifeste enregistre les options et empreintes permettant de vérifier la provenance. Les figures de configurations hétérogènes sont des nuages de points descriptifs ; elles ne prouvent pas l’effet causal d’un hyperparamètre.

Les noms des modèles sauvegardés sont désambiguïsés par rapport aux modèles déjà présents. Un dossier de logs non vide est refusé. Les résultats et le support de soutenance initiaux restent historiques et n’ont pas été recalculés par le nouveau protocole.
