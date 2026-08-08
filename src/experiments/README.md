# Bancs d'expérimentation

Deux scripts complémentaires :

- **`compare_all.py`** — le banc du **rendu final** : les 6 algos (K-means,
  Kohonen, PCA, AutoEncoder, VAE, GAN) × les 3 tâches × les 2 datasets, avec
  les métriques communes de `src/utils/metrics_common.py` (MSE/PSNR/SSIM,
  pureté, silhouette, FID simplifiée). Sort les tableaux `results_<ds>.{csv,md}`
  et les panneaux visuels dans `results/comparison/` — c'est ce que le rapport
  et les slides citent. Protocole détaillé dans la docstring du script.

  ```bash
  python src/experiments/compare_all.py              # mnist + quickdraw
  python src/experiments/compare_all.py --quick      # réglages allégés
  python src/experiments/compare_all.py --quick --datasets mnist \
    --algorithms kmeans pca --output-dir tmp/comparison-smoke
  ```

  Le script entraîne par défaut chaque algorithme dans le même protocole. Utilise
  `--reuse-gan` pour charger explicitement un GAN du registre. `--n-train`,
  `--n-test` et `--n-gen` fixent les volumes ; chaque sous-ensemble est équilibré
  par classe. Un `protocol_<dataset>.json` accompagne les tableaux et enregistre
  la graine, les volumes, les versions et les options du run.

- **`run_experiments.py`** — les sweeps d'hyperparamètres **VAE + PCA**
  (curated / ofat / grid), documentés ci-dessous.

# Banc d'expérimentation VAE + PCA

Automatise les tests du **VAE** et de la **PCA** en pilotant *exactement les
mêmes classes que l'app* (`vae.algo.VAE`, `pca.algo.PCAAlgo`). Chaque run appelle
leur `train(X, p)` avec le dict d'hyperparamètres des widgets, puis **range le
modèle dans le registre de l'app** avec la même méta (`algo`, `dataset`,
`n_samples`) — donc tout ce qui est entraîné ici est **directement visualisable
dans l'interface** (`python app.py`, choisir le dataset).

## Lancement

Depuis la racine du dépôt (environnement virtuel activé) :

```bash
# Défaut : 10 tests VAE conçus main + 6 PCA, sur TOUT le train/test de Quick, Draw!
python src/experiments/run_experiments.py

# Smoke test rapide (3 époques, peu d'images)
python src/experiments/run_experiments.py --quick

# Voir ce qui serait lancé, sans rien entraîner ni sauvegarder
python src/experiments/run_experiments.py --dry-run

# Les mêmes 10 tests sur MNIST aussi
python src/experiments/run_experiments.py --datasets "quickdraw,mnist"
```

Après un run : `python app.py`, sélectionner l'algo (VAE ou PCA) puis le dataset
— le sélecteur de modèles ne montre que ceux du dataset courant, et chaque modèle
de ce balayage y apparaît.

## Modes (`--mode`)

- **`curated`** (défaut) : **10 tests VAE conçus à la main** où *plusieurs*
  hyperparamètres changent d'un test à l'autre (latent, KL, architecture,
  activations, optimiseur, lr, batch, époques) + **6 tests PCA**. C'est la liste
  éditée dans `VAE_CURATED` / `PCA_CURATED` au début du script — modifie-la pour
  changer les tests.
- **`ofat`** : un seul paramètre bouge à la fois autour d'une baseline.
- **`grid`** : produit cartésien des listes `grid` du JSON de config.
- **`both`** : ofat + grid.

En `ofat`/`grid`/`both`, chaque dataset porte une *intensité* `full` ou `light`
(`--datasets "quickdraw:full,mnist:light"`) qui choisit l'ampleur des sweeps.
L'intensité est **ignorée** en `curated`.

## Données

- **Tout le train et tout le test** par défaut (`--n-train 0 --n-test 0`). Mets
  un entier pour plafonner (utile pour aller vite).
- Le **même** `(X_train, X_test)` sert à tous les runs d'un dataset, donc les MSE
  sont comparables.

> ⚠️ Au **premier** passage sur Quick, Draw!, les bitmaps `.npy` officiels sont
> téléchargés (mis en cache dans `data/numpy_bitmap/`).

## Options utiles

| Option | Effet |
|---|---|
| `--datasets "quickdraw,mnist"` | Datasets (intensité `:full`/`:light` hors curated) |
| `--algos vae,pca` | Quels algos tester |
| `--mode curated\|ofat\|grid\|both` | Mode de génération des runs (défaut curated) |
| `--n-train` / `--n-test` | Images d'entraînement / de test (**0 = tout**, le défaut) |
| `--epochs N` | Force les époques du VAE |
| `--max-runs N` | Plafonne le nombre de runs par dataset |
| `--no-save` | Ne **pas** ranger les modèles dans l'app (juste les logs) |
| `--output DIR` | Dossier des logs (défaut : `results/<horodatage>/`) |
| `--no-plots` | Pas de figures |
| `--quick` / `--dry-run` | Smoke test / listing sans entraîner |

Config maison via `--config` (voir [`configs/default.json`](configs/default.json)).
Le **même** `(X_train, X_test)` sert à tous les runs d'un dataset (graine
`--data-seed`), donc les MSE sont comparables entre configs.

## Sorties

**Les modèles** vont dans les registres de l'app : `src/vae/models/` et
`src/pca/models/` (nommés `{algo}_{auto_name}`, collisions désambiguïsées). C'est
là que l'app les lit.

**Les logs** vont dans `results/<horodatage>/` :

- `results.csv` — une ligne par run : hyperparamètres + `train_mse`, `test_mse`
  (par pixel), `inertia`, `code_floats`, taux de compression, temps, nom
  sauvegardé ; VAE : `kl_final`, `total_loss_final`, `latent_std_mean` ; PCA :
  `explained_variance_kept`. Écrit **au fil de l'eau**.
- `details.json` — idem + historiques d'entraînement par époque.
- `run_meta.json` — comment le balayage a été lancé.
- Figures de diagnostic **propres au VAE** (aucune comparaison PCA vs VAE), par
  dataset : `vae_kl_tradeoff_<dataset>.png`, `vae_mse_vs_latent_<dataset>.png`.
