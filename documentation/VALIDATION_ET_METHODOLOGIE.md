# Bilan scientifique et technique — 5 novembre 2026

## Résultat et portée

Le projet a été corrigé dans le workspace en préservant ses six méthodes et leurs implémentations pédagogiques. La rigueur du protocole, les erreurs numériques vérifiables et les parcours d’utilisation ont été traités avant toute revendication de performance.

**Validation locale : 93 tests réussis, six entraînements/sauvegardes/rechargements minimaux, 24 exécutions réelles de comparaison et un serveur Gradio avec export vérifiés.** Ces validations rapides ne démontrent ni convergence des réseaux ni supériorité scientifique d’une méthode.

Le ZIP initial est la référence historique. Ses 40 tests ont été exécutés dans le même environnement Python 3.12 installé pour cette intervention : `output/validation/baseline_tests.txt`. La suite actuelle a été exécutée avec succès dans `output/validation/tests_final.txt` et `tests.xml`. Les durées des deux suites ne sont pas comparables comme mesure de vitesse : imports, caches et couverture diffèrent.

## Audit prioritaire et corrections

| Constat initial | Correction | Preuve |
|---|---|---|
| Chaque configuration du balayage PCA/VAE évaluée sur le test | Validation stratifiée issue du train ; test réservé à un gagnant par méthode avec `--final-test` | `tests/test_sweep_selection.py`, `sweep_mnist_verified/` |
| Features de la « FID simplifiée » apprises sur le test | FD-PCA apprise sur train uniquement, partagée, gelée et sauvegardée | Tests du protocole et manifestes des comparaisons |
| Silhouettes de clusters/pixels et classes/latent mêlées | Colonnes distinctes et définitions explicites | `results_*.csv`, tests métriques |
| Ratio théorique assimilable à une taille de stockage réelle | Bits idéaux séparés des octets NPZ codes/modèle et amortissement explicite | Tests stockage et colonnes dédiées |
| Courbes d’exploration reliant des configurations hétérogènes | Nuages de points descriptifs, avertissement sur les facteurs confondants | Script de balayage et figures de validation |
| Moyenne VAE des mini-batches non pondérée | Moyenne pondérée par effectif ; lot final correctement traité | Test numérique contrôlé et comparaison avant/après |
| SOM appris en hexagonal mais générant avec quatre voisins | Six voisins cohérents ; grille 1×1 fonctionnelle | Tests géométriques et comparaison avant/après |
| Covariance PCA d’une classe singleton non finie | Covariance régularisée de dimension correcte | Tests numériques et comparaison avant/après |
| Codes latents hashables dépendant d’un tableau mutable | Copie indépendante, valeurs immuables et hash cohérent pour ±0 | Tests codec |
| Paramètres invalides, coûts ou NaN découverts tardivement | Validations partagées AE/VAE/GAN et gardes K-means/SOM/PCA | Tests paramètres et cas extrêmes |
| Import de l’application déclenchant MNIST | Construction explicite `build_app()` et données paresseuses | Tests UI et serveur réel |
| Cache métadonnées périmé et nom de modèle réutilisé | Invalidation par fichier/provenance, suffixes préservant les modèles existants | Tests registre/UI |
| Export avec noms de fichiers non portables et sans provenance | Noms normalisés et `manifest.json` ; rendus matplotlib sérialisés | Tests UI, ZIP réel et HTTP |
| Cache Quick, Draw! potentiellement invalide et split peu robuste aux petits effectifs | Validation des NPY, téléchargement temporaire unique, découpage stratifié | Tests données et téléchargement réel des trois catégories |
| Poids préentraînés annoncés mais absents du ZIP | README rectifié ; entraînement initial expliqué | Inventaire du ZIP initial |

La revue indépendante a également identifié le cas d’un gagnant VAE évincé du cache pendant une longue grille. Sa configuration est maintenant restaurée depuis les métadonnées avant l’évaluation finale ; un test vide les caches pour reproduire ce cas.

## Références numériques avant/après

Mesures réellement exécutées contre les sources du ZIP initial, avec SHA256 dans `output/validation/algorithm_regressions.json` :

| Vérification contrôlée | Avant | Après / attendu |
|---|---|---|
| Génération SOM 1×1 | `ValueError: high <= 0` | Trois images finies |
| Voisins d’un neurone intérieur SOM | 4 | 6, conformes aux coordonnées hexagonales |
| Covariance PCA d’une classe singleton en 2D | Matrice NaN | `1e-6 × I₂` |
| Perte VAE : décodeur nul, 17 images, dont une blanche, batch 16 | 0.03125 | 0.0588235296, attendu `1/17` |

Ces fixtures mesurent la correction mathématique, pas la qualité de reconnaissance ou de génération. Les gradients GAN ont été comparés à des différences finies ; Adam et la stabilité de la BCE sur des logits extrêmes ont aussi été vérifiés.

## Protocole scientifique actuel

1. MNIST conserve son test officiel. Quick, Draw! reçoit un test stratifié, avec au moins un exemple de chaque classe de part et d’autre.
2. Une validation stratifiée de 20 % est réservée dans le train. Chaque classe doit avoir au moins deux exemples pour permettre cette séparation.
3. Aucun label n’est fourni comme cible aux modèles du comparatif. La stratification et les métriques externes utilisent les labels ; les vues PCA par classe sont explicitement conditionnelles.
4. Le comparatif utilise des configurations fixées à l’avance et partage données et PCA d’évaluation entre méthodes et graines. Le balayage PCA/VAE sélectionne sur MSE de validation.
5. Le test n’est évalué qu’en bilan final, sans retour de ses mesures vers la sélection. Relancer un protocole en orientant les choix à partir du test invaliderait cette réserve.
6. Les graines, versions, options, configurations et empreintes des sources/données accompagnent les résultats. Elles n’impliquent pas une identité bit à bit entre matériels et plateformes.

FD-PCA compare des distributions dans un espace PCA entraîné sur le train. Ce n’est pas la FID standard. Les coûts NPZ sont des octets réellement sérialisés en mémoire dans le format annoncé ; ils ne sont pas une taille minimale du décodeur. Le coût du modèle ajusté est amorti sur le lot évalué, sans inclure l’environnement logiciel. Les deux silhouettes n’ont pas le même objet et ne doivent pas être fusionnées dans un classement.

## Essais réels bornés sur les deux datasets

Six méthodes × deux graines (42, 43) × deux datasets = **24 exécutions**. Pour chaque dataset : 200 images de train, 64 images de validation, 64 générations ; une époque pour les réseaux et le SOM. Les métriques essentielles et les coûts sérialisés sont finis. Les champs inapplicables, par exemple une silhouette de clusters pour une méthode continue, restent indéfinis.

**Ces nombres décrivent un test de fonctionnement à faible budget.** Les réseaux sont peu entraînés ; les architectures, dimensions latentes et temps de calcul diffèrent. Aucun classement général ne doit être déduit du tableau.

| Dataset | Méthode | MSE validation, moyenne ± écart type (2 graines) | FD-PCA, moyenne ± écart type |
|---|---|---|---|
| mnist | K-means (k=64) | 0.04822 ± 0.00033 | 13.837 ± 0.450 |
| mnist | Kohonen (8x8) | 0.05041 ± 0.00027 | 23.839 ± 1.167 |
| mnist | PCA (d=16) | 0.02747 ± 0.00000 | 13.765 ± 0.241 |
| mnist | AutoEncoder (z=16) | 0.23065 ± 0.00032 | 66.870 ± 0.063 |
| mnist | VAE (z=16) | 0.23072 ± 0.00016 | 65.389 ± 0.147 |
| mnist | GAN (z=64, inversion diagnostique) | 0.25852 ± 0.00201 | 56.593 ± 0.122 |
| quickdraw | K-means (k=64) | 0.07957 ± 0.00012 | 14.854 ± 2.274 |
| quickdraw | Kohonen (8x8) | 0.07690 ± 0.00201 | 20.959 ± 0.159 |
| quickdraw | PCA (d=16) | 0.05741 ± 0.00000 | 14.668 ± 0.553 |
| quickdraw | AutoEncoder (z=16) | 0.21425 ± 0.00002 | 73.186 ± 0.095 |
| quickdraw | VAE (z=16) | 0.21445 ± 0.00021 | 71.761 ± 0.446 |
| quickdraw | GAN (z=64, inversion diagnostique) | 0.24379 ± 0.00316 | 64.988 ± 3.341 |

Pour le GAN, la MSE ci-dessus correspond à une inversion approximative diagnostique, pas à un codec natif.

Preuves complètes :

- `output/validation/benchmark_mnist_final/`
- `output/validation/benchmark_quickdraw_final/`

Chaque dossier contient les lignes détaillées, statistiques, manifeste, PCA sauvegardée et panneaux PNG. `train_s` inclut les statistiques de génération ajustées sur train ; `evaluation_s` inclut initialisations, imports paresseux, métriques et sérialisation. Aucune accélération ni mesure d’inférence à chaud n’est revendiquée.

## Sélection et utilisation vérifiées

Le balayage réel `output/validation/sweep_mnist_verified/` a entraîné deux PCA et deux VAE sur 200 MNIST, comparé leur MSE sur 64 images de validation, puis évalué le seul gagnant par méthode sur 64 images du test avec `--final-test`. Le VAE utilise une époque, conformément à `--epochs 1` même en mode rapide. La sélection concerne la reconstruction seulement. Ce petit test n’est pas un bilan final de performance.

Le script `scripts/smoke_algorithms.py` a vérifié les six adaptateurs sur 96 entrées synthétiques, les reconstructions/codecs et la conservation des poids après sauvegarde/rechargement. Les registres sont isolés dans le dossier de sortie. Rapport : `output/validation/algorithms/smoke_algorithms.json`.

Le script `scripts/verify_app.py` a démarré Gradio sur `127.0.0.1`, avec un sous-ensemble réel de MNIST et une PCA de validation isolée. Résultats : HTTP 200, 186 composants, 120 événements ; export ZIP valide de 18 figures et manifeste de provenance. Le parcours de reconstruction a également été observé dans le navigateur. Preuves : `output/validation/ui/http_export_check.json`, `export_pca.zip` et `reconstruction.jpg`.

## Environnement et reproduction

Windows CPU, Python 3.12.11 ; versions exactes dans `requirements-lock.txt` et les manifestes. `pip check` ne détecte aucun conflit. Les dépendances ont été installées dans `.venv`, les caches de données sont locaux. Python 3.14 initial ne permettait pas la collecte complète faute de TensorFlow ; il n’est pas l’environnement validé pour l’application.

Depuis la racine avec l’environnement activé :

```powershell
python -m pip install -r requirements-lock.txt
python -m pip check
python -m pytest -q
python scripts/smoke_algorithms.py --output tmp/smoke
python src/app.py
```

Comparatif rapide réellement vérifié, en changeant le dossier de sortie s’il existe déjà :

```powershell
python src/experiments/compare_all.py --datasets mnist quickdraw --quick --epochs 1 --n-train 200 --n-test 64 --n-gen 64 --seeds 42 43 --evaluation-split validation
```

Sélection bornée avec évaluation du gagnant :

```powershell
python src/experiments/run_experiments.py --datasets mnist --algos pca,vae --quick --epochs 1 --max-runs 2 --n-train 200 --n-validation 64 --n-test 64 --no-save --final-test
```

Le workflow GitHub prépare les tests et le smoke synthétique sous Windows/Python 3.12. Il n’a pas été exécuté sur GitHub pendant cette intervention. Références de compatibilité : [installation TensorFlow](https://www.tensorflow.org/install/pip), [API Gradio Blocks](https://www.gradio.app/docs/gradio/blocks).

## Limites et suite scientifique nécessaire

- Entraînements complets, étude de convergence, budgets de calcul comparables et répétitions plus nombreuses restent nécessaires avant toute revendication de performance.
- Les diagnostics pixel/voisinage sont limités à des sous-échantillons ; ils ne démontrent pas l’absence de mémorisation ni une bonne qualité perceptuelle.
- La variabilité à deux graines ne constitue pas un intervalle de confiance fiable ni un test de significativité.
- Les dix notebooks sont inventoriés dans `notebook_inventory.json` mais n’ont pas été tous réexécutés. Le rapport, les tableaux, figures et slides initiaux sont explicitement historiques.
- La variante `quickdraw10`, les architectures convolutionnelles de toutes les grilles et les autres systèmes d’exploitation n’ont pas reçu le même contrôle de bout en bout que les deux datasets principaux et les configurations rapides.
- Les sources et données historiques sont conservées. Aucun déploiement distant, entraînement payant ou remplacement des implémentations maison n’a été effectué.
