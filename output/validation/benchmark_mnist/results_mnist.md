# Comparaison sur validation

Configurations fixées ; labels utilisés pour stratification et évaluation uniquement.
FD-PCA : PCA apprise sur train, partagée et gelée ; ce n'est pas la FID standard.
Les deux silhouettes ci-dessous ont des définitions différentes et ne se classent pas ensemble.

| Méthode | Seed | MSE ↓ | PSNR ↑ | SSIM ↑ | Pureté ↑ | Silh. clusters/pixels | Silh. classes/latent | FD-PCA ↓ |
|---|---|---|---|---|---|---|---|---|
| K-means (k=64) | 42 | 0.04845 | 13.15 | 0.5967 | 0.8281 | -0.001946 | — | 14.15 |
| Kohonen (8x8) | 42 | 0.0506 | 12.96 | 0.4668 | 0.7969 | -0.03089 | — | 23.01 |
| PCA (d=16) | 42 | 0.02747 | 15.61 | 0.6028 | — | — | 0.06281 | 13.59 |
| AutoEncoder (z=16) | 42 | 0.2304 | 6.375 | 0.003707 | — | — | -0.06703 | 66.83 |
| VAE (z=16) | 42 | 0.2308 | 6.367 | 0.003429 | — | — | -0.08267 | 65.49 |
| GAN (z=64, inversion diagnostique) | 42 | 0.2599 | 5.851 | 0.01341 | — | — | -0.4454 | 56.68 |
| K-means (k=64) | 43 | 0.04799 | 13.19 | 0.6052 | 0.8594 | 0.0072 | — | 13.52 |
| Kohonen (8x8) | 43 | 0.05022 | 12.99 | 0.463 | 0.7969 | 0.0127 | — | 24.66 |
| PCA (d=16) | 43 | 0.02747 | 15.61 | 0.6028 | — | — | 0.06281 | 13.94 |
| AutoEncoder (z=16) | 43 | 0.2309 | 6.366 | 0.003286 | — | — | -0.07502 | 66.91 |
| VAE (z=16) | 43 | 0.2306 | 6.371 | 0.003459 | — | — | -0.09794 | 65.28 |
| GAN (z=64, inversion diagnostique) | 43 | 0.2571 | 5.899 | 0.00579 | — | — | -0.5712 | 56.51 |

## Stockage mesuré

Code théorique : index idéalement compacté ou flottants float32. NPZ : octets réellement
sérialisés en mémoire, en-têtes inclus ; modèle ajusté complet et métadonnées inclus dans
le coût amorti sur ce lot d'évaluation. L'environnement logiciel est exclu.
Le GAN fournit une inversion approximative diagnostique, pas un codec natif.

| Méthode | Seed | Mode reconstruction | Bits théoriques/image | NPZ codes (octets) | NPZ modèle (octets) | Bits NPZ amortis/image |
|---|---|---|---|---|---|---|
| K-means (k=64) | 42 | native_codec | 6 | 269 | 40146 | 5052 |
| Kohonen (8x8) | 42 | native_codec | 6 | 269 | 136657 | 1.712e+04 |
| PCA (d=16) | 42 | native_codec | 512 | 4054 | 74395 | 9806 |
| AutoEncoder (z=16) | 42 | native_codec | 512 | 4036 | 777566 | 9.77e+04 |
| VAE (z=16) | 42 | native_codec | 512 | 4039 | 779546 | 9.795e+04 |
| GAN (z=64, inversion diagnostique) | 42 | bank_inversion_not_native_codec | 2048 | 2548 | 4043494 | 5.058e+05 |
| K-means (k=64) | 43 | native_codec | 6 | 269 | 39965 | 5029 |
| Kohonen (8x8) | 43 | native_codec | 6 | 269 | 136583 | 1.711e+04 |
| PCA (d=16) | 43 | native_codec | 512 | 4054 | 74395 | 9806 |
| AutoEncoder (z=16) | 43 | native_codec | 512 | 4052 | 777554 | 9.77e+04 |
| VAE (z=16) | 43 | native_codec | 512 | 4045 | 779523 | 9.795e+04 |
| GAN (z=64, inversion diagnostique) | 43 | bank_inversion_not_native_codec | 2048 | 1119 | 4043334 | 5.056e+05 |

Les diagnostics de diversité/voisinage et leurs tailles d'échantillon figurent dans le CSV.
Ils détectent des signaux grossiers ; ils ne prouvent ni qualité perceptuelle ni absence de mémorisation.
Avec une seule seed, aucune estimation de dispersion n'est disponible.
