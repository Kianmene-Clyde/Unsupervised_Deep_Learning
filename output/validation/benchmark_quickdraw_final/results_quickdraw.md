# Comparaison sur validation

Configurations fixées ; labels utilisés pour stratification et évaluation uniquement.
FD-PCA : PCA apprise sur train, partagée et gelée ; ce n'est pas la FID standard.
Les deux silhouettes ci-dessous ont des définitions différentes et ne se classent pas ensemble.

| Méthode | Seed | MSE ↓ | PSNR ↑ | SSIM ↑ | Pureté ↑ | Silh. clusters/pixels | Silh. classes/latent | FD-PCA ↓ |
|---|---|---|---|---|---|---|---|---|
| K-means (k=64) | 42 | 0.07948 | 11 | 0.4163 | 0.8438 | -0.01445 | — | 13.25 |
| Kohonen (8x8) | 42 | 0.07832 | 11.06 | 0.2783 | 0.9375 | -0.02384 | — | 21.07 |
| PCA (d=16) | 42 | 0.05741 | 12.41 | 0.431 | — | — | 0.1665 | 15.06 |
| AutoEncoder (z=16) | 42 | 0.2143 | 6.69 | 0.004834 | — | — | 0.005783 | 73.12 |
| VAE (z=16) | 42 | 0.2146 | 6.684 | 0.004262 | — | — | -0.01178 | 72.08 |
| GAN (z=64, inversion diagnostique) | 42 | 0.246 | 6.09 | 0.01497 | — | — | -0.03594 | 62.63 |
| K-means (k=64) | 43 | 0.07966 | 10.99 | 0.4061 | 0.8281 | -0.01996 | — | 16.46 |
| Kohonen (8x8) | 43 | 0.07548 | 11.22 | 0.2914 | 0.8906 | -0.01966 | — | 20.85 |
| PCA (d=16) | 43 | 0.05741 | 12.41 | 0.431 | — | — | 0.1665 | 14.28 |
| AutoEncoder (z=16) | 43 | 0.2142 | 6.691 | 0.004832 | — | — | 0.01948 | 73.25 |
| VAE (z=16) | 43 | 0.2143 | 6.69 | 0.004352 | — | — | -0.001083 | 71.45 |
| GAN (z=64, inversion diagnostique) | 43 | 0.2416 | 6.17 | 0.00519 | — | — | -0.1086 | 67.35 |

## Stockage mesuré

Code théorique : index idéalement compacté ou flottants float32. NPZ : octets réellement
sérialisés en mémoire, en-têtes inclus ; modèle ajusté complet et métadonnées inclus dans
le coût amorti sur ce lot d'évaluation. L'environnement logiciel est exclu.
Le GAN fournit une inversion approximative diagnostique, pas un codec natif.

| Méthode | Seed | Mode reconstruction | Bits théoriques/image | NPZ codes (octets) | NPZ modèle (octets) | Bits NPZ amortis/image |
|---|---|---|---|---|---|---|
| K-means (k=64) | 42 | native_codec | 6 | 266 | 60080 | 7543 |
| Kohonen (8x8) | 42 | native_codec | 6 | 269 | 159071 | 1.992e+04 |
| PCA (d=16) | 42 | native_codec | 512 | 4050 | 87668 | 1.146e+04 |
| AutoEncoder (z=16) | 42 | native_codec | 512 | 4060 | 777859 | 9.774e+04 |
| VAE (z=16) | 42 | native_codec | 512 | 4050 | 779639 | 9.796e+04 |
| GAN (z=64, inversion diagnostique) | 42 | bank_inversion_not_native_codec | 2048 | 3481 | 4043214 | 5.058e+05 |
| K-means (k=64) | 43 | native_codec | 6 | 269 | 57732 | 7250 |
| Kohonen (8x8) | 43 | native_codec | 6 | 269 | 158965 | 1.99e+04 |
| PCA (d=16) | 43 | native_codec | 512 | 4050 | 87668 | 1.146e+04 |
| AutoEncoder (z=16) | 43 | native_codec | 512 | 4045 | 777799 | 9.773e+04 |
| VAE (z=16) | 43 | native_codec | 512 | 4041 | 779520 | 9.795e+04 |
| GAN (z=64, inversion diagnostique) | 43 | bank_inversion_not_native_codec | 2048 | 1350 | 4043047 | 5.055e+05 |

Les diagnostics de diversité/voisinage et leurs tailles d'échantillon figurent dans le CSV.
Ils détectent des signaux grossiers ; ils ne prouvent ni qualité perceptuelle ni absence de mémorisation.
Avec une seule seed, aucune estimation de dispersion n'est disponible.
