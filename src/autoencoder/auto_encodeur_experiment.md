# Rapport d'expérimentations — AutoEncoder simple et Variational AutoEncoder (VAE)

## 0. Objectif général

Ce document sert à comparer expérimentalement plusieurs configurations d'un **AutoEncoder simple** et d'un **Variational AutoEncoder (VAE)**.

Les expérimentations portent pour le moment sur **trois classes uniquement** :

- Classe 1 : `chat`
- Classe 2 : `pomme`
- Classe 3 : `voiture`

L'objectif final est d'identifier, pour chaque algorithme, **la meilleure configuration pour chacune des trois tâches suivantes** :

1. **Compression / décompression**
2. **Visualisation de l'espace latent**
3. **Génération d'images**

Une expérimentation correspond à la variation d'**un seul hyperparamètre à la fois**. Tous les autres paramètres restent constants afin d'isoler l'effet réel de l'hyperparamètre étudié.

### Expérience `1` — Variation de `activation latent`

AutoencoderConfig(
    architecture="dense",
    latent_dim=2,
    hidden_dims=(128, 32),

    hidden_activation="relu",
    latent_activation="linear",
    output_activation="sigmoid",

    conv_filters=(32, 64),
    conv_strides=(2, 2),
    kernel_size=3,

    kernel_initializer="glorot_uniform",

    learning_rate=1e-3,
    batch_size=256,
    epochs=200,
    optimizer_name="adam",
    loss_name="mse",

    normalization="zero_one",
    seed=42,
)

**Objectif**

> Étudier l'effet de `activation latent` sur la compression / décompression, la visualisation de l'espace latent et la génération d'images.

**Hypothèse**

> `[Hypothèse avant l'expérience]`

**Variable modifiée**

| Configuration | Valeur testée |
|---|---------------|
| Référence | `linéaire`    |
| Variante 1 | `tanh`        |
| Variante 2 | `relu`        |
| Variante 3 | `sigmoid`     |

# PARTIE I — AUTOENCODER SIMPLE

# lineaire 
## 1.3 Évaluation des trois tâches

### Tâche 1 — Compression / décompression

![img_3.png](img_3.png)[IMAGE — Originales vs reconstructions]

### Tâche 2 — Visualisation de l'espace latent

![img_4.png](img_4.png)[IMAGE — Projection / espace latent avec les trois classes]

![img_6.png](img_6.png)

![img_7.png](img_7.png)

**Questions à traiter :**

- Les trois classes forment-elles des groupes distincts ?
- Certaines classes se chevauchent-elles ?
- L'espace latent semble-t-il continu et structuré ?
- La configuration facilite-t-elle l'interprétation visuelle ?

---

### Tâche 3 — Génération d'images

![img_5.png](img_5.png)


# Activation latent tanh
### A. Effet sur la compression / décompression

`[Analyse]`

![img_8.png](img_8.png)

### B. Effet sur la visualisation de l'espace latent

`[Analyse]`

![img_9.png](img_9.png)
![img_10.png](img_10.png)
![img_11.png](img_11.png)

### C. Effet sur la génération d'images

`[Analyse]`

![img_12.png](img_12.png)


# Activation latent relu
# Loss
![img_13.png](img_13.png)

### A. Effet sur la compression / décompression

`[Analyse]`

![img_14.png](img_14.png)

### B. Effet sur la visualisation de l'espace latent

`[Analyse]`

![img_15.png](img_15.png)
![img_16.png](img_16.png)
![img_17.png](img_17.png)

### C. Effet sur la génération d'images

`[Analyse]`

![img_18.png](img_18.png)


# Activation latent sigmoid
# Loss

![img_21.png](img_21.png)

### A. Effet sur la compression / décompression

`[Analyse]`

![img_20.png](img_20.png)

### B. Effet sur la visualisation de l'espace latent

`[Analyse]`

![img_22.png](img_22.png)
![img_23.png](img_23.png)
![img_24.png](img_24.png)

### C. Effet sur la génération d'images

`[Analyse]`

![img_26.png](img_26.png)

### Conclusion de l'expérience

La fonction d’activation de la couche latente influence directement la forme et la liberté de l’espace latent. Pour nos trois tâches :

Compression / décompression : linear est la plus adaptée car elle conserve davantage d’information sans contraindre les valeurs latentes.
Visualisation : linear offre généralement une représentation plus libre, tandis que tanh peut produire un espace plus compact et structuré.
Génération : linear est préférable car elle permet un espace latent plus flexible et moins limité que ReLU, tanh ou sigmoid.

Conclusion générale : parmi les activations testées, linear constitue le meilleur compromis global pour les trois tâches, tandis que les activations bornées ou contraintes peuvent limiter la richesse de la représentation latente.


# 3. Expérimentations — AutoEncoder simple

## AE-EXP-01 — Dimension latente `latent_dim`

4, 32, 64

**Objectif :** mesurer l'effet du niveau de compression sur les trois tâches.

**Hypothèse :**

- faible `latent_dim` → forte compression mais perte d'information possible ;
- grand `latent_dim` → meilleures reconstructions mais compression plus faible ;
- `latent_dim=2` → particulièrement adapté à la visualisation directe.

### Résultats

# 4
# Loss

![img_27.png](img_27.png)

### A. Compression / décompression

`[Analyse]`

![img_28.png](img_28.png)

### B. Visualisation de l'espace latent

`[Analyse]`

![img_29.png](img_29.png)
![img_30.png](img_30.png)![img_31.png](img_31.png)

### C. Génération

`[Analyse]`

![img_32.png](img_32.png)

# 32
# Loss

![img_33.png](img_33.png)

### A. Compression / décompression

`[Analyse]`

![img_34.png](img_34.png)

### B. Visualisation de l'espace latent

`[Analyse]`

![img_35.png](img_35.png)
![img_36.png](img_36.png)![img_37.png](img_37.png)

### C. Génération

`[Analyse]`

![img_38.png](img_38.png)

# 64
# Loss

![img_39.png](img_39.png)

### A. Compression / décompression

`[Analyse]`

![img_40.png](img_40.png)

### B. Visualisation de l'espace latent

`[Analyse]`

![img_41.png](img_41.png)
![img_42.png](img_42.png)![img_43.png](img_43.png)

### C. Génération

`[Analyse]`

![img_44.png](img_44.png)

### Conclusion

| Tâche | Meilleur `latent_dim` | Pourquoi ? |
|---|---|---|
| Compression / décompression | `[ ]` | `[ ]` |
| Visualisation | `[ ]` | `[ ]` |
| Génération | `[ ]` | `[ ]` |

---

## `hidden_dims`

### Valeurs testées

| Configuration | `hidden_dims` |
|`(128, 32)` |
|`(512, 256, 64)` |

**Objectif :** mesurer l'effet de la capacité du réseau.

### Résultats

# (128, 32)
# loss

![img_45.png](img_45.png)

### A. Compression / décompression

![img_46.png](img_46.png)

### B. Visualisation latente

![img_47.png](img_47.png)
![img_48.png](img_48.png)![img_49.png](img_49.png)

### C. Génération

![img_50.png](img_50.png)


# (512, 256, 64)
# loss

![img_51.png](img_51.png)

### A. Compression / décompression

![img_52.png](img_52.png)

### B. Visualisation latente

![img_53.png](img_53.png)
![img_54.png](img_54.png)![img_55.png](img_55.png)

### C. Génération

![img_56.png](img_56.png)

### Conclusion

| Tâche | Meilleure architecture | Pourquoi ? |
|---|---|---|
| Compression / décompression | `[ ]` | `[ ]` |
| Visualisation | `[ ]` | `[ ]` |
| Génération | `[ ]` | `[ ]` |

---

## `hidden_activation`

### Valeurs testées

| Configuration | Activation |
| AE-C2 | `tanh` |
| AE-C3 | `leaky_relu` |

# tanh
# Loss

![img_63.png](img_63.png)

### A. Compression / décompression

![img_64.png](img_64.png)

### B. Visualisation latente

![img_65.png](img_65.png)
![img_66.png](img_66.png)![img_67.png](img_67.png)


### C. Génération

![img_68.png](img_68.png)

# leaky_relu
# Loss

![img_57.png](img_57.png)

### A. Compression / décompression

![img_58.png](img_58.png)

### B. Visualisation latente

![img_59.png](img_59.png)
![img_60.png](img_60.png)![img_61.png](img_61.png)

### C. Génération

![img_62.png](img_62.png)

### Conclusion