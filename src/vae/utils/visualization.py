"""Figures du VAE côté app.

Les vues sont celles de l'autoencodeur — reconstructions côte à côte, nuage
latent, distribution des classes, traversée du décodeur, grille du manifold 2D,
génération, courbe de loss — et sont réutilisées telles quelles de
src/autoencoder/utils/visualization.py : un VAE se regarde exactement comme un
autoencodeur, seul le récit change (voir algo.py).

S'y ajoute UNE vue propre au VAE : la superposition des trois courbes de loss
(totale, reconstruction, KL), qui montre l'arbitrage que la KL divergence impose.
"""

import matplotlib.pyplot as plt
import numpy as np

# Réutilisées telles quelles : les figures sont identiques à celles de
# l'autoencodeur, on ne les redéfinit pas.
from autoencoder.utils.visualization import (  # noqa: F401
    make_manifold_canvas,
    plot_dominant_distribution,
    plot_generated_grid,
    plot_latent_panels,
    plot_loss_curve,
    plot_manifold,
    plot_traversal_grid,
    show_reconstructions,
)


def plot_vae_losses(reconstruction, kl, kl_weight, suptitle=""):
    """Les trois courbes de loss du VAE au fil des époques.

    reconstruction et kl sont les valeurs BRUTES par époque (MSE par pixel et KL
    sommée sur les dimensions). La loss totale est reconstruction + kl_weight·kl —
    ce que la descente minimise, et où l'on voit l'arbitrage : baisser la KL
    (latent plus proche de N(0,1)) coûte souvent de la reconstruction.
    """
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    kl = np.asarray(kl, dtype=np.float64)
    total = reconstruction + kl_weight * kl
    epochs = range(1, len(reconstruction) + 1)

    fig, (ax_total, ax_kl) = plt.subplots(1, 2, figsize=(12, 5))

    ax_total.plot(epochs, total, marker="o", markersize=3, label="Totale")
    ax_total.plot(epochs, reconstruction, marker="o", markersize=3,
                  label="Reconstruction (MSE/pixel)")
    ax_total.set_xlabel("Époque")
    ax_total.set_ylabel("Loss")
    ax_total.set_title(f"Totale = reconstruction + {kl_weight:g} · KL")
    ax_total.legend()
    ax_total.grid(True, alpha=0.3)

    ax_kl.plot(epochs, kl, marker="o", markersize=3, color="tab:red")
    ax_kl.set_xlabel("Époque")
    ax_kl.set_ylabel("KL divergence (sommée sur les dimensions)")
    ax_kl.set_title("KL(N(μ, σ²) ‖ N(0, 1))")
    ax_kl.grid(True, alpha=0.3)

    fig.suptitle(suptitle or "VAE — évolution des losses", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    return fig
