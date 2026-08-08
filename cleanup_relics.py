"""Range les reliquats du développement dans _to_delete/ — à lancer UNE fois.

Ce script ne SUPPRIME rien : il déplace dans _to_delete/ (à la racine du
projet) les fichiers identifiés comme reliquats par l'audit final, pour que
tu puisses vérifier puis supprimer le dossier toi-même.

    python cleanup_relics.py            # affiche ce qui serait déplacé
    python cleanup_relics.py --apply    # déplace réellement

Les reliquats et pourquoi :

1. src/vae/models/autoencoder_quickdraw_z2_dense_e100_n25715_latent_sigmoid.npz
   Un vecteur de poids d'AUTOENCODEUR égaré dans le registre VAE : le
   sélectionner comme VAE dans l'app lève « Poids incompatibles » (l'encodeur
   VAE a une tête z_log_var en plus, donc un vecteur plus long). Sa copie
   légitime existe déjà dans src/autoencoder/models/.

2. src/autoencoder/auto_encodeur_experiment.md
   Doublon exact de documentation/autoencoder/auto_encodeur_experiment.md —
   la documentation vit dans documentation/.

3. src/utils/autoencoder/ (compression.py, dim_reduction.py, generation.py)
   Trois fichiers VIDES (0 octet) — un squelette de découpage à la PCA jamais
   rempli : le vrai code de l'autoencodeur vit dans src/autoencoder/utils/.
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRASH = ROOT / "_to_delete"

RELICS = [
    "src/vae/models/autoencoder_quickdraw_z2_dense_e100_n25715_latent_sigmoid.npz",
    "src/autoencoder/auto_encodeur_experiment.md",
    "src/utils/autoencoder",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Déplace réellement (sinon, affichage seul).")
    args = parser.parse_args()

    moved = 0
    for rel in RELICS:
        src = ROOT / rel
        if not src.exists():
            print(f"  (absent, déjà rangé ?)  {rel}")
            continue
        dst = TRASH / rel
        print(f"  {rel}  ->  _to_delete/{rel}")
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
        moved += 1

    if not args.apply:
        print("\nRien n'a été déplacé (mode aperçu). Relance avec --apply.")
    elif moved:
        print(f"\n{moved} reliquat(s) rangé(s) dans {TRASH} — vérifie, puis "
              f"supprime ce dossier.")


if __name__ == "__main__":
    main()
