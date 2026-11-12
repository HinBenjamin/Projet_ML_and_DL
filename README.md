# 🚀 Projet Machine Learning / Deep Learning

## 👥 Membres de l'équipe
- **Benjamin HIN**
- **Alexandre**
- **Baptiste**

---

## 📁 Structure du projet

```text
Projet_ML_DL/
├── data/                      # Données (non versionné sur Git)
│   ├── raw/                   # Archive Zenodo / Arcadia (génomes, VCF, phénotypes)
│   ├── processed/             # Jeu de base, jeu enrichi (198 SNP), E. coli 50 antibiotiques et multi-espèces
│   ├── splits/                # Split 2-way d'origine (pangénome)
│   ├── splits_snp/            # Split 2-way avec SNP
│   ├── splits_snp_3way/       # ⭐ Split de référence train/validation/test
│   └── results/               # Sorties de modèles
├── src/                       # Code de préparation des données
│   ├── add_qrdr_snps.py       # Extraction des 198 SNP QRDR
│   ├── make_3way_splits_snp.py# Split groupé par MLST (référence)
│   ├── make_splits_snp.py     # Ancien split 2-way
│   ├── make_ecoli_all_antibiotics_dataset.py # Jeu E. coli 50 antibiotiques (features + labels)
│   ├── make_ecoli_long_features_dataset.py   # Fichier unique génome × antibiotique (features + résistance)
│   ├── export_ecoli_long_features_csv.py     # Export CSV du fichier unique
│   ├── make_multispecies_antibiotic_dataset.py # Jeu multi-espèces (80 cohortes, phénotypes)
│   └── data_loader.py         # load_train / load_validation / load_test
├── notebooks/                 # Un notebook de travail par membre
│   ├── exploration_alexandre.ipynb
│   ├── exploration_baptiste.ipynb
│   └── exploration_benjamin.ipynb
├── docs/                      # Documentation et rapport
│   ├── synthese_demarche.tex   # ⭐ Résumé de la démarche et de son articulation
│   ├── synthese_demarche.pdf   # Synthèse compilée
│   ├── rapport_demarche.tex   # Rapport complet
│   ├── donnees_snp.md         # Documentation du jeu enrichi
│   ├── split_3way.md          # Documentation du découpage
│   ├── resultats/             # CSV de résultats
│   └── archive/               # Anciennes versions du rapport
├── slides/                    # Présentation Reveal.js (Quarto)
├── requirements.txt           # Dépendances Python partagées
├── .gitignore                 # Ignore data/, artefacts LaTeX et fichiers système
└── README.md                  # Documentation du projet
```

---

## 📚 Documentation

- **[Synthèse de la démarche](docs/synthese_demarche.pdf)** ([source LaTeX](docs/synthese_demarche.tex)) — à lire en premier : objectif, étapes, articulation, chiffres clés, état d'avancement.
- **[Rapport complet](docs/rapport_demarche.tex)** — formules, références, justifications détaillées, glossaire (annexe C) et notations.
- **[Jeu de données enrichi](docs/donnees_snp.md)** — construction des 198 SNP QRDR.
- **[Découpage](docs/split_3way.md)** — train/validation/test groupé par MLST.
- **[Jeux de données antibiotiques & protocole multi-tâches](docs/protocole_multilabel.md)** — construction du jeu *E. coli* 50 antibiotiques (6 983 génomes, 67 060 labels, features + split MLST) et du jeu multi-espèces (80 cohortes, 11 espèces), formats Parquet, schémas, limites et démarche multi-tâches.

---

## ⚡ Démarrage rapide

```bash
pip install -r requirements.txt

# Charger le split de référence
python -c "from src.data_loader import load_train, load_validation, load_test; \
X, y, m = load_train(dataset='snp_3way'); print(X.shape, y.mean())"

# Construire le jeu E. coli 50 antibiotiques (features + labels)
python src/make_ecoli_all_antibiotics_dataset.py

# Construire le fichier unique (une ligne par génome × antibiotique)
python src/make_ecoli_long_features_dataset.py

# L'exporter en CSV (≈2,5 Go)
python src/export_ecoli_long_features_csv.py

# Construire le jeu multi-espèces (80 cohortes, phénotypes seuls)
python src/make_multispecies_antibiotic_dataset.py
```

### Recompiler la synthèse

```bash
cd docs && pdflatex synthese_demarche.tex && pdflatex synthese_demarche.tex
```

---

## 🤝 Organisation de l'équipe (Notebooks)

- Chaque membre travaille exclusivement dans son propre notebook (`exploration_<nom>.ipynb`).
- Avant de pousser vos modifications sur Git, pensez à vider les cellules de sortie du notebook (*Clear All Outputs*) pour garder un historique propre et léger.
