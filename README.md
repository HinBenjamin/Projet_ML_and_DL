# 🧬 Prédire la résistance aux antibiotiques à partir du génome

Ce projet explore la prédiction du phénotype **sensible / résistant** d'une bactérie à un antibiotique à partir de son génotype. Le cas d'étude principal est la résistance d'*Escherichia coli* à la **ciprofloxacine**, à partir de données pangénomiques enrichies de mutations ponctuelles connues (SNP).

L'objectif n'est pas seulement d'obtenir un bon score, mais d'obtenir un modèle **utile, interprétable et honnêtement évalué** : sans fuite de clones entre entraînement et test, avec des métriques adaptées au déséquilibre de classes, et avec une lecture biologique des variables utilisées.

---

## 1. Le problème

- *E. coli* est un pathogène majeur, dont les résistances augmentent.
- La **ciprofloxacine** est une fluoroquinolone très utilisée.
- La résistance aux fluoroquinolones est en grande partie portée par des **mutations ponctuelles** dans des gènes connus (QRDR : `gyrA`, `gyrB`, `parC`, `parE`, et des régulateurs comme `marR`, `acrR`, `soxR`).

L'idée directrice : si ces mutations suffisent à expliquer une grande partie du phénotype, un modèle simple et interprétable devrait atteindre les performances d'un modèle complexe. C'est précisément ce que le projet vérifie.

---

## 2. Les données

Le dépôt s'appuie sur trois jeux de données complémentaires, tous construits par les scripts de `src/` et stockés dans `data/` (non versionné).

### a) *E. coli* – ciprofloxacine (jeu de référence)

| Élément | Valeur |
|---|---:|
| Isolats | 5 935 |
| Features | 20 929 |
| — gènes / clusters pangénomiques | 20 731 |
| — SNP QRDR et régulateurs | 198 |
| Résistants | 1 285 (21,7 %) |

Les 198 SNP sont extraits d'un VCF de variant calling pangénomique, filtrés sur les contigs d'intérêt et encodés en binaire (0 = allèle de référence, 1 = allèle alternatif détecté). Chaque variant reçoit un nom lisible, par exemple `SNP_gyrA_Ser83Leu`, `SNP_parC_Ser80Ile`.

### b) *E. coli* – 50 antibiotiques (extension multi-tâches)

| Élément | Valeur |
|---|---:|
| Génomes avec phénotypes | 6 983 |
| Antibiotiques | 50 |
| Observations binaires (R/S) | 65 843 |
| Génomes avec features | 5 935 |

Ce jeu permet d'étudier la résistance au-delà d'un seul antibiotique et de poser les bases d'un modèle multi-tâches. Une variante « non sensible » (Résistant ou Intermédiaire = 1) est également fournie, ainsi qu'un fichier long « une ligne par génome × antibiotique » (62 666 lignes) prêt pour l'apprentissage.

### c) Jeu multi-espèces (exploration)

80 cohortes (espèce × antibiotique), **11 espèces bactériennes**, 146 159 observations et 31 704 génomes uniques, disponibles en table longue et en matrice large. Ce jeu sert de base d'exploration pour généraliser la démarche à d'autres pathogènes.

---

## 3. La démarche

### 3.1 Un découpage qui évite la fuite de lignage

Un split aléatoire classique plaçait **95,8 %** des isolats de test dans un groupe MLST déjà présent dans l'entraînement. Le modèle pouvait alors apprendre des **marqueurs de clones** plutôt que la résistance elle-même.

Le découpage de référence est donc **groupé par MLST** :

- `StratifiedGroupKFold` à 6 folds, stratifié sur le phénotype ;
- 4 folds pour l'entraînement, 1 pour la validation, 1 pour le test ;
- aucun groupe MLST connu ne peut apparaître dans deux splits ;
- les isolats `unknown` sont traités comme des groupes individuels ;
- le couple (validation, test) est choisi pour se rapprocher de 70/15/15 et du taux de résistance global.

| Split | Isolats | Résistants | Taux R |
|---|---:|---:|---:|
| Train | 4 074 | 975 | 23,9 % |
| Validation | 932 | 156 | 16,7 % |
| Test | 929 | 154 | 16,6 % |

Le déséquilibre entre splits est inhérent aux grands groupes MLST. Il impose des métriques adaptées.

### 3.2 Métriques adaptées au déséquilibre

Avec ~17 % de résistants, l'accuracy est trompeuse : un modèle qui prédit « toujours sensible » atteint déjà **83,4 %** sur le test. Les métriques retenues sont donc :

- **PR-AUC** (precision-recall, adaptée aux classes rares) ;
- **recall résistant** (ne pas manquer un résistant) ;
- **F1 résistant** ;
- **balanced accuracy** ;
- ROC-AUC et calibration (Brier score) en complément.

### 3.3 Modèles comparés

Toutes les expériences utilisent les mêmes splits, les mêmes features et les mêmes métriques :

1. **Régression logistique** — baseline linéaire, interprétable ;
2. **Random Forest** — modèle à arbres, non linéaire ;
3. **XGBoost** — gradient boosting, référence sur données tabulaires ;
4. **MLP PyTorch** — réseau `20 929 → 64 → 32 → 1`, ReLU, dropout 0,5, early stopping sur le PR-AUC de validation, loss pondérée (`pos_weight`).

Le MLP est volontairement une baseline simple, sur la même représentation brute que XGBoost, afin de mesurer l'apport réel du deep learning sans sélection de features avantageuse.

---

## 4. Résultats principaux

### Comparaison sur le jeu de test

| Modèle | Features | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---|---:|---:|---:|---:|
| **XGBoost** | 20 929 | **0,951** | 0,916 | 0,862 | 0,937 |
| Régression logistique | 198 SNP | 0,941 | **0,942** | — | — |
| XGBoost | 198 SNP | 0,922 | 0,942 | — | — |
| XGBoost | 20 731 gènes seuls | 0,810 | 0,643 | — | — |
| MLP | 20 929 | 0,775 | 0,714 | 0,696 | 0,824 |

### Ce qu'il faut retenir

- **Le signal est presque entièrement dans les 198 SNP.** Une régression logistique sur ces seules variables atteint un PR-AUC de 0,941, quasi identique à XGBoost sur 20 929 features.
- **Le pangénome seul est bruité** : il contient des marqueurs de lignage et généralise moins bien (PR-AUC 0,810).
- **Le deep learning n'apporte pas de gain ici.** Le MLP plafonne à 0,775 de PR-AUC, soit **17,6 points de moins** que XGBoost. C'est un résultat négatif utile : sur une matrice tabulaire de petite taille, le ML classique reste le bon outil.
- **Les variables importantes sont biologiquement connues** : `SNP_parC_Ser80Ile`, `SNP_gyrA_Asp87Asn`, `SNP_gyrA_Ser83Leu` dominent les importances. Le modèle s'appuie donc sur les mécanismes réels de résistance, ce qui est rassurant.
- **Les probabilités de XGBoost sont globalement calibrées** (Brier = 0,020), ce qui rend le choix d'un seuil de décision interprétable.

> Le bon modèle n'est pas le plus complexe : c'est celui qui capture le signal biologique, généralise sur des lignages non vus et reste interprétable.

---

## 5. Structure du dépôt

```text
Projet_ML_DL/
├── data/                         # Données (non versionnées)
│   ├── raw/                      # Sources brutes : génomes, VCF, phénotypes
│   ├── processed/                # Jeux construits (Parquet, CSV, manifestes)
│   ├── splits_snp_3way/          # ⭐ Split de référence train/val/test
│   ├── splits_snp/               # Ancien split 2-way avec SNP
│   ├── splits/                   # Split 2-way pangénome d'origine
│   └── results/                  # Sorties de modèles
├── src/
│   ├── add_qrdr_snps.py                        # Extraction des 198 SNP QRDR
│   ├── make_3way_splits_snp.py                 # Split groupé par MLST (référence)
│   ├── make_splits_snp.py                      # Ancien split 2-way
│   ├── make_ecoli_all_antibiotics_dataset.py   # Jeu E. coli × 50 antibiotiques
│   ├── make_ecoli_long_features_dataset.py     # Table longue génome × antibiotique
│   ├── export_ecoli_long_features_csv.py       # Export CSV de cette table
│   ├── make_multispecies_antibiotic_dataset.py # Jeu multi-espèces (80 cohortes)
│   └── data_loader.py                          # Chargement des jeux préparés
├── slides/                       # Présentation (Quarto / Reveal.js)
├── notebooks/                    # Espace d'exploration par membre
└── README.md
```

Chaque script de construction écrit un **manifeste JSON** (comptages, contrôles, empreintes SHA-256) à côté de ses sorties, pour garantir la traçabilité des données.

---

## 6. Démarrage rapide

### Dépendances

```bash
pip install pandas numpy pyarrow scikit-learn xgboost torch
```

### Charger le jeu de référence

```python
from src.data_loader import load_train, load_validation, load_test

X_train, y_train, meta_train = load_train(dataset="snp_3way")
X_val,   y_val,   meta_val   = load_validation(dataset="snp_3way")
X_test,  y_test,  meta_test  = load_test(dataset="snp_3way")

print(X_train.shape, y_train.mean())
```

### Reconstruire les données (si `data/` est vide)

Les sources brutes doivent être placées dans `data/raw/`, puis :

```bash
# 1. Ajouter les 198 SNP QRDR au jeu pangénomique
python src/add_qrdr_snps.py \
    --vcf data/raw/Ecoli_AMR_GenotypePhenotype/vcf_files/qrdr/qrdr_non_silent.vcf \
    --csv data/processed/ecoli_ciprofloxacin.csv \
    --out-csv data/processed/ecoli_ciprofloxacin_snp.csv \
    --out-parquet data/processed/ecoli_ciprofloxacin_snp.parquet

# 2. Construire le split 3-way groupé par MLST
python src/make_3way_splits_snp.py

# 3. Construire le jeu E. coli 50 antibiotiques
python src/make_ecoli_all_antibiotics_dataset.py

# 4. Construire la table longue génome × antibiotique
python src/make_ecoli_long_features_dataset.py

# 5. Construire le jeu multi-espèces
python src/make_multispecies_antibiotic_dataset.py
```

---

## 7. Limites et perspectives

**Limites actuelles**

- Test de petite taille (929 isolats, 154 résistants) → intervalles de confiance larges.
- Split groupé par MLST, mais pas par phylogénie fine.
- Pas de validation externe sur une autre cohorte.
- Le seuil de décision optimal a été choisi et évalué sur la validation → estimation optimiste.
- Calibration explorée mais non corrigée (Platt / isotonic).
- Une seule architecture de deep learning, un seul seed.
- Les features pangénomiques sont partiellement non annotées : risque de marqueurs de lignage.

**Pistes**

- Validation croisée groupée à 5 folds, plusieurs seeds, intervalles de confiance par bootstrap.
- Validation externe et évaluation du seuil figé sur le test.
- Correction de la calibration et intégration dans la décision.
- Analyse d'erreurs par lignage et confrontation à la littérature.
- Enrichissement biologique : pompes d'efflux, plasmides, gènes acquis.
- Sélection de features SNP + gènes, MLP sur les 198 SNP, autoencodeur.

---

## 8. Équipe

- **Benjamin HIN**
- **Alexandre**
- **Baptiste**

Projet réalisé dans le cadre d'un cours de Machine Learning / Deep Learning.
