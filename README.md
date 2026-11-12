# 🧬 Prédire la résistance aux antibiotiques à partir du génome

Ce projet prédit si une souche d'*Escherichia coli* est **sensible ou résistante à la ciprofloxacine** à partir de son génome.

Le jeu de référence combine le pangénome et **198 SNP connus** (QRDR et régulateurs). L'objectif n'est pas seulement d'obtenir un bon score, mais un modèle **utile, interprétable et évalué sans fuite de clones**.

**À retenir**

- **XGBoost** atteint un PR-AUC de **0,951** sur le test, avec un F1 de **0,940** et un recall de **0,909** au seuil 0,842.
- Le **MLP sur les 20 929 features brutes** plafonne à **0,730** : le deep learning n'apporte pas de gain sur cette représentation.
- Le **MLP sur 50 features sélectionnées par SHAP** remonte à **0,954**, au niveau de XGBoost : la représentation compte plus que l'architecture.

---

## 1. Le problème

- *E. coli* est un pathogène majeur dont les résistances augmentent.
- La **ciprofloxacine** est une fluoroquinolone très utilisée.
- La résistance aux fluoroquinolones est en grande partie portée par des **mutations ponctuelles** dans des gènes connus : `gyrA`, `gyrB`, `parC`, `parE`, ainsi que des régulateurs comme `marR`, `acrR` et `soxR`.

Hypothèse du projet : si ces mutations expliquent l'essentiel du phénotype, un modèle simple et interprétable devrait atteindre les performances d'un modèle complexe.

---

## 2. Les données

Le jeu de référence contient **5 935 isolats** et **20 929 features** (pangénome + 198 SNP), dont **1 285 résistants (21,7 %)**.

Les **198 SNP** sont extraits d'un VCF pangénomique, filtrés sur les gènes d'intérêt puis encodés en binaire :

- `0` = allèle de référence ou aucun variant détecté ;
- `1` = allèle alternatif détecté.

Chaque variant porte un nom lisible, par exemple `SNP_gyrA_Ser83Leu` ou `SNP_parC_Ser80Ile`.

### Split train / validation / test

Le split est **groupé par MLST** : un même groupe n'apparaît que dans un seul split. Cela évite au modèle d'apprendre des marqueurs de clones plutôt que la résistance.

| Split | Isolats | Résistants | Taux R |
|---|---:|---:|---:|
| Train | 4 074 | 975 | 23,9 % |
| Validation | 932 | 156 | 16,7 % |
| Test | 929 | 154 | 16,6 % |

Le déséquilibre entre splits est inhérent aux grands groupes MLST.

> Seul le jeu ciprofloxacine sert aux résultats ci-dessous. Le jeu 50 antibiotiques est préparé mais n'est pas évalué ici.

---

## 3. Protocole

### 3.1 Des métriques adaptées au déséquilibre

Avec environ 17 % de résistants, l'accuracy est trompeuse : un modèle qui prédit **toujours sensible** atteint déjà **83,4 %** sur le test.

Les métriques principales sont donc :

- **PR-AUC** : adaptée aux classes rares et indépendante du seuil ;
- **recall résistant** : ne pas manquer un résistant ;
- **F1 résistant** ;
- **balanced accuracy**.

### 3.2 Modèles comparés

Tous les modèles utilisent les mêmes splits et les mêmes métriques :

1. **Régression logistique** : baseline linéaire, interprétable ;
2. **Random Forest** : modèle à arbres ;
3. **XGBoost** : gradient boosting, référence sur données tabulaires ;
4. **MLP PyTorch** : deux variantes.
   - **brut** : `20 929 → 512 → 128 → 32 → 1`, BatchNorm, ReLU, dropout 0,3 / 0,2 ;
   - **top-50** : `50 → 64 → 16 → 1`, dropout 0,5.

Le MLP est entraîné avec AdamW, une loss pondérée (`pos_weight ≈ 3,18`) et un early stopping sur le **PR-AUC de validation**.

Les modèles sont évalués sur la validation au **seuil 0,5**. Seuls XGBoost et les deux MLP passent ensuite sur le test : le seuil y est choisi sur la validation par F1 maximal, puis **figé** et appliqué au test.

---

## 4. Résultats

> Tous les chiffres de cette section proviennent de `notebooks/exploration_benjamin.ipynb`.

### 4.1 Validation, seuil 0,5

| Modèle | Features | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---|---:|---:|---:|---:|
| **XGBoost** | 20 929 | **0,969** | 0,917 | 0,923 | 0,951 |
| MLP top-50 SHAP | 50 | 0,964 | 0,917 | 0,902 | 0,947 |
| Random Forest | 20 929 | 0,803 | 0,385 | 0,556 | 0,692 |
| MLP brut | 20 929 | 0,685 | 0,795 | 0,537 | 0,780 |
| Régression logistique | 20 929 | 0,554 | 0,314 | 0,405 | 0,633 |

Le **PR-AUC** ne dépend pas du seuil ; en revanche, le recall, le F1 et la balanced accuracy en dépendent.

### 4.2 Test, seuil optimisé sur la validation

| Modèle | Features | Seuil | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | 20 929 | 0,842 | 0,951 | 0,909 | 0,940 | 0,952 |
| MLP brut | 20 929 | 0,776 | 0,730 | 0,714 | 0,701 | 0,825 |
| MLP top-50 SHAP | 50 | 0,960 | **0,954** | 0,909 | 0,943 | 0,953 |

### 4.3 Ce qu'il faut retenir

- **XGBoost est très performant.** Au seuil 0,842, il ne produit que **4 faux positifs** et **14 faux négatifs** sur les 929 isolats de test.
- **Le deep learning n'apporte pas de gain sur la matrice brute.** Le MLP brut plafonne à **0,730** de PR-AUC test, soit **22 points de moins** que XGBoost. Son meilleur epoch est le premier, signe d'un sur-apprentissage immédiat.
- **Avec une représentation compacte, le MLP devient compétitif.** Sur les **50 features** sélectionnées par SHAP avec XGBoost (sur un échantillon de 1 000 lignes du train), il atteint **0,954**, quasi identique à XGBoost. Mais la sélection est elle-même supervisée par XGBoost : cela ne prouve pas une supériorité du deep learning.
- **L'interprétabilité est explorée dans le notebook** via SHAP (beeswarm, barres, waterfall) et une analyse des erreurs.
- La courbe de calibration de XGBoost est tracée sur la validation, mais la calibration n'est pas corrigée.

> Le bon modèle n'est pas le plus complexe : c'est celui qui capture le signal biologique, généralise sur des lignages non vus et reste interprétable.

---

## 5. Structure du dépôt

```text
Projet_ML_DL/
├── data/                         # Données (non versionnées)
│   ├── raw/                      # Sources brutes : génomes, VCF, phénotypes
│   ├── processed/                # Jeux construits (Parquet, CSV, manifestes)
│   ├── splits_snp_3way/          # ⭐ Split de référence train / validation / test
│   ├── splits_all_antibiotics/   # Split train / validation / test du jeu 50 antibiotiques
│   └── results/                  # Sorties de modèles
├── src/
│   ├── add_qrdr_snps.py                        # Extraction des 198 SNP
│   ├── make_3way_splits_snp.py                 # Split groupé par MLST
│   ├── make_ecoli_all_antibiotics_dataset.py   # Jeu E. coli × 50 antibiotiques
│   ├── make_ecoli_long_features_dataset.py     # Table longue génome × antibiotique
│   ├── make_multispecies_antibiotic_dataset.py # Jeu multi-espèces
│   └── data_loader.py                          # Chargement des jeux préparés
├── notebooks/                    # Exploration et modélisation
├── slides/                       # Présentation Quarto / Reveal.js
└── README.md
```

Chaque script de construction écrit un **manifeste JSON** (comptages, contrôles, empreintes SHA-256) à côté de ses sorties pour garantir la traçabilité.

---

## 6. Démarrage rapide

### Dépendances

```bash
pip install pandas numpy pyarrow scikit-learn xgboost torch shap matplotlib seaborn jupyter
```

### Charger le jeu de référence

```python
from src.data_loader import load_train, load_validation, load_test

X_train, y_train, meta_train = load_train(dataset="snp_3way")
X_val, y_val, meta_val = load_validation(dataset="snp_3way")
X_test, y_test, meta_test = load_test(dataset="snp_3way")

print(X_train.shape, y_train.mean())
```

Les modèles, les métriques, SHAP et le MLP sont dans `notebooks/exploration_benjamin.ipynb`.

### Reconstruire les données (si `data/` est vide)

Placer les sources brutes dans `data/raw/`, puis :

```bash
# 1. Ajouter les 198 SNP au jeu pangénomique
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

# 5. Construire le split 3-way du jeu 50 antibiotiques
python src/make_3way_splits_all_antibiotics.py
```

---

## 7. Limites

- **Test petit** : 929 isolats, 154 résistants, donc des intervalles de confiance larges.
- **Split groupé par MLST**, mais pas par phylogénie fine.
- **Pas de validation externe** sur une autre cohorte.
- **Seuil choisi sur la validation**, puis appliqué au test : le test reste indépendant, mais le seuil n'est pas validé sur un second jeu.
- **Calibration explorée mais non corrigée.**
- **Deep learning** : un seul entraînement, sans graine fixée pour PyTorch ; le top-50 dépend d'une sélection SHAP issue de XGBoost.
- **Features pangénomiques partiellement non annotées** : risque de marqueurs de lignage.

### Pistes

- Validation croisée groupée, plusieurs seeds, intervalles de confiance par bootstrap.
- Validation externe et validation croisée imbriquée pour le seuil et la sélection de features.
- Correction de la calibration et intégration dans la décision.
- Analyse d'erreurs par lignage et confrontation à la littérature.
- Enrichissement biologique : pompes d'efflux, plasmides, gènes acquis.
- MLP sur les 198 SNP, sélection de features sans XGBoost, autoencodeur.

---

## 8. Équipe

- **Benjamin HIN**
- **Alexandre**
- **Baptiste**

Projet réalisé dans le cadre d'un cours de Machine Learning / Deep Learning.
