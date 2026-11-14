# 🧬 Prédire la résistance aux antibiotiques à partir du génome

Ce projet prédit le phénotype **sensible / résistant** d'*Escherichia coli* à un antibiotique à partir de son génotype. Deux jeux de données sont étudiés :

1. **Ciprofloxacine** — cas d'école, toujours modélisé sur le pangénome complet enrichi de 198 SNP QRDR (jamais sur les SNP seuls) ;
2. **48 antibiotiques** — extension multi-phénotypes, avec sélection des antibiotiques exploitables puis exploration détaillée d'un antibiotique.

L'objectif n'est pas seulement d'obtenir un bon score, mais un modèle **utile, interprétable et honnêtement évalué** : split groupé pour éviter la fuite de clones, métriques adaptées au déséquilibre, **régularisation Lasso (L1)** pour sélectionner les variables, seuil choisi sur la validation, et lecture biologique des variables.

> **Source de vérité.** Les résultats de modélisation de ce README sont extraits des sorties des notebooks `notebooks/ciprofloxacine_exploration.ipynb` et `notebooks/all_antibiotic.ipynb` ; les comptages de données ont été vérifiés directement dans les fichiers Parquet. La présentation `slides/presentation_antibiotiques.qmd` est alignée sur le notebook 48 antibiotiques. En revanche, `slides/presentation_ml.qmd` contient encore des chiffres cipro d'une version antérieure : en cas de doute, les notebooks font foi.

---

## 1. En bref

| | Ciprofloxacine | 48 antibiotiques |
|---|---:|---|
| Génomes | 5 935 | 6 949 |
| Features | 20 929 | 34 744 |
| Dont SNP QRDR | 198 | inclus dans les features |
| Phénotypes | 1 (ciprofloxacine) | 48 |
| Résistants | 1 285 (21,7 %) | 11 732 labels R (17,9 % des labels observés) |
| Notebook | `ciprofloxacine_exploration.ipynb` | `all_antibiotic.ipynb` |

**Résultats marquants :**

- Sur la ciprofloxacine, **XGBoost** atteint un **PR-AUC test de 0,951** (recall R 0,909, F1 R 0,940) avec un seuil choisi sur la validation.
- Un **MLP complet** sur les 20 929 features est très en retrait (**PR-AUC test 0,730**).
- Un **MLP sur les 50 features les plus importantes selon SHAP** remonte à **0,954** de PR-AUC test, soit un niveau comparable à XGBoost — mais il dépend de la sélection de variables supervisée par XGBoost.
- La **régression logistique L1 (Lasso)**, entraînée sur les **20 929 features**, atteint **0,973 de PR-AUC en validation** et sélectionne **68 variables** ; elle retrouve en tête les trois mutations QRDR `parC_Ser80Ile`, `gyrA_Ser83Leu` et `gyrA_Asp87Asn`.
- Sur la **céfuroxime**, les écarts entre modèles sont faibles : XGBoost a le meilleur PR-AUC test (0,665), la régression L1 le meilleur F1 R (0,672) et la meilleure balanced accuracy (0,772).

> **Choix méthodologique assumé.** Aucun modèle n'est entraîné sur les **198 SNP seuls** : toutes les expériences ciprofloxacine partent de la matrice complète (20 731 gènes/clusters + 198 SNP). La réduction se fait par **régularisation Lasso (L1)**, qui annule les coefficients des variables inutiles et conserve 68 features. Les chiffres « 198 SNP seuls » de `slides/presentation_ml.qmd` proviennent d'une version antérieure et ne correspondent pas au protocole actuel ; ils ne sont pas repris ici.

---

## 2. Le problème

- *E. coli* est un pathogène majeur dont les résistances augmentent.
- La **ciprofloxacine** est une fluoroquinolone très utilisée.
- La résistance aux fluoroquinolones est en grande partie portée par des **mutations ponctuelles** dans des gènes connus (QRDR : `gyrA`, `gyrB`, `parC`, `parE`, et des régulateurs comme `marR`, `acrR`, `soxR`).

L'idée directrice : si ces mutations suffisent à expliquer une grande partie du phénotype, un modèle simple et interprétable devrait approcher les performances d'un modèle complexe. Le projet vérifie cette hypothèse avec une **régression logistique L1 (Lasso)** entraînée sur la matrice complète : elle peut s'appuyer sur les SNP connus sans être restreinte à eux, et montre aussi les limites de l'exercice.

---

## 3. Les données

Les fichiers préparés sont dans `data/processed/`. Les sources brutes sont dans `data/raw/` (non versionnées, ~8 Go).

### 3.1 *E. coli* — ciprofloxacine (jeu de référence)

Fichier : `data/processed/ecoli_ciprofloxacin_snp.parquet`.

| Élément | Valeur |
|---|---:|
| Isolats | 5 935 |
| Colonnes totales | 20 933 |
| Features | 20 929 |
| — gènes / clusters pangénomiques | 20 731 |
| — SNP QRDR et régulateurs | 198 |
| Résistants | 1 285 (21,7 %) |

Les 198 SNP sont extraits du VCF `data/raw/Ecoli_AMR_GenotypePhenotype/vcf_files/qrdr/qrdr_non_silent.vcf`, filtrés sur les contigs de `gyrA`, `gyrB`, `parC`, `parE`, `marR`, `acrR` et `soxR`, puis encodés en binaire :

- `0` = allèle de référence / pas de variant détecté ;
- `1` = allèle alternatif détecté.

Chaque variant reçoit un nom lisible, par exemple `SNP_gyrA_Ser83Leu` ou `SNP_parC_Ser80Ile`.

Ces 198 SNP ne sont **jamais utilisés seuls** : ils sont concaténés aux 20 731 features pangénomiques pour former la matrice de 20 929 features. C'est la Lasso (L1) qui décide ensuite quelles variables conserver.

Splits : `data/splits_snp_3way/` (versionnés).

| Split | Isolats | Résistants | Taux R |
|---|---:|---:|---:|
| Train | 4 074 | 975 | 23,9 % |
| Validation | 932 | 156 | 16,7 % |
| Test | 929 | 154 | 16,6 % |

### 3.2 *E. coli* — 48 antibiotiques

Fichier : `data/processed/ecoli_all_snp.parquet`.

| Élément | Valeur |
|---|---:|
| Génomes | 6 949 |
| Colonnes totales | 34 796 |
| Features binaires | 34 744 |
| Antibiotiques | 48 |
| Labels observés (R ou S) | 65 561 |
| Labels résistants | 11 732 (17,9 % des labels observés) |
| Lignées / groupes | 456 |
| Isolats sans lignée (`inconnu`) | 466 |

Encodage des phénotypes dans le fichier source : `1` = résistant, `0` = sensible, `-1` = non testé. Dans les splits, `-1` est converti en `NA` ; chaque antibiotique est traité comme un problème binaire séparé.

Splits : `data/splits_all_snp_3way/` (**non versionnés**, à générer avec le script de la section 8).

| Split | Génomes | Labels observés | Labels R | Taux R |
|---|---:|---:|---:|---:|
| Train | 4 722 | 44 823 | 8 302 | 18,5 % |
| Validation | 1 110 | 10 517 | 1 728 | 16,4 % |
| Test | 1 117 | 10 221 | 1 702 | 16,7 % |

### 3.3 Sélection des antibiotiques exploitables

Le notebook `all_antibiotic.ipynb` calcule, pour chaque antibiotique, le nombre d'individus testés et la prévalence de la résistance sur les trois splits. Deux critères sont proposés :

- **strict** : au moins 100 individus testés et au moins 10 % de classe minoritaire sur chaque split ;
- **élargi** : au moins 100 individus testés et au moins 5 % de classe minoritaire.

Résultat de la sélection :

| Critère | Nombre | Antibiotiques |
|---|---:|---|
| Strict | 9 | ciprofloxacin, ampicillin, cefuroxime, amoxicillin/clavulanic acid, trimethoprim/sulfamethoxazole, amoxicillin, tobramycin, trimethoprim, cefepime |
| Élargi | 14 | les 9 stricts + gentamicin, ceftazidime, cefotaxime, aztreonam, nalidixic acid |

Dans le notebook, la cellule de sélection retient actuellement `ATB_CIBLES = CANDIDATS_STRICT` (9 antibiotiques) — le commentaire de la cellule mentionne `CANDIDATS_LARGE`, mais la ligne exécutée est bien `CANDIDATS_STRICT`. L'exploration détaillée se fait ensuite sur **un seul antibiotique à la fois**, via la variable `ATB` (par défaut : `cefuroxime`).

---

## 4. Les notebooks

### 4.1 `notebooks/ciprofloxacine_exploration.ipynb`

Notebook de référence, autonome. Il charge directement :

- `data/processed/ecoli_ciprofloxacin_snp.parquet` pour l'exploration descriptive ;
- `data/splits_snp_3way/` pour l'entraînement et l'évaluation.

Contenu :

1. importation des données et des splits ;
2. régression logistique classique et pénalisée L1 / Lasso (balayage de `C`) ;
3. Random Forest et XGBoost ;
4. comparaison des modèles ;
5. choix du seuil maximisant le F1 sur la **validation** ;
6. courbe de calibration ;
7. évaluation sur le **test** ;
8. SHAP et analyse d'un faux positif / faux négatif ;
9. MLP PyTorch sur toutes les features ;
10. MLP PyTorch sur les 50 features les plus importantes selon SHAP.

> **Note scikit-learn.** Le notebook ciprofloxacine utilise `LogisticRegression(penalty='l1', solver='liblinear')`. Depuis scikit-learn 1.8, `penalty` est déprécié et un avertissement s'affiche ; dans l'exécution actuelle, la L1 est bien active (68 coefficients non nuls sur 20 929). Pour une reproduction sans avertissement, utiliser `l1_ratio=1` et laisser `penalty` à sa valeur par défaut, comme le fait déjà `all_antibiotic.ipynb`.

### 4.2 `notebooks/all_antibiotic.ipynb`

Notebook « entonnoir » pour les 48 antibiotiques :

1. chargement de `data/splits_all_snp_3way/` ;
2. couverture et prévalence par antibiotique ;
3. calcul des candidats stricts et élargis ;
4. sélection de l'antibiotique à explorer (`ATB`) ;
5. régression logistique, L1, Random Forest, XGBoost ;
6. seuil, calibration, test, SHAP, analyse d'erreurs ;
7. MLP complet et MLP top-50.

Les résultats affichés dans le notebook portent sur **la céfuroxime**. Changer `ATB` relance l'analyse pour un autre antibiotique éligible.

> **Prérequis technique.** La cellule de sélection importe `prepare_data_for_atb` depuis `src/pipeline.py`. Ce module est absent du commit actuel. Pour exécuter le notebook, restaurez-le depuis l'historique :
>
> ```bash
> git restore --source=5450a78 -- src/pipeline.py src/__init__.py
> ```
>
> Le notebook `ciprofloxacine_exploration.ipynb`, lui, ne dépend pas de ce module.

### 4.3 Périmètre exact

Pour éviter les contresens :

- le notebook ciprofloxacine **ne restreint jamais** les features aux 198 SNP : il travaille sur les 20 929 features et la **Lasso (L1)** met à zéro les coefficients inutiles ;
- le notebook 48 antibiotiques **ne boucle pas** sur les 14 candidats : il prépare la sélection et explore un antibiotique à la fois ;
- les hyperparamètres des modèles classiques sont majoritairement les valeurs par défaut ; seul `C` de la L1 est balayé ;
- les MLP sont des baselines volontairement simples, avec un seul seed et une seule architecture par variante.

---

## 5. Protocole d'évaluation

### 5.1 Split groupé pour éviter la fuite de lignage

Un split aléatoire plaçait **95,8 %** des isolats de test dans un groupe MLST déjà présent dans l'entraînement : le modèle pouvait apprendre des **marqueurs de clones** plutôt que la résistance.

Le découpage utilisé est donc **groupé par lignée** :

- `StratifiedGroupKFold` à 6 folds ;
- 4 folds pour l'entraînement, 1 pour la validation, 1 pour le test ;
- aucun groupe connu ne peut apparaître dans deux splits ;
- les isolats sans lignée (`unknown` / `inconnu`) sont traités comme des groupes individuels ;
- le couple (validation, test) est choisi pour se rapprocher de 4/6, 1/6, 1/6 et équilibrer les taux de résistance.

Pour la ciprofloxacine, la stratification porte sur `y` et les groupes sur `mlst_group`. Pour le jeu 48 antibiotiques, la stratification porte sur la **charge de résistance** (nombre d'antibiotiques résistants par génome) et les groupes sur `lignee`.

### 5.2 Métriques

Avec ~17 % de résistants, l'accuracy est trompeuse : un modèle qui prédit « toujours sensible » atteint déjà **83,4 %** sur le test ciprofloxacine et **89,0 %** sur le test céfuroxime.

Les métriques retenues sont donc :

| Métrique | Ce qu'elle mesure | Pourquoi |
|---|---|---|
| **PR-AUC** | précision × rappel, tous seuils | adaptée aux classes rares ; baseline = prévalence |
| **Recall R** | fraction de résistants détectés | ne pas manquer un résistant |
| **F1 R** | compromis précision / rappel sur la classe R | synthèse utile pour la décision |
| **Balanced accuracy** | moyenne des recalls des deux classes | robuste au déséquilibre |
| ROC-AUC | discrimination globale | complément, moins sensible au déséquilibre |
| Calibration | fiabilité des probabilités | un seuil n'a de sens que si les probabilités sont calibrées |

### 5.3 Choix du seuil

Le seuil est choisi pour maximiser le **F1 sur la validation**, jamais sur le test, puis appliqué au test. C'est le protocole correct, mais le seuil est optimisé sur la validation : l'estimation reste optimiste, faute de validation croisée imbriquée.

Pour la Lasso, le paramètre `C` est également choisi sur la validation (PR-AUC maximale) ; le test n'intervient pas dans ce choix, mais cette sélection ajoute une petite part d'optimisme.

Le test n'est utilisé ni pour choisir le modèle, ni pour choisir le seuil, ni pour l'early stopping. Il sert à l'évaluation finale et à l'analyse SHAP.

---

## 6. Résultats

### 6.1 Ciprofloxacine — validation (seuil 0,5)

| Modèle | Features | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---:|---:|---:|---:|---:|
| Régression logistique | 20 929 | 0,554 | 0,314 | 0,405 | 0,633 |
| Régression logistique L1 (`C = 0,05`) | 68 sélectionnées | **0,973** | 0,904 | 0,910 | 0,944 |
| Random Forest | 20 929 | 0,803 | 0,385 | 0,556 | 0,692 |
| XGBoost | 20 929 | 0,969 | **0,917** | **0,923** | **0,951** |
| MLP complet | 20 929 | 0,685 | 0,795 | 0,537 | 0,780 |
| MLP top-50 SHAP | 50 | 0,964 | **0,917** | 0,902 | 0,947 |

*La cellule de comparaison du notebook affiche la L1 à `C = 0,1` (PR-AUC 0,972) ; le balayage retient `C = 0,05` (0,973).*

*La L1 n'est pas réévaluée sur le test dans le notebook : le seuil final et l'évaluation de test portent sur XGBoost et les deux MLP. Le 0,973 est donc une performance de **validation**.*

### 6.2 Ciprofloxacine — test (seuil choisi sur la validation)

| Modèle | Seuil | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | 0,842 | 0,951 | 0,909 | 0,940 | 0,952 |
| MLP complet | 0,776 | 0,730 | 0,714 | 0,701 | 0,825 |
| MLP top-50 SHAP | 0,960 | **0,954** | 0,909 | **0,943** | **0,953** |

Avec XGBoost au seuil 0,842 : 140 vrais positifs, 14 faux négatifs, 4 faux positifs, 771 vrais négatifs (accuracy 0,981, précision 0,972).

**Lecture :**

- XGBoost est la référence : très bon PR-AUC, recall et F1, avec seulement 4 faux positifs sur le test.
- En validation, la **Lasso (L1)** fait jeu égal avec XGBoost (0,973 vs 0,969) avec **68 variables seulement** : c'est le résultat le plus interprétable du projet.
- Le MLP complet sur-apprend immédiatement : l'early stopping retient l'**epoch 0** et le PR-AUC test plafonne à 0,730.
- Le MLP top-50 rattrape XGBoost (0,954 vs 0,951) mais **seulement après sélection des 50 features par SHAP à partir de XGBoost**. Il ne montre donc pas un gain autonome du deep learning ; il suggère que la réduction de dimension joue un rôle majeur.
- L'écart XGBoost / MLP top-50 est de 0,3 point de PR-AUC : il n'est pas interprétable sans intervalle de confiance.

### 6.3 Céfuroxime — validation (seuil 0,5)

Antibiotique par défaut de `all_antibiotic.ipynb` : 2 676 / 769 / 656 individus testés (train / val / test), prévalence R 14,3 % / 11,6 % / 11,0 %.

| Modèle | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---:|---:|---:|---:|
| Régression logistique | 0,593 | 0,438 | 0,557 | 0,710 |
| Régression logistique L1 (`C = 0,01`) | 0,662 | 0,348 | 0,508 | 0,673 |
| Random Forest | 0,645 | 0,303 | 0,454 | 0,649 |
| XGBoost | **0,688** | **0,483** | **0,632** | **0,739** |

### 6.4 Céfuroxime — test (seuil choisi sur la validation)

| Modèle | Seuil | PR-AUC | Recall R | F1 R | Balanced acc |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0,173 | **0,665** | 0,542 | 0,619 | 0,758 |
| Régression logistique L1 | 0,330 | 0,654 | 0,556 | **0,672** | **0,772** |
| MLP complet | 0,937 | 0,623 | 0,444 | 0,593 | 0,719 |
| MLP top-50 SHAP | 0,667 | 0,635 | **0,597** | 0,623 | 0,779 |

**Lecture :**

- Les écarts sont faibles : aucun modèle ne domine clairement.
- XGBoost a le meilleur PR-AUC, mais la régression L1 a le meilleur F1 et la meilleure balanced accuracy.
- Le seuil optimal de XGBoost est bas (0,173) : le modèle est prudent et privilégie le rappel.
- Le test ne contient que 656 individus dont 72 résistants : les différences de 1 à 3 points ne sont pas fiables.

### 6.5 Interprétation biologique

La régression L1 (Lasso) sur la ciprofloxacine sélectionne **68 variables sur 20 929** (0,32 %). Les trois coefficients positifs dominants sont :

| Feature | Coefficient (échelle standardisée) |
|---|---:|
| `SNP_parC_Ser80Ile` | +1,31 |
| `SNP_gyrA_Ser83Leu` | +1,10 |
| `SNP_gyrA_Asp87Asn` | +1,07 |

Les variables suivantes ont des coefficients beaucoup plus faibles et incluent des gènes/clusters pangénomiques, possiblement des marqueurs de lignage. Le modèle s'appuie donc d'abord sur des mutations **connues** de résistance aux fluoroquinolones, ce qui est biologiquement rassurant.

SHAP est calculé sur le jeu de **test** pour l'interprétation (et sur un échantillon du train pour sélectionner le top-50 du MLP). L'analyse d'erreurs isole :

- ciprofloxacine, XGBoost au seuil 0,842 : **4 faux positifs** et **14 faux négatifs** ;
- céfuroxime, XGBoost au seuil 0,173 : **15 faux positifs** et **33 faux négatifs**.

> Importance ≠ causalité. Les features corrélées se partagent l'importance, et une validation externe reste nécessaire.

---

## 7. Structure du dépôt

```text
Projet_ML_DL/
├── data/
│   ├── processed/
│   │   ├── ecoli_ciprofloxacin_snp.parquet   # 5 935 × 20 933
│   │   └── ecoli_all_snp.parquet             # 6 949 × 34 796
│   ├── splits_snp_3way/                      # splits cipro, versionnés
│   │   ├── X_{train,val,test}.parquet
│   │   └── y_{train,val,test}.parquet
│   ├── splits_all_snp_3way/                  # splits 48 ATB, à générer
│   │   ├── X_{train,val,test}.parquet
│   │   ├── Y_{train,val,test}.parquet
│   │   └── split_summary.json
│   └── raw/                                  # sources brutes, non versionnées
├── notebooks/
│   ├── ciprofloxacine_exploration.ipynb      # notebook de référence
│   ├── ciprofloxacine_exploration.html       # export Quarto
│   ├── all_antibiotic.ipynb                  # sélection + exploration d'un ATB
│   └── all_antibiotic.{html,pdf}             # exports
├── src/
│   ├── add_qrdr_snps.py                      # extraction des 198 SNP QRDR
│   ├── make_3way_splits_snp.py               # split cipro groupé par MLST
│   ├── make_3way_splits_all_snp.py           # split 48 ATB groupé par lignée
│   └── data_loader.py                        # utilitaire historique, non utilisé par les notebooks
├── slides/
│   ├── presentation_ml.qmd                   # présentation cipro (chiffres anciens)
│   ├── presentation_antibiotiques.qmd        # présentation 48 antibiotiques
│   ├── images/
│   └── custom.scss
└── README.md
```

Les scripts de construction historiques (jeu 48 antibiotiques, multi-espèces, export CSV) ont été retirés de l'arbre courant. Les fichiers `data/processed/*.parquet` déjà versionnés permettent de travailler sans les reconstruire.

---

## 8. Installation et démarrage rapide

### 8.1 Dépendances

Python 3.10+ (les sorties des notebooks ont été produites avec Python 3.12).

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install pandas numpy pyarrow scikit-learn xgboost torch shap matplotlib seaborn joblib jupyterlab
```

### 8.2 Charger les splits ciprofloxacine

C'est ce que fait le notebook de référence. Les chemins fonctionnent depuis la racine du dépôt ou depuis `notebooks/`.

```python
from pathlib import Path
import pandas as pd

DATA = Path("data/splits_snp_3way")

X_train = pd.read_parquet(DATA / "X_train.parquet")
y_train = pd.read_parquet(DATA / "y_train.parquet")["y"]

X_val = pd.read_parquet(DATA / "X_val.parquet")
y_val = pd.read_parquet(DATA / "y_val.parquet")["y"]

X_test = pd.read_parquet(DATA / "X_test.parquet")
y_test = pd.read_parquet(DATA / "y_test.parquet")["y"]

print(X_train.shape, y_train.mean())
# (4074, 20929) 0.2393...
```

### 8.3 Charger les splits 48 antibiotiques

```python
from pathlib import Path
import pandas as pd

DATA = Path("data/splits_all_snp_3way")

X_train = pd.read_parquet(DATA / "X_train.parquet")
Y_train = pd.read_parquet(DATA / "Y_train.parquet")

print(X_train.shape, Y_train.shape)
# (4722, 34744) (4722, 48)

# Un antibiotique = une colonne de Y ; NA = non testé
y_cefuroxime = Y_train["cefuroxime"]
print(y_cefuroxime.notna().sum(), y_cefuroxime.mean())
```

### 8.4 Générer les splits 48 antibiotiques

Le dossier `data/splits_all_snp_3way/` n'est pas versionné. Avant d'ouvrir `all_antibiotic.ipynb`, lancez :

```bash
python src/make_3way_splits_all_snp.py
```

Le script lit `data/processed/ecoli_all_snp.parquet` et écrit les splits, les métadonnées et `split_summary.json`. Il vérifie notamment qu'aucune lignée connue ne traverse deux splits.

Pour régénérer les splits ciprofloxacine (déjà versionnés) :

```bash
python src/make_3way_splits_snp.py
```

### 8.5 Ajouter les SNP QRDR (optionnel)

Le script `add_qrdr_snps.py` attend un CSV pangénomique de base, qui n'est pas fourni dans le dépôt actuel. Exemple d'appel :

```bash
python src/add_qrdr_snps.py \
    --vcf data/raw/Ecoli_AMR_GenotypePhenotype/vcf_files/qrdr/qrdr_non_silent.vcf \
    --csv data/processed/ecoli_ciprofloxacin.csv \
    --out-csv data/processed/ecoli_ciprofloxacin_snp.csv \
    --out-parquet data/processed/ecoli_ciprofloxacin_snp.parquet \
    --min-carriers 5
```

Le script écrit aussi un manifeste JSON à côté du Parquet (comptages, variants retenus, paramètres).

### 8.6 Lancer les notebooks

```bash
jupyter lab
```

Ordre recommandé :

1. `notebooks/ciprofloxacine_exploration.ipynb` — autonome, ne nécessite que les fichiers versionnés ;
2. `notebooks/all_antibiotic.ipynb` — nécessite `data/splits_all_snp_3way/` et `src/pipeline.py`.

---

## 9. Limites

- **Test petit** : 929 isolats / 154 résistants pour la ciprofloxacine ; 656 / 72 pour la céfuroxime. Les intervalles de confiance sont larges et les écarts de quelques points ne sont pas fiables.
- **Split groupé par MLST / lignée**, pas par phylogénie fine. Une même étude peut être présente dans plusieurs splits.
- **Pas de validation externe** sur une autre cohorte.
- **Seuil et `C` de la Lasso optimisés sur la validation**, puis évalués sur le test : estimation optimiste, sans validation croisée imbriquée.
- **Calibration explorée mais non corrigée** (Platt / isotonic).
- **Deep learning** : une seule architecture par variante, un seul seed pour les modèles classiques ; les MLP ne fixent pas explicitement `torch.manual_seed`, donc leurs résultats peuvent varier d'une exécution à l'autre.
- **Pas de modèle SNP-only (choix assumé)** : le notebook n'isole jamais les 198 SNP et travaille toujours sur la matrice complète. Il ne permet donc pas de conclure que « les SNP suffisent », mais ce n'était pas l'objectif : la Lasso sélectionne les variables utiles, SNP ou non.
- **Le notebook 48 antibiotiques n'explore qu'un antibiotique à la fois** ; les résultats présentés concernent la céfuroxime.
- **Features pangénomiques partiellement non annotées** : risque de marqueurs de lignage.
- **Slides** : `presentation_antibiotiques.qmd` suit le notebook 48 antibiotiques ; `presentation_ml.qmd` conserve des chiffres cipro d'une version antérieure (LR sur 198 SNP seuls, ancien MLP à 0,775, Brier 0,020) qui ne sont pas reproduits par les notebooks actuels.

---

## 10. Perspectives

- Validation croisée groupée à 5 folds, plusieurs seeds, intervalles de confiance par bootstrap.
- Validation externe et évaluation d'un seuil **figé** sur le test.
- Comparaison explicite **pangénome seul vs pangénome + 198 SNP**, sans jamais entraîner de modèle SNP-only.
- Correction de la calibration (Platt / isotonic) et intégration dans la décision.
- Analyse d'erreurs par lignage et confrontation à la littérature.
- Passage à un vrai modèle multi-tâches sur les 48 antibiotiques, au lieu d'un modèle par antibiotique.
- Enrichissement biologique : pompes d'efflux, plasmides, gènes acquis.
- Sélection conjointe de features SNP + gènes et autoencodeur, toujours sur la matrice complète.
- Alignement des slides sur les sorties actuelles des notebooks.

---

## 11. Équipe

- **Benjamin HIN**
- **Alexandre**
- **Baptiste**

Projet réalisé dans le cadre d'un cours de Machine Learning / Deep Learning.
