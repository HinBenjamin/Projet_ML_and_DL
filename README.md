# 🧬 Prédire la résistance aux antibiotiques à partir du génome

On cherche à prédire si une souche d'*E. coli* est **résistante ou sensible** à un antibiotique **à partir de son génome**. Chaque génome est décrit par la présence ou l'absence de ~20 000 gènes (pangénome) et par 198 mutations connues (SNP). On compare des modèles de ML (régression logistique, Lasso, Random Forest, XGBoost) et de DL (MLP). On vérifie aussi que les variables qu'ils utilisent ont un sens biologique.

L'objectif n'est pas seulement d'obtenir un bon score, mais un modèle **utile, interprétable et honnêtement évalué** : split groupé pour éviter la fuite de clones, métriques adaptées au déséquilibre, **régularisation Lasso (L1)** pour sélectionner les variables, seuil choisi sur la validation, et lecture biologique des variables.

Le projet se déroule en deux temps :

1. **Ciprofloxacine** : le cas d'étude principal, traité en détail.
2. **48 antibiotiques** : extension à d'autres molécules. On sélectionne les antibiotiques pour lesquels on a assez de données (9 au critère strict). On en explore un en détail (la céfuroxime), puis on applique un pipeline automatique à 8 antibiotiques.

> **Source des résultats.** Les chiffres de ce README viennent des sorties des notebooks `notebooks/ciprofloxacine_exploration.ipynb` et `notebooks/all_antibiotic.ipynb`.

---

## Sommaire

1. [Résultats clés](#1-résultats-clés)
2. [Contexte biologique](#2-contexte-biologique)
3. [Les données](#3-les-données)
4. [Méthode](#4-méthode)
5. [Résultats détaillés](#5-résultats-détaillés)
6. [Limites et perspectives](#6-limites-et-perspectives)
7. [Reproduire les résultats](#7-reproduire-les-résultats)
8. [Structure du dépôt](#8-structure-du-dépôt)
9. [Glossaire](#9-glossaire)

---

## 1. Résultats clés

| Ciprofloxacine                           | Modèle                         |                 PR-AUC |
| ---------------------------------------- | ------------------------------- | ---------------------: |
| Meilleur modèle évalué sur le test    | XGBoost                         |        **0,951** |
| Modèle le plus simple et interprétable | Lasso (68 variables sur 20 929) | 0,973 *(validation)* |
| Deep learning sur toutes les variables   | MLP                             |                  0,730 |

- **Un modèle simple suffit presque.** Le Lasso ne garde que 68 variables (0,3 %) et obtient un score proche de celui de XGBoost.
- **Les variables retenues sont biologiquement cohérentes.** En tête : trois mutations connues de résistance aux fluoroquinolones, `parC_Ser80Ile`, `gyrA_Ser83Leu` et `gyrA_Asp87Asn`.
- **Le deep learning n'apporte pas de gain propre.** Le MLP sur toutes les variables sur-apprend. Il n'atteint le niveau de XGBoost (0,954) que si on lui donne les 50 meilleures variables choisies par XGBoost.
- **Les performances varient beaucoup d'un antibiotique à l'autre.** Sur 8 antibiotiques, le PR-AUC test va de 0,98 (ciprofloxacine) et 0,96 (ampicilline) à 0,65 (céfuroxime) et 0,51 (amoxicilline/acide clavulanique). La difficulté dépend surtout du mécanisme biologique de la résistance.

> ⚠️ Le score du Lasso est mesuré sur la **validation**, le jeu qui a aussi servi à choisir son hyperparamètre `C`. Il est donc un peu optimiste et n'est pas directement comparable au score de test de XGBoost (voir [§ 5.1](#51-ciprofloxacine)).

---

## 2. Contexte biologique

- *E. coli* est une bactérie pathogène majeure, et ses résistances aux antibiotiques augmentent.
- La **ciprofloxacine** est un antibiotique très utilisé de la famille des **fluoroquinolones**. Elle bloque deux enzymes indispensables à la copie de l'ADN : l'ADN gyrase (gènes `gyrA`, `gyrB`) et la topoisomérase IV (gènes `parC`, `parE`).
- La résistance vient surtout de **mutations ponctuelles** dans une petite région de ces gènes, la **QRDR** (*Quinolone Resistance-Determining Region*). Des mutations dans des régulateurs de pompes d'efflux (`marR`, `acrR`, `soxR`) peuvent aussi y contribuer.

**Hypothèse du projet :** si quelques mutations connues expliquent l'essentiel de la résistance, un modèle simple et interprétable devrait approcher les performances d'un modèle complexe. On teste cette hypothèse avec un Lasso entraîné sur **toutes** les variables. On ne lui impose pas les SNP : on regarde s'il les retrouve de lui-même.

---

## 3. Les données

Les données viennent d'un jeu public de génomes d'*E. coli* associés à des phénotypes de résistance (dossier `Ecoli_AMR_GenotypePhenotype`). Les fichiers prêts à l'emploi sont dans `data/processed/` et sont versionnés. Les données brutes (~4 Go) ne le sont pas.

### 3.1 Ciprofloxacine (jeu principal)

Fichier : `data/processed/ecoli_ciprofloxacin_snp.parquet`, une ligne par génome.

| Élément                                   |           Valeur |
| ------------------------------------------- | ---------------: |
| Génomes                                    |            5 935 |
| Variables (features)                        | **20 929** |
| — gènes du pangénome (présence/absence) |           20 731 |
| — SNP QRDR et régulateurs                 |              198 |
| Résistants                                 |   1 285 (21,7 %) |

**Les SNP** sont extraits du fichier de variants (VCF) du jeu de données, sur les gènes `gyrA`, `gyrB`, `parC`, `parE`, `marR`, `acrR` et `soxR`. Chaque mutation devient une colonne binaire (`0` = séquence de référence, `1` = mutation présente) avec un nom lisible, par exemple `SNP_gyrA_Ser83Leu`. Ce nom signifie que la sérine en position 83 de GyrA est remplacée par une leucine.

> **Choix de méthode.** Aucun modèle n'est entraîné sur les 198 SNP seuls. Ils sont toujours ajoutés aux 20 731 gènes, et c'est au modèle de trouver les variables utiles.

### 3.2 48 antibiotiques (extension)

Fichier : `data/processed/ecoli_all_snp.parquet`.

| Élément                             |                                   Valeur |
| ------------------------------------- | ---------------------------------------: |
| Génomes                              |                                    6 949 |
| Variables binaires                    |                                   34 744 |
| Antibiotiques                         |                                       48 |
| Résultats de test observés (R ou S) |                                   65 561 |
| — dont résistants                   |                          11 732 (17,9 %) |
| Lignées                              | 456 (+ 466 génomes sans lignée connue) |

Tous les génomes n'ont pas été testés sur tous les antibiotiques. Dans le fichier source, `1` = résistant, `0` = sensible et `-1` = non testé (converti en valeur manquante). Chaque antibiotique est traité comme un problème de classification binaire séparé.

**Quels antibiotiques étudier ?** Un antibiotique n'est exploitable que s'il a été testé sur assez de génomes et s'il y a assez de résistants. Le notebook `all_antibiotic.ipynb` applique deux critères, vérifiés sur chacun des trois jeux (train, validation, test) :

| Critère                    | Règle                                                   | Antibiotiques retenus                                                                                                                                            |
| --------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Strict** (utilisé) | ≥ 100 génomes testés et ≥ 10 % de classe minoritaire | **9** : ciprofloxacin, ampicillin, cefuroxime, amoxicillin/clavulanic acid, trimethoprim/sulfamethoxazole, amoxicillin, tobramycin, trimethoprim, cefepime |
| Élargi                     | ≥ 100 génomes testés et ≥ 5 % de classe minoritaire  | 14 : les 9 stricts + gentamicin, ceftazidime, cefotaxime, aztreonam, nalidixic acid                                                                              |

Le notebook range ces 9 antibiotiques dans la liste `ATB_CIBLES`. L'analyse détaillée se fait ensuite sur **un antibiotique à la fois**, choisi avec la variable `ATB`. Les sorties enregistrées portent sur la céfuroxime (§ 5.3). Une exécution automatique sur 8 antibiotiques est présentée au § 5.4.

---

## 4. Méthode

### 4.1 Découpage train / validation / test sans fuite entre clones

Des bactéries d'une même lignée (même **MLST**) sont presque identiques génétiquement. Avec un découpage aléatoire, **95,8 %** des génomes de test avaient un « cousin » dans le jeu d'entraînement. Le modèle pouvait alors reconnaître un clone au lieu d'apprendre la résistance, et le score de test aurait été trop optimiste.

On découpe donc **par lignée** : une lignée est entièrement dans le train, ou entièrement dans la validation, ou entièrement dans le test.

- `StratifiedGroupKFold` en 6 parts : 4 pour l'entraînement, 1 pour la validation, 1 pour le test (≈ 4/6, 1/6, 1/6) ;
- le taux de résistants est équilibré entre les trois jeux ;
- les génomes sans lignée connue forment chacun leur propre groupe.

| Ciprofloxacine | Génomes | Résistants | Taux R |
| -------------- | -------: | ----------: | -----: |
| Train          |    4 074 |         975 | 23,9 % |
| Validation     |      932 |         156 | 16,7 % |
| Test           |      929 |         154 | 16,6 % |

| 48 antibiotiques | Génomes | Résultats observés | dont R | Taux R |
| ---------------- | -------: | -------------------: | -----: | -----: |
| Train            |    4 722 |               44 823 |  8 302 | 18,5 % |
| Validation       |    1 110 |               10 517 |  1 728 | 16,4 % |
| Test             |    1 117 |               10 221 |  1 702 | 16,7 % |

Pour la ciprofloxacine, les groupes sont les `mlst_group`. Pour les 48 antibiotiques, ce sont les `lignee`, et la stratification porte sur le nombre d'antibiotiques auxquels chaque génome résiste.

**Rôle de chaque jeu :**

- **train** : entraîner les modèles ;
- **validation** : comparer les modèles, choisir `C` du Lasso, choisir le seuil de décision et arrêter l'entraînement des MLP (*early stopping*) ;
- **test** : uniquement l'évaluation finale et l'interprétation (SHAP). Il ne sert à aucun choix.

### 4.2 Métriques

Il y a ~17 % de résistants dans le test. Un modèle qui répond toujours « sensible » obtient donc déjà **83,4 %** d'accuracy sur la ciprofloxacine (89,0 % sur la céfuroxime) : l'accuracy n'est pas informative ici. On utilise plutôt :

| Métrique                               | Ce qu'elle mesure                                                | Pourquoi                                                                         |
| --------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **PR-AUC** (métrique principale) | qualité précision / rappel sur tous les seuils                 | adaptée aux classes rares ; un modèle au hasard obtient la prévalence (~0,17) |
| **Recall R**                      | part des résistants détectés                                  | en clinique, rater un résistant est l'erreur la plus grave                      |
| **F1 R**                          | compromis précision / rappel sur les résistants                | sert à choisir le seuil                                                         |
| **Balanced accuracy**             | moyenne du rappel des deux classes                               | robuste au déséquilibre                                                        |
| ROC-AUC                                 | capacité globale à séparer les classes                        | complément                                                                      |
| Calibration                             | une probabilité de 0,8 correspond-elle à 80 % de résistants ? | un seuil n'a de sens que si les probabilités sont fiables                       |

### 4.3 Modèles comparés

| Modèle                                                 | Rôle                                                                                                                                                            |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Régression logistique                                  | modèle linéaire de référence                                                                                                                                 |
| **Lasso** (régression logistique pénalisée L1) | modèle linéaire qui **met à zéro** les coefficients des variables inutiles, donc sélectionne lui-même ses variables ; `C` choisi sur la validation |
| Random Forest                                           | ensemble d'arbres                                                                                                                                                |
| XGBoost                                                 | arbres en *boosting* ; capte les interactions entre variables                                                                                                  |
| MLP complet                                             | réseau de neurones (PyTorch) sur les 20 929 variables                                                                                                           |
| MLP top-50                                              | même type de réseau, sur les 50 variables les plus importantes selon SHAP (calculé avec XGBoost)                                                              |

Les hyperparamètres sont ceux par défaut, sauf `C` pour le Lasso. Les MLP sont des modèles de base volontairement simples (une architecture, un seul seed).

### 4.4 Seuil de décision

Un modèle donne une probabilité, et on déclare « résistant » au-dessus d'un seuil. Au lieu du seuil de 0,5 par défaut, on prend celui qui **maximise le F1 sur la validation**, puis on l'applique tel quel au test.

### 4.5 Interprétation

On utilise deux approches complémentaires :

- **Coefficients du Lasso** : variables standardisées, donc coefficients comparables entre eux. Un coefficient positif pousse vers « résistant ».
- **Valeurs SHAP** sur XGBoost : contribution de chaque variable à chaque prédiction, calculée sur le test.

Si les deux méthodes mettent en avant les mêmes mutations, le signal est solide. On regarde aussi les **erreurs** (faux positifs et faux négatifs) une par une avec SHAP.

---

## 5. Résultats détaillés

### 5.1 Ciprofloxacine

**Validation** (seuil 0,5) :

| Modèle                        | Variables utilisées |          PR-AUC |        Recall R |            F1 R |   Balanced acc. |
| ------------------------------ | -------------------: | --------------: | --------------: | --------------: | --------------: |
| Régression logistique         |               20 929 |           0,554 |           0,314 |           0,405 |           0,633 |
| **Lasso** (`C = 0,05`) |         **68** | **0,973** |           0,904 |           0,910 |           0,944 |
| Random Forest                  |               20 929 |           0,803 |           0,385 |           0,556 |           0,692 |
| XGBoost                        |               20 929 |           0,969 | **0,917** | **0,923** | **0,951** |
| MLP complet                    |               20 929 |           0,685 |           0,795 |           0,537 |           0,780 |
| MLP top-50 SHAP                |                   50 |           0,964 | **0,917** |           0,902 |           0,947 |

**Test** (seuil choisi sur la validation) :

| Modèle           | Seuil |          PR-AUC | Recall R |            F1 R |   Balanced acc. |
| ----------------- | ----: | --------------: | -------: | --------------: | --------------: |
| **XGBoost** | 0,842 |           0,951 |    0,909 |           0,940 |           0,952 |
| MLP complet       | 0,776 |           0,730 |    0,714 |           0,701 |           0,825 |
| MLP top-50 SHAP   | 0,960 | **0,954** |    0,909 | **0,943** | **0,953** |

XGBoost sur le test : sur 154 résistants, 140 sont détectés et 14 sont manqués, avec seulement 4 faux positifs parmi 775 sensibles (accuracy 0,981, précision 0,972).

**Lecture :**

- **XGBoost** est le modèle de référence : excellent sur le test, très peu de fausses alertes.
- Le **Lasso** égale XGBoost en validation avec seulement 68 variables. C'est le résultat le plus interprétable du projet, mais il reste à confirmer sur le test.
- La **régression logistique** sans pénalité et la **Random Forest** souffrent du grand nombre de variables (20 929 pour ~4 000 génomes).
- Le **MLP complet** sur-apprend dès la première époque (l'*early stopping* retient l'époque 0).
- Le **MLP top-50** rattrape XGBoost, mais seulement grâce aux variables choisies par XGBoost. Le gain vient de la réduction du nombre de variables, pas du deep learning.
- Un écart de 0,3 point entre XGBoost et le MLP top-50 n'est pas significatif sur un test de cette taille.

### 5.2 Ce que le modèle a appris

Le Lasso garde **68 variables sur 20 929**. Les trois plus importantes sont des mutations QRDR bien décrites dans la littérature :

| Variable              | Coefficient (standardisé) |
| --------------------- | -------------------------: |
| `SNP_parC_Ser80Ile` |                      +1,31 |
| `SNP_gyrA_Ser83Leu` |                      +1,10 |
| `SNP_gyrA_Asp87Asn` |                      +1,07 |

Les variables suivantes ont des coefficients bien plus faibles. Ce sont surtout des gènes du pangénome, qui marquent probablement des lignées plutôt qu'un mécanisme de résistance. Le modèle s'appuie donc d'abord sur des mécanismes **connus**, ce qui est rassurant.

> Importance ≠ causalité. Des variables corrélées se partagent l'importance, et une validation sur des données externes reste nécessaire.

### 5.3 Céfuroxime : exploration détaillée dans `all_antibiotic.ipynb`

Le notebook déroule le même protocole que pour la ciprofloxacine, sur un seul antibiotique. Les sorties enregistrées portent sur la **céfuroxime** : 2 676 / 769 / 656 génomes, et 14,3 % / 11,6 % / 11,0 % de résistants (train / val / test).

**Test** (chaque modèle a son seuil, choisi sur la validation) :

| Modèle              | Seuil |          PR-AUC |        Recall R |            F1 R |   Balanced acc. |
| -------------------- | ----: | --------------: | --------------: | --------------: | --------------: |
| XGBoost              | 0,173 | **0,665** |           0,542 |           0,619 |           0,758 |
| Lasso (`C = 0,01`) | 0,330 |           0,654 |           0,556 | **0,672** |           0,772 |
| MLP complet          | 0,937 |           0,623 |           0,444 |           0,593 |           0,719 |
| MLP top-50 SHAP      | 0,667 |           0,635 | **0,597** |           0,623 | **0,779** |

- Les modèles sont **très proches** entre eux et nettement moins bons que sur la ciprofloxacine.
- XGBoost a le meilleur PR-AUC, le Lasso le meilleur F1 R, et le MLP top-50 le meilleur recall et la meilleure balanced accuracy. Avec 72 résistants dans le test, ces écarts ne sont pas significatifs.
- XGBoost au seuil 0,173 : 39 résistants détectés sur 72, 33 manqués, et 15 faux positifs.

Explication probable : la résistance aux céphalosporines comme la céfuroxime combine plusieurs mécanismes (β-lactamases souvent portées par des plasmides, porines, efflux), mal captés par une matrice de présence/absence. Les mutations `gyrA` / `parC` qui ressortent aussi sur la céfuroxime y jouent probablement le rôle de **marqueurs de lignée** (co-résistance), pas de cause.

### 5.4 Passage à l'échelle : exécution du pipeline sur 8 antibiotiques

`src/pipeline.py` contient une fonction `run_antibiotic_pipeline(atb, data)`. Elle applique automatiquement le même protocole à un antibiotique : XGBoost et Lasso, choix du modèle et du seuil sur la validation, évaluation sur le test. Une **version précédente** de `all_antibiotic.ipynb` (commit `5450a78`) l'a lancée en boucle sur **8 antibiotiques** : les 5 premiers du critère strict + gentamicine, ceftazidime et céfotaxime. Cette boucle n'est plus dans la version actuelle du notebook. Les résultats ci-dessous viennent de cette exécution : fichier `reports/summary_by_atb.csv` (présent dans ce commit, plus dans l'arbre actuel) et figure `slides/images/fig_batch_pr_auc.png`.

| Antibiotique                  | Critère | Génomes test | % R test |     PR-AUC test | Hasard (= % R) |
| ----------------------------- | -------- | ------------: | -------: | --------------: | -------------: |
| ciprofloxacin                 | strict   |           939 |   17,6 % | **0,977** |           0,18 |
| ampicillin                    | strict   |           784 |   47,8 % | **0,960** |           0,48 |
| trimethoprim/sulfamethoxazole | strict   |           522 |   30,3 % |           0,853 |           0,30 |
| cefotaxime                    | élargi  |           852 |   11,5 % |           0,838 |           0,12 |
| gentamicin                    | élargi  |           933 |    9,1 % |           0,810 |           0,09 |
| ceftazidime                   | élargi  |           921 |    9,6 % |           0,721 |           0,10 |
| cefuroxime                    | strict   |           656 |   11,0 % |           0,649 |           0,11 |
| amoxicillin/clavulanic acid   | strict   |           608 |   31,1 % |           0,513 |           0,31 |

Dans les 8 cas, XGBoost est le modèle retenu.

**Lecture :**

- Tous les antibiotiques font mieux que le hasard, mais les écarts de performance sont **très grands**.
- La performance suit la **biologie** plus que la prévalence. L'amoxicilline/acide clavulanique a 31 % de résistants mais seulement 0,51 de PR-AUC, alors que la ciprofloxacine, avec 18 % de résistants, atteint 0,98. Une résistance portée par quelques mutations ciblées est facile à prédire ; une résistance qui combine plusieurs mécanismes l'est beaucoup moins.
- Ces scores ne sont **pas directement comparables** à ceux des § 5.1 et 5.3. Le pipeline utilise d'autres hyperparamètres XGBoost, la matrice à 34 744 variables et ses propres splits. Par exemple, la ciprofloxacine donne 0,977 ici contre 0,951 dans le notebook dédié, et la céfuroxime 0,649 contre 0,665.

---

## 6. Limites et perspectives

**Limites**

- **Test de petite taille** : 154 résistants pour la ciprofloxacine, 72 pour la céfuroxime. Les intervalles de confiance sont larges, et des écarts de quelques points ne sont pas fiables.
- **Ciprofloxacine : le Lasso n'est évalué qu'en validation**, sur le jeu qui a aussi servi à choisir `C`. Pour la céfuroxime, il est bien évalué sur le test.
- **Le seuil est choisi sur la validation** puis appliqué au test : sans validation croisée imbriquée, l'estimation reste un peu optimiste.
- **Calibration** : elle est examinée, mais pas corrigée (pas de Platt ni d'isotonic).
- **Découpage par MLST / lignée**, pas par distance phylogénétique fine. Une même étude d'origine peut se retrouver dans plusieurs jeux.
- **Un seul découpage et un seul seed** : on ne mesure pas la variabilité des résultats.
- **Extension partielle** : l'exploration détaillée ne porte que sur la céfuroxime. L'exécution automatique couvre 8 antibiotiques, et non les 9 du critère strict : amoxicilline, tobramycine, triméthoprime et céfépime n'ont pas été lancés, alors que gentamicine, ceftazidime et céfotaxime, qui relèvent du critère élargi, l'ont été.

**Perspectives**

- Évaluer le Lasso sur le test et comparer systématiquement ses variables à celles de SHAP.
- Validation croisée groupée (5 folds), plusieurs seeds et intervalles de confiance par bootstrap.
- Mesurer l'apport des SNP en comparant pangénome seul et pangénome + 198 SNP.

## 7. Reproduire les résultats

### 7.1 Installation

Python 3.12 et les bibliothèques suivantes :

```bash
pip install numpy pandas pyarrow scikit-learn xgboost shap torch matplotlib seaborn joblib jupyter
```

### 7.2 Données

Les fichiers de `data/processed/` et les splits ciprofloxacine (`data/splits_snp_3way/`) sont **versionnés** : on peut lancer le notebook ciprofloxacine directement.

Les splits des 48 antibiotiques ne sont pas versionnés. Il faut les générer une fois, depuis la racine du dépôt :

```bash
python src/make_3way_splits_all_snp.py   # crée data/splits_all_snp_3way/
```

Pour reconstruire les splits ciprofloxacine (déjà fournis) :

```bash
python src/make_3way_splits_snp.py       # crée data/splits_snp_3way/
```

`src/add_qrdr_snps.py` ajoute les SNP QRDR au tableau du pangénome. Il nécessite les données brutes (fichier VCF) ; voir l'exemple d'utilisation en tête du script.

### 7.3 Notebooks à exécuter

| Notebook                                       | Contenu                                                                                  | Prérequis                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `notebooks/ciprofloxacine_exploration.ipynb` | **notebook principal** : modèles ML, seuil, calibration, test, SHAP, erreurs, MLP | aucun                                                  |
| `notebooks/all_antibiotic.ipynb`             | sélection des antibiotiques puis analyse complète d'un antibiotique (`ATB`)          | splits 48 ATB générés ; utilise `src/pipeline.py` |

Exécuter les notebooks de haut en bas. Des exports HTML / PDF sont disponibles à côté de chaque notebook pour une lecture sans exécution.

---

## 8. Structure du dépôt

```text
Projet_ML_and_DL/
├── data/
│   ├── processed/                           # tableaux prêts à l'emploi (versionnés)
│   │   ├── ecoli_ciprofloxacin_snp.parquet  # 5 935 génomes × 20 929 variables
│   │   └── ecoli_all_snp.parquet            # 6 949 génomes × 34 744 variables, 48 ATB
│   ├── splits_snp_3way/                     # train / val / test ciprofloxacine (versionnés)
│   ├── splits_all_snp_3way/                 # train / val / test 48 ATB (à générer)
│   └── Ecoli_AMR_GenotypePhenotype/         # données brutes (non versionnées)
├── notebooks/
│   ├── ciprofloxacine_exploration.ipynb     # notebook principal (+ export .html)
│   ├── all_antibiotic.ipynb                 # extension 48 ATB (+ exports .html / .pdf)
│   └── exploration_*.ipynb                  # brouillons personnels des membres de l'équipe
├── src/
│   ├── add_qrdr_snps.py                     # extraction des 198 SNP QRDR depuis le VCF
│   ├── make_3way_splits_snp.py              # découpage ciprofloxacine, groupé par MLST
│   ├── make_3way_splits_all_snp.py          # découpage 48 ATB, groupé par lignée
│   └── pipeline.py                          # préparation des données + pipeline complet par antibiotique
├── slides/
│   ├── presentation_antibiotiques.qmd       # présentation Quarto (+ export .html)
│   ├── images/                              # figures
│   └── custom.scss
└── README.md
```

---

## 9. Glossaire

| Terme                    | Définition                                                                                                                                       |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pangénome**     | ensemble de tous les gènes trouvés chez au moins une souche de l'espèce ; chaque souche n'en possède qu'une partie                            |
| **SNP**            | *Single Nucleotide Polymorphism* : mutation d'une seule base de l'ADN                                                                           |
| **QRDR**           | *Quinolone Resistance-Determining Region* : région de `gyrA` / `parC` où se concentrent les mutations de résistance aux fluoroquinolones |
| **MLST / lignée** | typage qui regroupe des souches génétiquement très proches (des « clones »)                                                                  |
| **R / S**          | résistant / sensible à l'antibiotique (phénotype mesuré en laboratoire)                                                                       |
| **Lasso (L1)**     | régression pénalisée qui met à zéro les coefficients des variables peu utiles                                                                |
| **PR-AUC**         | aire sous la courbe précision-rappel ; 1 = parfait, prévalence = hasard                                                                         |
| **SHAP**           | méthode qui mesure la contribution de chaque variable à une prédiction                                                                         |
| **Early stopping** | arrêt de l'entraînement d'un réseau quand le score de validation cesse de progresser                                                           |

---

## Équipe

- **Benjamin HIN**
- **Alexandre AITOUR**
- **Baptiste JACQUET**

Projet réalisé dans le cadre d'un cours de Machine Learning / Deep Learning.
