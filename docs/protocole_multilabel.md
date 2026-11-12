# Jeux de données antibiotiques : *E. coli* 50 antibiotiques et multi-espèces

Ce document décrit **comment ont été construits** les deux jeux de données
antibiotiques du projet, **dans quel format** ils sont livrés et **comment les
utiliser** sans fuite de données.

Deux jeux sont produits car les sources disponibles ne permettent pas un seul
tableau homogène :

| Jeu | Source principale | Génomes | Antibiotiques | Features génomiques | Usage |
|---|---|---:|---:|---|---|
| **E. coli 50 antibiotiques** | `phenotype_matrix_08302024.csv` | 6 983 (5 935 avec features) | 50 | oui (20 929) | **jeu principal**, extension directe du projet ciprofloxacine |
| **Multi-espèces** | 80 fichiers `Data_*.txt` | 31 704 | 44 standardisés (46 libellés bruts) | non | analyse multi-espèces, phénotypes seuls |

Le premier jeu est celui qui répond le mieux à l'objectif « tous les
antibiotiques » pour la partie ML/DL du projet : il contient les 50
antibiotiques du panel *E. coli* et se branche directement sur les features
pangénome + SNP déjà utilisées. Le second est un jeu complémentaire, plus
large en espèces mais limité aux phénotypes.

---

# 1. Jeu principal : *E. coli*, 50 antibiotiques

## 1.1 Sources

| Fichier | Contenu | Dimensions |
|---|---|---|
| `data/raw/Ecoli_AMR_GenotypePhenotype/phenotype_matrix/phenotype_matrix_08302024.csv` | phénotypes bruts | 6 983 génomes × 50 antibiotiques |
| `data/raw/Ecoli_AMR_GenotypePhenotype/strains_metadata.csv` | métadonnées BV-BRC | 6 985 lignes |
| `data/processed/ecoli_ciprofloxacin_snp.parquet` | features pangénome + 198 SNP QRDR | 5 935 génomes × 20 929 features |
| `data/splits_snp_3way/metadata_{train,val,test}.parquet` | split 3-way groupé par MLST | 5 935 génomes |

Les 50 antibiotiques couvrent notamment les bêta-lactamines, les
aminoglycosides, les fluoroquinolones, les carbapénèmes, les tétracyclines,
les sulfamides et les phénicols.

## 1.2 Définition des labels

La matrice source contient quatre situations : `Resistant`, `Susceptible`,
`Intermediate` et une valeur manquante.

Deux encodages binaires sont fournis :

| Encodage | Résistant | Intermédiaire | Sensible | Manquant |
|---|---:|---:|---:|---:|
| **Principal** (`y_*`) | 1 | `NA` | 0 | `NA` |
| **Non sensible** (`y_ns_*`) | 1 | 1 | 0 | `NA` |

Le choix principal est **conservateur** : un intermédiaire n'est ni un
résistant confirmé, ni un sensible. L'encodage `y_ns_*` correspond à la
définition « non sensible » (R + I) utilisée dans certaines études.

> **Règle importante** : `NA` signifie « non testé » ou « intermédiaire
> exclu ». Il ne faut **jamais** remplacer `NA` par `0` : cela inventerait des
> sensibilités et fausserait l'apprentissage.

Le masque `obs_*` suit l'encodage principal : il vaut `False` pour un
intermédiaire, même si la valeur brute existe. La colonne `n_tested` compte
les intermédiaires, `n_observed` non. La table longue conserve les
intermédiaires avec `raw_label = "Intermediate"` et `phenotype = NA`.

## 1.3 Fichiers produits

Tout est écrit dans `data/processed/ecoli_all_antibiotics/` :

| Fichier | Dimensions | Contenu |
|---|---:|---|
| `ecoli_all_antibiotics_labels.parquet` | 6 983 × 107 | métadonnées + `n_tested`, `n_observed`, `n_resistant`, 50 colonnes `y_*` et 50 colonnes `obs_*` |
| `ecoli_all_antibiotics_labels_ns.parquet` | 6 983 × 55 | métadonnées + 50 colonnes `y_ns_*` (R ou I = 1) |
| `ecoli_all_antibiotics_mask.parquet` | 6 983 × 54 | métadonnées + 50 colonnes `obs_*` (`True` = label binaire R ou S observé) |
| `ecoli_all_antibiotics_long.parquet` | 67 060 × 10 | une ligne par label observé, avec le libellé brut R/S/I |
| `ecoli_all_antibiotics_labels.csv.gz` | 6 983 × 107 | version CSV compressée des labels (interopérabilité) |
| `ecoli_all_antibiotics_long.csv.gz` | 67 060 × 10 | version CSV compressée de la table longue |
| `ecoli_all_antibiotics_ml.parquet` | 5 935 × 20 984 | métadonnées + `split`, `fold_3way`, 50 labels `y_*` et 20 929 features |
| `ecoli_all_antibiotics_long_features.parquet` | 62 666 × 20 939 | fichier unique : une ligne par couple génome × antibiotique, features + `antibiotic`, `resistance` et `split` |
| `ecoli_all_antibiotics_long_features.csv` | 62 666 × 20 939 | même table en CSV (≈2,5 Go, non versionné) |
| `ecoli_all_antibiotics_long_features_index.csv` | 50 lignes | effectifs et prévalence par antibiotique dans le fichier long |
| `ecoli_all_antibiotics_long_features_manifest.json` | — | comptages, schéma et empreintes du fichier long |
| `ecoli_all_antibiotics_drug_summary.csv` | 50 lignes | effectifs, prévalence R et NS, couverture des features |
| `ecoli_all_antibiotics_manifest.json` | — | comptages, contrôles, empreintes SHA-256 |

### Schéma des métadonnées

| Colonne | Description |
|---|---|
| `Genome.Name` | identifiant fonctionnel du génome (clé de jointure avec la matrice de phénotypes) |
| `sra` | accession SRA, clé de jointure avec les features |
| `bvbrc_genome_id` | identifiant BV-BRC (`562.x`), conservé en texte |
| `mlst` | MLST brut BV-BRC (plusieurs allèles possibles, séparés par des virgules) |
| `mlst_group` | groupe MLST nettoyé, présent dans le fichier ML |
| `split` | `train`, `validation` ou `test` (fichier ML uniquement) |
| `fold_3way` | fold 0–5 du split groupé (fichier ML uniquement) |
| `n_tested` | nombre d'antibiotiques testés (R, S ou I) pour ce génome |
| `n_observed` | nombre de labels binaires observés (R ou S, hors I) |
| `n_resistant` | nombre de résistances confirmées |

Les colonnes `y_*` sont en `Int8` nullable : `0`, `1` ou `NA`.
Les colonnes `obs_*` sont booléennes : `True` si le label binaire est observé
(R ou S), `False` si la valeur est `Intermediate` ou manquante. C'est le
masque à utiliser pour la perte.

## 1.4 Comment le jeu a été construit

Le script est
[`src/make_ecoli_all_antibiotics_dataset.py`](../src/make_ecoli_all_antibiotics_dataset.py).

### Étape 1 — Lire et valider la matrice de phénotypes

- lecture en texte pour préserver les identifiants ;
- suppression de l'index de ligne sans nom ;
- vérification de l'unicité de `Genome.Name` ;
- vérification que les 50 colonnes d'antibiotiques sont présentes ;
- vérification que chaque valeur appartient à
  `{Resistant, Susceptible, Intermediate, manquant}`.

Toute anomalie fait échouer le script avec un message explicite.

### Étape 2 — Ajouter les métadonnées

La table `strains_metadata.csv` apporte `Genome.ID`, `SRA.Accession` et
`MLST`. La jointure se fait sur `Genome.Name`. Les 6 983 génomes de la matrice
ont tous des métadonnées.

### Étape 3 — Construire les labels

Pour chaque antibiotique :

```python
y      = raw.map({"Resistant": 1, "Susceptible": 0})                # principal
y_ns   = raw.map({"Resistant": 1, "Intermediate": 1, "Susceptible": 0})
obs    = y.notna()                                                  # masque binaire
tested = raw.notna()                                                # inclut les I
```

Les noms de colonnes sont standardisés en snake_case :

| Libellé source | Colonne |
|---|---|
| `amoxicillin/clavulanic acid` | `amoxicillin_clavulanic_acid` |
| `sulfamethoxazole/trimethoprim` | `sulfamethoxazole_trimethoprim` |
| `nalidixic acid` | `nalidixic_acid` |
| `cefotaxime/clavulanic acid` | `cefotaxime_clavulanic_acid` |

### Étape 4 — Construire la table longue

La matrice est dépilée (`melt`) puis filtrée sur les labels observés. Chaque
ligne contient le libellé brut (`Resistant`, `Susceptible` ou `Intermediate`),
ce qui permet de vérifier ou de changer la définition binaire sans revenir à
la source.

### Étape 5 — Fusionner avec les features

Les features existantes (`ecoli_ciprofloxacin_snp.parquet`) contiennent
5 935 génomes. La fusion se fait sur `Genome.Name`. Deux contrôles sont
effectués :

- les labels ciprofloxacine du nouveau jeu doivent être **identiques** à la
  colonne `y` historique (0 incohérence observée) ;
- les groupes MLST des features et du split doivent être identiques
  (0 incohérence observée).

Le split 3-way existant est ajouté au fichier ML : les colonnes `split` et
`fold_3way` permettent de réutiliser exactement le même découpage groupé par
MLST que pour la ciprofloxacine.

### Étape 6 — Synthèses et manifeste

Le script écrit les statistiques par antibiotique, les comptages, les
contrôles et les empreintes SHA-256 des sources et des sorties.

## 1.5 Statistiques

| Indicateur | Valeur |
|---|---:|
| Génomes avec labels | 6 983 |
| Génomes avec features | 5 935 |
| Antibiotiques | 50 |
| Labels bruts observés (R + S + I) | 67 060 |
| Labels binaires utilisables (R + S) | 65 843 |
| Résistants | 11 866 |
| Sensibles | 53 977 |
| Intermédiaires (exclus de l'encodage principal) | 1 217 |
| Cellules manquantes (génome × antibiotique) | 282 090 |
| Antibiotiques testés par génome (moyenne) | 9,60 |
| Antibiotiques testés par génome (médiane) | 10 |
| Split train / validation / test | 4 074 / 932 / 929 |

Le fichier `ecoli_all_antibiotics_ml.parquet` contient 62 666 labels binaires :
les 1 048 génomes sans features sont absents et les intermédiaires sont exclus
de l'encodage principal.

Le tableau complet par antibiotique est dans
`ecoli_all_antibiotics_drug_summary.csv`. Les lignes les plus observées sont :

| Antibiotique | Classe | Observés | R | S | I | Manquants | Prévalence R | Prévalence NS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gentamicin | aminoglycoside | 6 043 | 594 | 5 330 | 119 | 940 | 9,8 % | 11,8 % |
| ciprofloxacin | fluoroquinolone | 6 039 | 1 291 | 4 657 | 91 | 944 | 21,4 % | 22,9 % |
| ceftazidime | bêta-lactamine (céphalosporine 3e gén.) | 5 981 | 563 | 5 195 | 223 | 1 002 | 9,4 % | 13,1 % |
| piperacillin/tazobactam | bêta-lactamine / inhibiteur | 5 808 | 248 | 5 382 | 178 | 1 175 | 4,3 % | 7,3 % |
| cefotaxime | bêta-lactamine (céphalosporine 3e gén.) | 5 415 | 789 | 4 587 | 39 | 1 568 | 14,6 % | 15,3 % |
| ampicillin | bêta-lactamine | 4 874 | 2 440 | 2 430 | 4 | 2 109 | 50,1 % | 50,1 % |
| meropenem | carbapénème | 4 665 | 35 | 4 622 | 8 | 2 318 | 0,8 % | 0,9 % |
| cefuroxime | bêta-lactamine (céphalosporine 2e gén.) | 4 115 | 545 | 3 559 | 11 | 2 868 | 13,2 % | 13,5 % |
| amoxicillin/clavulanic acid | bêta-lactamine / inhibiteur | 3 779 | 1 122 | 2 598 | 59 | 3 204 | 29,7 % | 31,3 % |
| tigecycline | tétracycline (glycylcycline) | 3 713 | 3 | 3 700 | 10 | 3 270 | 0,1 % | 0,4 % |
| trimethoprim/sulfamethoxazole | diaminopyrimidine / sulfamide | 3 424 | 1 026 | 2 360 | 38 | 3 559 | 30,0 % | 31,1 % |

Les antibiotiques très peu testés (par exemple `cefalexin`, `sulbactam`,
`cefotaxime/clavulanic acid` : 1 observation) ne doivent pas être utilisés
comme tâche principale. Ils peuvent être conservés dans le jeu pour un modèle
multi-tâches avec perte masquée, mais leurs métriques seront instables.

## 1.6 Utilisation

### Charger les labels

```python
import pandas as pd
from src.data_loader import load_ecoli_antibiotics_labels

labels = load_ecoli_antibiotics_labels()
y_columns = [c for c in labels.columns if c.startswith("y_")]
obs_columns = [c for c in labels.columns if c.startswith("obs_")]

Y = labels[y_columns]          # Int8 : 0, 1 ou NA
M = labels[obs_columns]        # booléen : label binaire observé (R ou S) ?
```

### Charger le jeu prêt pour l'apprentissage

```python
from src.data_loader import load_ecoli_antibiotics_ml

df = load_ecoli_antibiotics_ml()
y_columns = [c for c in df.columns if c.startswith("y_")]
feature_columns = [
    c for c in df.columns
    if c not in {"sra", "Genome.Name", "mlst_group", "split", "fold_3way", *y_columns}
]

X = df[feature_columns]
Y = df[y_columns]
mask = Y.notna()

train = df["split"] == "train"
val = df["split"] == "validation"
test = df["split"] == "test"
```

### Charger le fichier unique « génome × antibiotique »

Le fichier `ecoli_all_antibiotics_long_features.parquet` rassemble tout dans une
seule table : chaque ligne correspond à un couple génome × antibiotique observé,
avec les features et des colonnes explicites `antibiotic` et `resistance`.

```python
from src.data_loader import load_ecoli_antibiotics_long_features

tout = load_ecoli_antibiotics_long_features()          # 62 666 lignes
cipro = load_ecoli_antibiotics_long_features("ciprofloxacin")

# Colonnes de métadonnées et de label
id_cols = [
    "Genome.Name", "sra", "mlst_group", "split", "fold_3way",
    "antibiotic_label", "antibiotic", "drug_class", "resistance", "raw_label",
]
X = cipro.drop(columns=id_cols)
y = cipro["resistance"]
```

> **Attention** : les 20 929 features sont dupliquées pour chaque antibiotique
> du même génome. Pour entraîner un modèle multi-tâches, préférez
> `ecoli_all_antibiotics_ml.parquet` (une ligne par génome, 50 colonnes `y_*`).
> Le fichier long est pratique pour filtrer un antibiotique, faire des
> statistiques ou joindre les labels à d'autres features.

### Réutiliser le split groupé par MLST

Le fichier ML contient déjà `split` et `fold_3way`, repris du split 3-way de
la ciprofloxacine. Les 5 935 génomes sont couverts :

| Split | Génomes |
|---|---:|
| train | 4 074 |
| validation | 932 |
| test | 929 |

Pour une comparaison honnête avec les résultats ciprofloxacine déjà obtenus,
utilisez ces colonnes plutôt qu'un split aléatoire.

### Exemple de perte masquée pour un modèle multi-tâches

```python
import torch
from torch import nn

class MaskedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, mask):
        # targets et mask : mêmes dimensions que logits
        raw = self.loss(logits, targets.float())
        raw = raw * mask.float()
        return raw.sum() / mask.float().sum().clamp_min(1.0)
```

Le masque doit être calculé **par lot** : les antibiotiques non testés pour
les génomes du lot ne contribuent pas à la perte.

## 1.7 Limites du jeu *E. coli*

- **Features manquantes pour 1 048 génomes** : ils ont des labels mais pas de
  features pangénome/SNP. Ils sont conservés dans les fichiers de labels et la
  table longue, pas dans le fichier ML.
- **Labels très creux** : seuls 67 060 des 349 150 couples
  génome × antibiotique ont une valeur brute (19,2 %), et 65 843 ont un label
  binaire utilisable (18,9 %). Certains antibiotiques ont moins de
  10 observations.
- **Trois définitions possibles de la résistance** : R seul (principal),
  R + I (non sensible), ou exclusion des intermédiaires. Le choix doit être
  documenté et constant entre train et test.
- **Pas de phylogénie fine** : le split est groupé par MLST, pas par clone
  phylogénétique. Les folds phylogénétiques de
  `data/raw/Supplemental_data/` peuvent être utilisés pour une évaluation plus
  stricte lorsque la cohorte le permet.
- **Pas de valeur clinique directe** : les labels sont des phénotypes de
  laboratoire agrégés, pas une décision thérapeutique.

---

# 2. Jeu complémentaire : multi-espèces, 80 cohortes

## 2.1 Sources

Le dossier `data/raw/genome_id/` contient un fichier par couple
(espèce, antibiotique) :

```text
Data_<Espèce>_<antibiotique>.txt
```

Chaque fichier a deux colonnes : `genome_id` et `resistant_phenotype`
(0 = sensible, 1 = résistant).

| Élément | Valeur |
|---|---:|
| Fichiers source | 80 |
| Espèces | 11 |
| Cohortes espèce × antibiotique | 80 |
| Libellés d'antibiotiques bruts | 46 |
| Antibiotiques après standardisation | 44 |
| Observations (génome × antibiotique) | 146 159 |
| Génomes uniques | 31 704 |
| Résistants | 51 399 (35,17 %) |
| Sensibles | 94 760 (64,83 %) |

### Pourquoi 46 libellés bruts mais 44 antibiotiques standardisés ?

| Nom brut | Nom standardisé | Raison |
|---|---|---|
| `ethiomide` | `ethionamide` | faute de frappe |
| `cotrimoxazole` | `trimethoprim_sulfamethoxazole` | synonyme de l'association |
| `beta-lactam` | `beta_lactam` | normalisation du tiret |

Les cohortes brutes restent distinctes dans `antibiotic` et `cohort` ; elles
ne sont fusionnées que dans `antibiotic_std`. `beta_lactam` est un libellé de
**classe**, pas une molécule unique.

## 2.2 Fichiers produits

Tout est écrit dans `data/processed/amr_multispecies/` :

| Fichier | Contenu |
|---|---|
| `amr_long.parquet` | table longue, format principal |
| `amr_long.csv.gz` | même table en CSV compressé |
| `amr_wide.parquet` | 31 704 génomes × 44 antibiotiques (`Int8`, `NA` = non testé) |
| `amr_by_species/<espèce>.parquet` | sous-matrice large par espèce |
| `amr_drug_summary.csv` | statistiques par couple espèce × antibiotique |
| `amr_species_summary.csv` | statistiques par espèce |
| `amr_antibiotic_mapping.csv` | correspondance brut → standardisé → classe |
| `amr_manifest.json` | comptages, contrôles, empreintes SHA-256 |

### Schéma de la table longue

| Colonne | Type | Description |
|---|---|---|
| `genome_id` | `string` | identifiant BV-BRC, conservé en texte |
| `species` | `string` | espèce bactérienne |
| `antibiotic` | `string` | nom brut de l'antibiotique |
| `antibiotic_std` | `string` | nom standardisé en snake_case |
| `drug_class` | `string` | classe pharmacologique |
| `phenotype` | `int8` | 0 = sensible, 1 = résistant |
| `cohort` | `string` | `espèce::antibiotique_brut` |
| `source_file` | `string` | fichier d'origine |
| `source_row` | `int32` | ligne d'origine (1-based) |

## 2.3 Construction

Le script est
[`src/make_multispecies_antibiotic_dataset.py`](../src/make_multispecies_antibiotic_dataset.py).

1. **Découpage des noms de fichiers** à partir de la liste des 11 espèces
   (les noms d'antibiotiques contiennent eux-mêmes des `_`).
2. **Lecture en texte** des identifiants pour ne pas perdre les zéros finaux.
3. **Validation** : pas de valeur manquante, phénotype dans `{0, 1}`, pas de
   `genome_id` dupliqué dans un fichier.
4. **Standardisation** des noms d'antibiotiques et ajout de la classe
   pharmacologique.
5. **Empilement** des 80 tables, puis contrôles globaux :
   - unicité de `(cohort, genome_id)` ;
   - absence de conflit après fusion des synonymes ;
   - absence de valeurs manquantes.
6. **Pivot** vers la matrice large, avec `NA` pour les combinaisons non
   testées.
7. **Synthèses** et **manifeste**.

## 2.4 Statistiques par espèce

| Espèce | Observations | Génomes | Cohortes | Antibiotiques bruts | Antibiotiques std | Prévalence |
|---|---:|---:|---:|---:|---:|---:|
| *Acinetobacter baumannii* | 5 309 | 1 144 | 7 | 7 | 7 | 65,3 % |
| *Campylobacter jejuni* | 395 | 395 | 1 | 1 | 1 | 68,9 % |
| *Enterococcus faecium* | 277 | 277 | 1 | 1 | 1 | 48,4 % |
| *Escherichia coli* | 15 942 | 2 493 | 13 | 13 | 13 | 27,9 % |
| *Klebsiella pneumoniae* | 19 107 | 1 761 | 13 | 13 | 13 | 59,8 % |
| *Mycobacterium tuberculosis* | 52 371 | 13 550 | 11 | 11 | 10 | 26,2 % |
| *Neisseria gonorrhoeae* | 825 | 680 | 2 | 2 | 2 | 39,6 % |
| *Pseudomonas aeruginosa* | 2 930 | 891 | 5 | 5 | 5 | 44,8 % |
| *Salmonella enterica* | 18 408 | 1 922 | 11 | 11 | 11 | 24,7 % |
| *Staphylococcus aureus* | 15 340 | 3 325 | 9 | 9 | 9 | 35,4 % |
| *Streptococcus pneumoniae* | 15 255 | 5 266 | 7 | 7 | 6 | 41,4 % |

Les prévalences varient de 24,7 % à 68,9 % : l'accuracy seule est trompeuse.

## 2.5 Utilisation

```python
import pandas as pd
from src.data_loader import (
    load_multispecies_antibiotics_long,
    load_multispecies_antibiotics_wide,
)

long = load_multispecies_antibiotics_long()
ecoli = long.query(
    "species == 'Escherichia_coli' and antibiotic_std == 'ciprofloxacin'"
)

wide = load_multispecies_antibiotics_wide()
ecoli_matrix = load_multispecies_antibiotics_wide(species="Escherichia_coli")
```

La matrice large est adaptée à un modèle multi-étiquettes par espèce. La table
longue est adaptée aux jointures avec des features et aux modèles par cohorte.

## 2.6 Limites du jeu multi-espèces

- **Pas de features génomiques** pour les 10 espèces autres que *E. coli*.
- **Cohortes non emboîtées** : les fichiers par antibiotique ne décrivent pas
  toujours les mêmes isolats. Ce n'est pas un multi-étiquettes complet.
- **Pas de split prêt à l'emploi** : le découpage dépend des features
  disponibles.
- **Synonymes fusionnés** : `ethiomide`/`ethionamide` et
  `cotrimoxazole`/`trimethoprim_sulfamethoxazole` restent des cohortes
  distinctes dans `cohort`.
- **Identifiants en texte** : ne pas convertir `genome_id` en float pour
  l'affichage.
- **E. coli multi-espèces ≠ E. coli 50 antibiotiques** : le sous-ensemble
  *E. coli* des 80 fichiers (13 cohortes) est distinct du panel 50
  antibiotiques. Ne concaténez pas les deux sans vérifier les identifiants,
  les cohortes et les définitions de phénotype.

---

# 3. Protocole recommandé pour le multi-antibiotiques

## 3.1 Quel jeu pour quelle question ?

| Question | Jeu recommandé |
|---|---|
| Étendre la ciprofloxacine aux 50 antibiotiques *E. coli* | **jeu E. coli 50** |
| Comparer les performances entre antibiotiques | jeu E. coli 50 (colonnes `y_*`) |
| Tester un modèle multi-tâches avec features | fichier `ecoli_all_antibiotics_ml.parquet` |
| Étudier plusieurs espèces | jeu multi-espèces |
| Faire du transfert entre espèces | jeu multi-espèces, avec précautions |

## 3.2 Découpage sans fuite

- Un génome ne doit jamais être à la fois dans le train et dans le test.
- Pour les 5 935 génomes *E. coli* avec features, utilisez les colonnes
  `split` et `fold_3way` du fichier ML : elles reprennent le split groupé par
  MLST déjà utilisé pour la ciprofloxacine.
- Si vous construisez un split multi-espèces, groupez par `genome_id` et, si
  possible, par lignée clonale.
- Ne mélangez pas les lignes d'un même génome entre plusieurs folds.
- Attention aux cohortes qui se recouvrent : un génome peut apparaître dans
  plusieurs fichiers d'antibiotiques. Vérifiez les doublons avant de
  concaténer.

## 3.3 Métriques

Avec des prévalences de 0,1 % à 98 %, l'accuracy est inutilisable seule.
Rapportez au minimum :

- PR-AUC (average precision) ;
- recall, précision, F1 ;
- balanced accuracy ;
- calibration (courbe de fiabilité, Brier score) ;
- matrice de confusion.

Donnez toujours la prévalence et le nombre d'observations à côté de la
métrique. Pour les antibiotiques avec moins de 30 observations, les métriques
sont trop instables pour conclure : rapportez-les comme descriptives.

## 3.4 Modèle multi-tâches

- Utilisez une sortie par antibiotique.
- Calculez la perte uniquement sur les labels observés (masque).
- Ne remplacez jamais `NA` par `0`.
- Envisagez des poids de classe ou un seuil décisionnel par antibiotique.
- Commencez par des modèles simples (régression logistique, arbres) sur les
  antibiotiques bien observés, puis étendez.

## 3.5 Stratégie de départ conseillée

1. **Ciprofloxacine** : reproduire les résultats existants avec
   `y_ciprofloxacin` et le split `fold_3way`.
2. **Antibiotiques bien observés** : ajouter progressivement `gentamicin`,
   `ceftazidime`, `piperacillin/tazobactam`, `cefotaxime`, `ampicillin`.
3. **Multi-tâches** : entraîner un modèle à 50 sorties avec perte masquée,
   puis comparer aux modèles par antibiotique.
4. **Antibiotiques rares** : ne les inclure que dans une analyse descriptive
   ou avec une régularisation forte.

---

# 4. Régénération

Depuis la racine du projet :

```bash
# Jeu principal E. coli 50 antibiotiques
python src/make_ecoli_all_antibiotics_dataset.py

# Jeu complémentaire multi-espèces
python src/make_multispecies_antibiotic_dataset.py
```

Options utiles :

```bash
# E. coli : choisir les sources et la sortie
python src/make_ecoli_all_antibiotics_dataset.py \
  --phenotypes data/raw/Ecoli_AMR_GenotypePhenotype/phenotype_matrix/phenotype_matrix_08302024.csv \
  --metadata data/raw/Ecoli_AMR_GenotypePhenotype/strains_metadata.csv \
  --features data/processed/ecoli_ciprofloxacin_snp.parquet \
  --splits data/splits_snp_3way \
  --output-dir data/processed/ecoli_all_antibiotics

# E. coli : Parquet uniquement, sans CSV compressé
python src/make_ecoli_all_antibiotics_dataset.py --no-csv

# E. coli : fichier unique long (une ligne par génome × antibiotique)
python src/make_ecoli_long_features_dataset.py
python src/make_ecoli_long_features_dataset.py --min-observations 30

# Export CSV du fichier long (≈2,5 Go)
python src/export_ecoli_long_features_csv.py

# Multi-espèces : Parquet uniquement, sans CSV compressé
python src/make_multispecies_antibiotic_dataset.py --no-csv
```

Les scripts sont déterministes pour les tables produites. Le manifeste
contient un horodatage, mais les données et les empreintes des sources
permettent de vérifier la reproductibilité.

---

# 5. Arborescence des sorties

```text
data/processed/
├── ecoli_all_antibiotics/
│   ├── ecoli_all_antibiotics_labels.parquet
│   ├── ecoli_all_antibiotics_labels_ns.parquet
│   ├── ecoli_all_antibiotics_mask.parquet
│   ├── ecoli_all_antibiotics_long.parquet
│   ├── ecoli_all_antibiotics_labels.csv.gz
│   ├── ecoli_all_antibiotics_long.csv.gz
│   ├── ecoli_all_antibiotics_ml.parquet
│   ├── ecoli_all_antibiotics_drug_summary.csv
│   └── ecoli_all_antibiotics_manifest.json
└── amr_multispecies/
    ├── amr_long.parquet
    ├── amr_long.csv.gz
    ├── amr_wide.parquet
    ├── amr_by_species/
    │   ├── Acinetobacter_baumannii.parquet
    │   ├── Campylobacter_jejuni.parquet
    │   ├── Enterococcus_faecium.parquet
    │   ├── Escherichia_coli.parquet
    │   ├── Klebsiella_pneumoniae.parquet
    │   ├── Mycobacterium_tuberculosis.parquet
    │   ├── Neisseria_gonorrhoeae.parquet
    │   ├── Pseudomonas_aeruginosa.parquet
    │   ├── Salmonella_enterica.parquet
    │   ├── Staphylococcus_aureus.parquet
    │   └── Streptococcus_pneumoniae.parquet
    ├── amr_drug_summary.csv
    ├── amr_species_summary.csv
    ├── amr_antibiotic_mapping.csv
    └── amr_manifest.json
```

---

# 6. Résumé

- **Jeu principal** : *E. coli*, 6 983 génomes, 50 antibiotiques,
  67 060 labels bruts observés (dont 65 843 binaires R/S), 5 935 génomes avec
  20 929 features génomiques et le split 3-way groupé par MLST. Fichier prêt à
  l'emploi : `ecoli_all_antibiotics_ml.parquet`.
- **Jeu complémentaire** : 80 cohortes, 11 espèces, 44 antibiotiques
  standardisés, 146 159 observations phénotypiques, sans features. Fichier
  principal : `amr_multispecies/amr_long.parquet`.
- **Format** : Parquet pour l'usage ML (types préservés, `NA` explicites),
  CSV compressé pour l'interopérabilité, tableaux de synthèse et manifeste
  pour la traçabilité.
- **Règle d'or** : `NA` ≠ sensible ; un génome ne doit jamais fuiter entre
  train et test ; les métriques doivent être interprétées avec la prévalence.
