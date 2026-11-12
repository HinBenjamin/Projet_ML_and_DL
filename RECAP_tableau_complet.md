# Récapitulatif — `tableau_complet.parquet`

> Table complète (features + phénotypes) du jeu **_E. coli_ multi-antibiotiques**.
> Chemin : `data/processed/test/tableau_complet.parquet`

## Vue d'ensemble

| Élément | Valeur |
|---|---:|
| Lignes (génomes) | **6,949** |
| Colonnes | **34,796** |
| Taille fichier | 45.1 Mo |
| Taille en RAM | ~0.24 Go |
| Valeurs manquantes (`NaN`) | 0 |
| Types | `int8` pour les features/labels, `object` pour `souche`/`lignee`/`etude`, `int64` pour `bloc` |

## Structure des colonnes

| Groupe | Nb colonnes | Encodage | Densité moyenne de `1` |
|---|---:|---|---:|
| Identifiants | 4 | texte / entier | — |
| Phénotypes antibiotiques | 48 | `1` = R, `0` = S, `-1` = NA | — |
| Mutations / SNP | 2,303 | `0` / `1` | 4.3 % |
| Présence gènes / locus | 18,494 | `0` / `1` | 36.7 % |
| Clusters pangénomiques | 13,947 | `0` / `1` | 36.3 % |
| **Total features** | **34,792** | | |

### Identifiants

- **`souche`** — accession du génome (6,949 valeurs uniques, 1 ligne = 1 génome).
- **`lignee`** — lignage MLST (456 valeurs, dont `inconnu`) ; utilisé pour les splits groupés.
- **`etude`** — étude source (26 études, ex. `sans référence | Norway`, `28720578 | United Kingdom`).
- **`bloc`** — fold de split (`-1` = lignée inconnue, puis `0` à `4`).

### Phénotypes

- **48 antibiotiques**, de `amikacin` à `trimethoprim/sulfamethoxazole`.
- `1` = résistant, `0` = sensible, `-1` = non testé / non disponible.
- Beaucoup de `-1` : la couverture varie fortement d'un antibiotique à l'autre.

### Features génomiques

- **Mutations / SNP** : noms lisibles, ex. `lon_2_Ala58Ser`, `gyrA_<mutation>`, `parC_<mutation>`.
- **Gènes / locus** : présence binaire, ex. `LMHECDEF_00001`, `PNGICKPE_04154`.
- **Clusters pangénomiques** : présence binaire, ex. `Cluster_13961_+_+_GCA_003335465.1_..._CO_F`.

## Top 15 antibiotiques par nombre de résistants

| Antibiotique | R | S | NA | % R parmi testés |
|---|---:|---:|---:|---:|
| ampicillin | 2,429 | 2,422 | 2,098 | 50.1 % |
| ciprofloxacin | 1,285 | 4,650 | 1,014 | 21.7 % |
| amoxicillin/clavulanic acid | 1,119 | 2,591 | 3,239 | 30.2 % |
| trimethoprim/sulfamethoxazole | 1,021 | 2,353 | 3,575 | 30.3 % |
| cefotaxime | 784 | 4,584 | 1,581 | 14.6 % |
| amoxicillin | 664 | 467 | 5,818 | 58.7 % |
| gentamicin | 589 | 5,321 | 1,039 | 10.0 % |
| ceftazidime | 556 | 5,189 | 1,204 | 9.7 % |
| cefuroxime | 543 | 3,558 | 2,848 | 13.2 % |
| trimethoprim | 340 | 358 | 6,251 | 48.7 % |
| piperacillin/tazobactam | 246 | 5,371 | 1,332 | 4.4 % |
| tetracycline | 237 | 151 | 6,561 | 61.1 % |
| ceftriaxone | 219 | 215 | 6,515 | 50.5 % |
| tobramycin | 201 | 815 | 5,933 | 19.8 % |
| cefalotin | 194 | 173 | 6,582 | 52.9 % |

## Mutations les plus fréquentes

| Mutation | Génomes porteurs |
|---|---:|
| `ampC_Pro210Ala` | 6,527 |
| `acrB_2_Asp740Ala` | 6,340 |
| `phoP_Ile44Leu` | 6,163 |
| `ampC_His312Arg` | 6,095 |
| `dacB_Gln182Pro` | 6,057 |
| `basS_Asn358Tyr` | 5,934 |
| `rob_2_Gly269Ser` | 5,927 |
| `rob_2_Ile240Met` | 5,917 |
| `rob_2_Ile237Leu` | 5,900 |
| `rob_2_Ile131Val` | 5,808 |

## Remarques

- Malgré le dossier `test/`, le fichier contient **tous les génomes** (6,949) : c'est la matrice complète `X + y` fusionnée, et non le seul split de test.
- Le README mentionne 50 antibiotiques ; cette version en contient **48**.
- Toutes les lignes possèdent au moins une feature génomique.
- Fichier chargé notamment dans `notebooks/exploration_benjamin.ipynb`.
