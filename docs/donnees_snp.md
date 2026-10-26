# Jeu de données enrichi avec les SNP QRDR

## Objectif

Le fichier `ecoli_ciprofloxacin.csv` ne contient que la **présence/absence de
gènes et de clusters pan-génomiques variables**. Les gènes core, dont `gyrA`,
`gyrB`, `parC` et `parE`, sont présents chez toutes les souches et ont donc été
filtrés comme invariants. Les mutations ponctuelles responsables de la résistance
aux fluoroquinolones (par exemple `gyrA` Ser83Leu, `gyrA` Asp87Asn, `parC`
Ser80Ile) ne sont donc pas représentées.

Le jeu enrichi ajoute uniquement les **SNP non-silencieux des gènes QRDR et des
régulateurs associés**, extraits du VCF de variant calling pangénomique :

| Contig pangénome | Gène |
|---|---|
| `LMHECDEF_04343` | `gyrA` |
| `LMHECDEF_04466` | `gyrB` |
| `LMHECDEF_01124` | `parC` |
| `LMHECDEF_01105` | `parE` |
| `LMHECDEF_00662` | `marR` |
| `LMHECDEF_00326` | `acrR_1` |
| `LMHECDEF_00443` | `acrR_2` |
| `LMHECDEF_03751` | `soxR` |

Les 2,4 millions de variants du pangénome ne sont **pas** ajoutés : la matrice
serait énorme, creuse, et la majorité des variants sont des marqueurs de
lignage plutôt que des déterminants de résistance.

## Source et sélection

- VCF source : `output.non_silent.vcf.gz` de l'archive Zenodo
  `10.5281/zenodo.12692732`.
- Extraction des contigs cibles via `grep`, puis annotation `snpEff` déjà
  présente dans le VCF (`INFO/ANN`).
- Les sites **multi-alléliques** sont éclatés : chaque allèle alternatif (`ALT`)
  devient une feature distincte avec sa propre annotation protéique.
- Les allèles `synonymous_variant` sont exclus.
- Filtre de fréquence : seuls les variants présents chez **au moins 5 isolats**
  sont conservés (`--min-carriers 5`), soit **198 features SNP**.

## Encodage

Chaque variant devient une colonne binaire :

- `0` : allèle de référence / pas de variant détecté ;
- `1` : allèle alternatif détecté.

> Important : le VCF est un VCF de **présence de variants**. Le génotype `.`
> signifie « pas de variant détecté » (référence), et non « donnée manquante ».
> Il est donc encodé `0`.

Les noms de colonnes suivent le format :

```text
SNP_<gène>_<modification protéique>
```

Exemples :

```text
SNP_gyrA_Ser83Leu
SNP_gyrA_Asp87Asn
SNP_parC_Ser80Ile
SNP_parC_Glu84Val
```

En cas de doublon (même gène et même modification protéique à des positions ou
pour des allèles différents), la position et l'allèle sont ajoutés :

```text
SNP_parC_Ser80Arg_238_C
SNP_parC_Ser80Arg
```

## Structure finale

Le fichier conserve **exactement les mêmes 5 935 lignes** et les mêmes
identifiants que le CSV d'origine. Les colonnes sont organisées ainsi :

```text
sra | Genome.Name | mlst_group | y | <20 731 features pangénome> | <198 features SNP>
```

Soit **20 933 colonnes** au total.

Exemple simplifié :

| sra | Genome.Name | mlst_group | y | LMHECDEF_00001 | ... | SNP_gyrA_Ser83Leu | SNP_parC_Ser80Ile |
|---|---|---|---:|---:|---|---:|---:|
| ERR4035565 | Escherichia coli ... | ST131 | 1 | 1 | ... | 1 | 0 |
| ERR4035569 | Escherichia coli ... | ST73 | 0 | 0 | ... | 0 | 0 |

Les colonnes de métadonnées et la cible `y` sont inchangées. Les features
pangénome sont également inchangées ; les SNP sont ajoutés **à la fin**.

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `data/processed/ecoli_ciprofloxacin_snp.csv` | CSV complet enrichi (249,8 Mo) |
| `data/processed/ecoli_ciprofloxacin_snp.parquet` | Version Parquet (31,7 Mo), features SNP en `uint8` |
| `data/processed/ecoli_ciprofloxacin_snp.manifest.json` | Nombre de SNP, liste et détails des features, échantillons manquants |
| `data/splits_snp/X_train.parquet` | 4 748 x 20 929 features (uint8) |
| `data/splits_snp/X_test.parquet` | 1 187 x 20 929 features (uint8) |
| `data/splits_snp/y_train.parquet`, `y_test.parquet` | Cibles (0/1) |
| `data/splits_snp/metadata_train.parquet`, `metadata_test.parquet` | Métadonnées et folds |

Le split train/test est **identique** au split d'origine afin de comparer, à
découpage constant :

1. pangénome seul (`dataset="base"`) ;
2. pangénome + SNP QRDR (`dataset="snp"`).

Cela ne corrige pas le biais de lignage du split aléatoire. Pour une évaluation
sans fuite de clones, il faudra ensuite un split groupé par `mlst_group` ou les
folds phylogénétiques fournis dans `data/raw/Supplemental_data/`.

Un split 3-way groupé par MLST est déjà disponible dans `data/splits_snp_3way/`
et documenté dans `docs/split_3way.md` :

```python
from src.data_loader import load_train, load_validation, load_test

X_train, y_train, meta_train = load_train(dataset="snp_3way")
X_val, y_val, meta_val = load_validation(dataset="snp_3way")
X_test, y_test, meta_test = load_test(dataset="snp_3way")
```

## Utilisation

```python
from src.data_loader import load_train, load_test

# Pangénome seul
X_train, y_train, meta_train = load_train(dataset="base")

# Pangénome + SNP QRDR
X_train_snp, y_train_snp, meta_train_snp = load_train(dataset="snp")
X_test_snp, y_test_snp, meta_test_snp = load_test(dataset="snp")
```

## Validation

Le script d'ajout vérifie et journalise :

- le nombre de lignes et de colonnes ;
- le nombre de features SNP ;
- les échantillons du CSV absents du VCF ;
- les échantillons du VCF absents du CSV ;
- le nombre de porteurs par variant (`snp_details` dans le manifeste).

Contrôles effectués après génération :

```python
df = pd.read_parquet("data/processed/ecoli_ciprofloxacin_snp.parquet")
assert df.shape == (5935, 20933)
assert df["sra"].is_unique
assert not df.isna().any().any()
assert set(df["y"].unique()) == {0, 1}
```

## Mutations connues retrouvées

| Feature | Porteurs |
|---|---:|
| `SNP_gyrA_Ser83Leu` | 1 548 |
| `SNP_gyrA_Asp87Asn` | 1 210 |
| `SNP_parC_Ser80Ile` | 1 223 |
| `SNP_parC_Glu84Val` | 536 |
| `SNP_gyrA_Asp87Tyr` | 20 |
| `SNP_gyrA_Asp87Gly` | 16 |
| `SNP_gyrA_Asp87His` | 10 |
| `SNP_parC_Glu84Lys` | 9 |

## Limites

- Seuls les variants **non-silencieux** des gènes ciblés sont ajoutés.
- Beaucoup de variants fréquents dans ces gènes sont des **marqueurs de
  lignage** (par ex. `SNP_soxR_Gly74Arg` : 4 494 porteurs, `SNP_acrR_1_Glu75Asp`
  : 4 391, `SNP_parC_Asp475Glu` : 3 911, `SNP_parE_Phe506Leu` : 3 807) et non
  des déterminants directs de résistance. Une sélection de features ou une
  régularisation reste nécessaire.
- Les génotypes non détectés sont encodés `0` ; ils peuvent masquer un défaut de
  couverture. Le manifeste permet de vérifier les variants concernés.
- Le split reste aléatoire : les performances restent optimistes tant que la
  fuite de lignage n'est pas corrigée.
- L'ajout de SNP ne remplace pas une validation biologique des mutations
  détectées (notamment la nomenclature protéique issue de snpEff).
