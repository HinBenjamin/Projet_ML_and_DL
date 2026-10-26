# Split train / validation / test groupé par MLST

## Pourquoi un split groupé ?

Le split aléatoire d'origine plaçait 95,8 % des isolats de test dans un groupe
MLST déjà présent dans le train. Le modèle pouvait donc apprendre des marqueurs
de clones plutôt que des mécanismes de résistance, ce qui gonflait les
performances.

Le split 3-way ci-dessous garde les groupes MLST **intacts** : un groupe MLST
connu n'apparaît que dans un seul split (train, validation ou test).

## Méthode

- `StratifiedGroupKFold` à **6 folds**, stratifié sur `y` ;
- **4 folds** pour le train, **1 fold** pour la validation, **1 fold** pour le test ;
- le couple (validation, test) est choisi pour se rapprocher de 70/15/15 et du
  taux de résistance global ;
- les isolats `mlst_group == "unknown"` sont traités comme des groupes
  individuels (un groupe par `sra`), faute d'information de clonalité ;
- graine aléatoire : `42`.

## Résultat

| Split | Isolats | Résistants | Taux de résistance | Groupes MLST connus |
|---|---:|---:|---:|---:|
| Train | 4 074 | 975 | 23,93 % | 268 |
| Validation | 932 | 156 | 16,74 % | 79 |
| Test | 929 | 154 | 16,58 % | 75 |
| **Total** | **5 935** | **1 285** | **21,65 %** | **422** |

Contrôles effectués :

- aucun `sra` en double entre les splits ;
- aucun groupe MLST connu présent dans deux splits ;
- 5 935 isolats répartis au total ;
- 20 929 features par split (20 731 pangénome + 198 SNP).

> Le taux de résistance est plus élevé dans le train (23,9 %) que dans la
> validation et le test (≈16,6 %). C'est une conséquence directe du respect des
> groupes : les grands groupes MLST très résistants (ST131, ST1193, etc.) ne
> peuvent pas être répartis entre les splits. Il faut donc interpréter les
> résultats avec des métriques adaptées au déséquilibre : **PR-AUC, recall,
> F1, balanced accuracy**, et non l'accuracy seule.

## Fichiers produits

| Fichier | Contenu |
|---|---|
| `data/splits_snp_3way/X_train.parquet` | 4 074 × 20 929 (`uint8`) |
| `data/splits_snp_3way/X_val.parquet` | 932 × 20 929 (`uint8`) |
| `data/splits_snp_3way/X_test.parquet` | 929 × 20 929 (`uint8`) |
| `data/splits_snp_3way/y_train.parquet` | Cible train |
| `data/splits_snp_3way/y_val.parquet` | Cible validation |
| `data/splits_snp_3way/y_test.parquet` | Cible test |
| `data/splits_snp_3way/metadata_train.parquet` | `sra`, `Genome.Name`, `mlst_group`, `y`, `split`, `fold_3way` |
| `data/splits_snp_3way/metadata_val.parquet` | Idem validation |
| `data/splits_snp_3way/metadata_test.parquet` | Idem test |
| `data/splits_snp_3way/split_summary.json` | Résumé, folds choisis, statistiques |

## Utilisation

```python
from src.data_loader import load_train, load_validation, load_test

X_train, y_train, meta_train = load_train(dataset="snp_3way")
X_val, y_val, meta_val = load_validation(dataset="snp_3way")
X_test, y_test, meta_test = load_test(dataset="snp_3way")
```

Le split 2-way précédent reste disponible :

```python
X_train, y_train, meta_train = load_train(dataset="snp")
X_test, y_test, meta_test = load_test(dataset="snp")
```

## Régénération

```bash
python src/make_3way_splits_snp.py
```

Options utiles :

```bash
python src/make_3way_splits_snp.py --n-splits 6 --seed 42
```

## Limites

- Le split est groupé par MLST, mais **pas par phylogenie fine** : deux clones
  très proches peuvent avoir des MLST différents. Pour une évaluation plus
  stricte, utiliser les folds phylogénétiques de
  `data/raw/Supplemental_data/`.
- Les isolats `unknown` sont répartis individuellement ; on ne peut pas garantir
  qu'ils ne sont pas clonaux.
- Le déséquilibre train/validation/test est inhérent aux grands groupes MLST.
  Il est préférable de ne pas rééquilibrer artificiellement le test.
- Ce split ne remplace pas le split 2-way pour comparer des résultats déjà
  obtenus : il définit un **nouveau protocole d'évaluation** plus honnête.
