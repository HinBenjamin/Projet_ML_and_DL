#!/usr/bin/env python3
"""Crée un split train / validation / test groupé par lignée pour le jeu 48 antibiotiques.

Le fichier ``ecoli_all_snp.parquet`` contient 6 949 génomes, 4 identifiants
(``souche``, ``lignee``, ``etude``, ``bloc``), 48 phénotypes (-1 = non testé)
et 34 744 features pangénome + SNP.

Stratégie (identique à ``make_3way_splits_snp.py``, adaptée au multi-label) :

- ``StratifiedGroupKFold`` à 6 folds, groupes = ``lignee`` ;
- la stratification porte sur la **charge de résistance** (nombre
  d'antibiotiques résistants par génome, en classes) plutôt que sur un seul
  antibiotique, afin d'équilibrer les 48 phénotypes ;
- 4 folds forment le train, 1 fold la validation, 1 fold le test ;
- le couple (validation, test) est choisi pour minimiser l'écart de taux de
  résistance sur les antibiotiques suffisamment observés, puis l'écart aux
  proportions 4/6, 1/6, 1/6 ;
- les isolats ``lignee == "inconnu"`` sont traités comme des groupes
  individuels (un groupe par ``souche``), faute d'information de clonalité.

Sorties (``data/splits_all_snp_3way/``) :

- ``X_{train,val,test}.parquet`` : features binaires (uint8) ;
- ``Y_{train,val,test}.parquet`` : 48 phénotypes Int8, -1 converti en NA ;
- ``metadata_{train,val,test}.parquet`` : identifiants, split, fold, compteurs ;
- ``split_summary.json`` : paramètres, contrôles et statistiques par split.

Exemple :
    python src/make_3way_splits_all_snp.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

META_COLS = ["souche", "lignee", "etude", "bloc"]
# Bornes de la charge de résistance : 0, 1, 2, 3-4, 5-8, 9+.
RESISTANCE_BIN_EDGES = [-1, 0, 1, 2, 4, 8, 100]
MIN_OBS_FOR_BALANCE = 200
UNKNOWN_LINEAGE = "inconnu"


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare les phénotypes (bloc contigu avec des -1) des features binaires."""
    missing = [column for column in META_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes de métadonnées absentes : {missing}")

    start = len(META_COLS)
    drug_cols: list[str] = []
    for column in df.columns[start:]:
        series = df[column]
        if series.dtype == np.int8 and series.min() < 0:
            drug_cols.append(column)
        else:
            break

    feature_cols = [column for column in df.columns[start + len(drug_cols):]]
    if not drug_cols:
        raise ValueError("Aucune colonne de phénotype (-1 = non testé) détectée")
    if not feature_cols:
        raise ValueError("Aucune feature détectée après les phénotypes")

    feature_values = df[feature_cols].to_numpy()
    if feature_values.min() < 0:
        raise ValueError("Des valeurs négatives ont été trouvées dans les features")
    if feature_values.max() > 1:
        raise ValueError("Les features doivent être binaires (0/1)")

    return drug_cols, feature_cols


def _group_labels(df: pd.DataFrame) -> np.ndarray:
    """Lignée MLST, ou identifiant unique pour les lignées 'inconnu'."""
    lignee = df["lignee"].astype(str).values
    souche = df["souche"].astype(str).values
    return np.where(lignee == UNKNOWN_LINEAGE, souche, lignee)


def _resistance_strata(drugs: np.ndarray) -> np.ndarray:
    """Classe chaque génome selon son nombre d'antibiotiques résistants.

    ``drugs`` est la matrice des phénotypes (-1 = non testé). La charge de
    résistance est plus informative qu'un seul antibiotique pour équilibrer
    les 48 phénotypes entre les splits.
    """
    observed = drugs >= 0
    n_resistant = (drugs == 1).sum(axis=1)
    n_tested = observed.sum(axis=1)
    if (n_tested == 0).any():
        raise ValueError("Des génomes n'ont aucun phénotype observé")

    strata = pd.cut(
        n_resistant,
        bins=RESISTANCE_BIN_EDGES,
        labels=False,
        include_lowest=True,
    )
    if pd.isna(strata).any():
        raise ValueError("Charge de résistance hors des bornes prévues")
    return np.asarray(strata).astype(int)


def _fold_rates(
    drugs: np.ndarray, eval_drug_indices: np.ndarray, folds: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compte les observations et les résistants par (fold, antibiotique)."""
    n_drugs = len(eval_drug_indices)
    counts = np.zeros((n_splits, n_drugs), dtype=np.int64)
    resistant = np.zeros((n_splits, n_drugs), dtype=np.int64)
    for fold in range(n_splits):
        mask = folds == fold
        subset = drugs[np.ix_(mask, eval_drug_indices)]
        counts[fold] = (subset >= 0).sum(axis=0)
        resistant[fold] = (subset == 1).sum(axis=0)
    return counts, resistant


def _best_fold_pair(
    counts: np.ndarray,
    resistant: np.ndarray,
    overall: np.ndarray,
    sizes: np.ndarray,
    n_rows: int,
    n_splits: int,
) -> tuple[int, int, pd.DataFrame]:
    """Choisit le couple (fold validation, fold test) le plus équilibré.

    Priorité au taux de résistance moyen sur les antibiotiques évaluables,
    puis aux proportions train/validation/test (4/6, 1/6, 1/6).
    """
    target = np.array([1.0 - 2.0 / n_splits, 1.0 / n_splits, 1.0 / n_splits])
    rows = []
    for val_fold in range(n_splits):
        for test_fold in range(n_splits):
            if val_fold == test_fold:
                continue

            val_mask = np.zeros(n_splits, dtype=bool)
            val_mask[val_fold] = True
            test_mask = np.zeros(n_splits, dtype=bool)
            test_mask[test_fold] = True
            pair_masks = [~val_mask & ~test_mask, val_mask, test_mask]

            pair_counts = np.stack([counts[m].sum(axis=0) for m in pair_masks])
            pair_resistant = np.stack([resistant[m].sum(axis=0) for m in pair_masks])
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.where(pair_counts > 0, pair_resistant / pair_counts, np.nan)
            deviations = np.abs(rates - overall[None, :])

            pair_sizes = np.array([sizes[m].sum() for m in pair_masks])
            size_score = np.abs(pair_sizes / n_rows - target).sum()
            rate_score = np.nanmean(deviations)
            score = 10.0 * rate_score + 2.0 * size_score

            rows.append(
                {
                    "val_fold": val_fold,
                    "test_fold": test_fold,
                    "n_train": int(pair_sizes[0]),
                    "n_val": int(pair_sizes[1]),
                    "n_test": int(pair_sizes[2]),
                    "mean_abs_rate_deviation": float(rate_score),
                    "mean_rate_train": float(np.nanmean(rates[0])),
                    "mean_rate_val": float(np.nanmean(rates[1])),
                    "mean_rate_test": float(np.nanmean(rates[2])),
                    "score": float(score),
                }
            )

    table = pd.DataFrame(rows).sort_values(
        ["score", "val_fold", "test_fold"], ignore_index=True
    )
    best = table.iloc[0]
    return int(best.val_fold), int(best.test_fold), table


def _split_stats(
    metadata: pd.DataFrame, labels: pd.DataFrame, drugs: np.ndarray
) -> dict:
    """Statistiques d'un split (tailles, charge de résistance, par antibiotique)."""
    observed = drugs >= 0
    rates = {}
    observed_counts = {}
    for index, column in enumerate(labels.columns):
        column_observed = observed[:, index]
        observed_counts[column] = int(column_observed.sum())
        rates[column] = (
            float(drugs[column_observed, index].mean())
            if column_observed.any()
            else None
        )

    return {
        "n": int(len(metadata)),
        "n_groups": int(metadata["groupe"].nunique()),
        "n_unknown_lineage": int((metadata["lignee"].astype(str) == UNKNOWN_LINEAGE).sum()),
        "n_tested_labels": int(observed.sum()),
        "n_resistant_labels": int((drugs == 1).sum()),
        "mean_resistant_per_genome": float(
            pd.Series((drugs == 1).sum(axis=1)).mean()
        ),
        "observed_per_drug": observed_counts,
        "resistance_rate_per_drug": rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/ecoli_all_snp.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/splits_all_snp_3way"))
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-obs-for-balance",
        type=int,
        default=MIN_OBS_FOR_BALANCE,
        help="Nombre minimal d'observations pour inclure un antibiotique dans "
        "l'équilibrage des folds.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    drug_cols, feature_cols = _split_columns(df)
    drugs = df[drug_cols].to_numpy(dtype=np.int8)
    meta = df[META_COLS].copy()
    meta["groupe"] = _group_labels(df)
    groups = meta["groupe"].to_numpy()
    strata = _resistance_strata(drugs)

    # Antibiotiques suffisamment observés pour peser dans le choix des folds.
    n_observed = (drugs >= 0).sum(axis=0)
    eval_indices = np.flatnonzero(n_observed >= args.min_obs_for_balance)
    if len(eval_indices) == 0:
        raise ValueError("Aucun antibiotique ne dépasse le seuil d'observations")
    overall_rates = np.array(
        [drugs[drugs[:, i] >= 0, i].mean() for i in eval_indices]
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    folds = np.full(len(meta), -1, dtype=int)
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(meta, strata, groups)):
        folds[test_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("Tous les isolats n'ont pas reçu de fold.")

    sizes = np.array([(folds == fold).sum() for fold in range(args.n_splits)])
    fold_counts, fold_resistant = _fold_rates(
        drugs, eval_indices, folds, args.n_splits
    )
    val_fold, test_fold, candidate_table = _best_fold_pair(
        fold_counts,
        fold_resistant,
        overall_rates,
        sizes,
        len(meta),
        args.n_splits,
    )

    split = np.full(len(meta), "train", dtype=object)
    split[folds == val_fold] = "validation"
    split[folds == test_fold] = "test"

    # Contrôle : aucun groupe (lignée connue ou souche 'inconnu') ne traverse
    # deux splits.
    overlap = []
    for group in pd.unique(groups):
        labels = set(split[groups == group])
        if len(labels) > 1:
            overlap.append((group, sorted(labels)))
    if overlap:
        raise RuntimeError(f"Groupes présents dans plusieurs splits : {overlap[:10]}")
    if len(pd.unique(meta.loc[split == "train", "groupe"])) + len(
        pd.unique(meta.loc[split == "validation", "groupe"])
    ) + len(pd.unique(meta.loc[split == "test", "groupe"])) != len(
        pd.unique(groups)
    ):
        raise RuntimeError("Groupes dupliqués entre les splits")

    args.out.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, suffix in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        mask = split == name
        X = df.loc[mask, feature_cols].astype(np.uint8)
        Y = df.loc[mask, drug_cols].astype("Int8")
        Y = Y.mask(Y < 0)  # -1 (non testé) -> NA

        part_meta = meta.loc[mask].copy()
        part_meta["split"] = name
        part_meta["fold_3way"] = folds[mask]
        part_meta["n_tested"] = (drugs[mask] >= 0).sum(axis=1).astype("int16")
        part_meta["n_resistant"] = (drugs[mask] == 1).sum(axis=1).astype("int16")

        # Contrôles d'intégrité.
        assert X.shape == (int(mask.sum()), len(feature_cols))
        assert not X.isna().any().any()
        assert Y.shape == (int(mask.sum()), len(drug_cols))
        assert (Y.isna() == (drugs[mask] < 0)).all().all()
        assert (
            part_meta["n_tested"].to_numpy()
            == Y.notna().sum(axis=1).to_numpy()
        ).all()
        assert (
            part_meta["n_resistant"].to_numpy()
            == Y.eq(1).sum(axis=1).to_numpy()
        ).all()

        X.to_parquet(args.out / f"X_{suffix}.parquet", index=False)
        Y.to_parquet(args.out / f"Y_{suffix}.parquet", index=False)
        part_meta.to_parquet(args.out / f"metadata_{suffix}.parquet", index=False)

        stats = _split_stats(part_meta, Y, drugs[mask])
        stats["proportion"] = float(mask.sum() / len(meta))
        split_stats[name] = stats

    summary = {
        "source": str(args.data),
        "out": str(args.out),
        "schema_version": "1.0",
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_drugs": int(len(drug_cols)),
        "drug_columns": drug_cols,
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "val_fold": int(val_fold),
        "test_fold": int(test_fold),
        "target_proportions": {
            "train": 1.0 - 2.0 / args.n_splits,
            "validation": 1.0 / args.n_splits,
            "test": 1.0 / args.n_splits,
        },
        "stratification": (
            "Charge de résistance par génome (nombre d'antibiotiques R), "
            f"en classes {RESISTANCE_BIN_EDGES[1:-1]}."
        ),
        "balance_drugs": {
            "min_observations": int(args.min_obs_for_balance),
            "n_drugs": int(len(eval_indices)),
            "drugs": [drug_cols[i] for i in eval_indices],
            "overall_resistance_rate": {
                drug_cols[i]: float(overall_rates[j])
                for j, i in enumerate(eval_indices)
            },
        },
        "unknown_handling": (
            "Chaque isolat 'inconnu' est traité comme un groupe unique (souche)."
        ),
        "group_integrity": (
            "Aucune lignée connue (ni aucune souche 'inconnu') n'apparaît dans "
            "plusieurs splits."
        ),
        "limitations": [
            "Le split est groupé par lignée, pas par étude : une même étude "
            "peut être présente dans plusieurs splits.",
            "Le groupement par lignée MLST ne garantit pas l'absence de "
            "phylogénie fine partagée entre splits.",
        ],
        "splits": split_stats,
        "candidate_fold_pairs": candidate_table.head(10).to_dict(orient="records"),
    }

    (args.out / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"X_train : {split_stats['train']['n']} x {len(feature_cols)}")
    print(f"X_val   : {split_stats['validation']['n']} x {len(feature_cols)}")
    print(f"X_test  : {split_stats['test']['n']} x {len(feature_cols)}")
    print(f"Antibiotiques : {len(drug_cols)}")
    for name in ("train", "validation", "test"):
        stats = split_stats[name]
        print(
            f"{name:10s} : {stats['n']:5d} génomes | "
            f"{stats['mean_resistant_per_genome']:.2f} R/génome | "
            f"{stats['n_tested_labels']:6d} labels observés"
        )
    print(f"Folds retenus : validation={val_fold}, test={test_fold}")
    print(f"Écrit dans {args.out}")


if __name__ == "__main__":
    main()
