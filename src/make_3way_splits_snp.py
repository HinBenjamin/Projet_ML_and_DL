#!/usr/bin/env python3
"""Crée un split train / validation / test groupé par MLST.

Le split aléatoire initial laissait 95,8 % des isolats de test partager un
groupe MLST avec le train. Ici, les groupes MLST sont gardés intacts : un même
groupe ne peut pas être présent dans deux splits.

Stratégie :
- `StratifiedGroupKFold` à 6 folds, avec stratification sur `y` ;
- 4 folds forment le train, 1 fold la validation, 1 fold le test ;
- le couple (validation, test) est choisi pour se rapprocher de 70/15/15 et
  du taux de résistance global ;
- les isolats `mlst_group == "unknown"` sont traités comme des groupes
  individuels (un groupe par `sra`), faute d'information de clonalité.

Exemple :
    python src/make_3way_splits_snp.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

META_COLS = ["sra", "Genome.Name", "mlst_group", "y"]


def _group_labels(df: pd.DataFrame) -> np.ndarray:
    """Groupe MLST, ou identifiant unique pour les 'unknown'."""
    mlst = df["mlst_group"].astype(str).values
    sra = df["sra"].astype(str).values
    return np.where(mlst == "unknown", sra, mlst)


def _best_fold_pair(
    folds: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    target: np.ndarray = np.array([0.70, 0.15, 0.15]),
) -> tuple[int, int, pd.DataFrame]:
    """Choisit le couple (fold validation, fold test) le plus équilibré."""
    overall = y.mean()
    rows = []
    for val_fold in range(n_splits):
        for test_fold in range(n_splits):
            if val_fold == test_fold:
                continue
            train_mask = (folds != val_fold) & (folds != test_fold)
            val_mask = folds == val_fold
            test_mask = folds == test_fold

            sizes = np.array([train_mask.sum(), val_mask.sum(), test_mask.sum()])
            rates = np.array([y[train_mask].mean(), y[val_mask].mean(), y[test_mask].mean()])
            # Priorité au taux de résistance, puis aux proportions.
            score = 10.0 * np.abs(rates - overall).sum() + 2.0 * np.abs(sizes / len(y) - target).sum()
            rows.append({
                "val_fold": val_fold,
                "test_fold": test_fold,
                "n_train": int(sizes[0]),
                "n_val": int(sizes[1]),
                "n_test": int(sizes[2]),
                "rate_train": float(rates[0]),
                "rate_val": float(rates[1]),
                "rate_test": float(rates[2]),
                "score": float(score),
            })
    table = pd.DataFrame(rows).sort_values(
        ["score", "val_fold", "test_fold"], ignore_index=True
    )
    best = table.iloc[0]
    return int(best.val_fold), int(best.test_fold), table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/ecoli_ciprofloxacin_snp.parquet"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/splits_snp_3way"))
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    meta = df[META_COLS].copy()
    groups = _group_labels(meta)
    y = meta["y"].to_numpy()
    overall_rate = float(y.mean())

    splitter = StratifiedGroupKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )
    folds = np.full(len(meta), -1, dtype=int)
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(meta, y, groups)):
        folds[test_idx] = fold
    if (folds < 0).any():
        raise RuntimeError("Tous les isolats n'ont pas reçu de fold.")

    val_fold, test_fold, table = _best_fold_pair(
        folds, y, args.n_splits
    )
    split = np.full(len(meta), "train", dtype=object)
    split[folds == val_fold] = "validation"
    split[folds == test_fold] = "test"

    # Contrôle : aucun groupe MLST connu ne traverse deux splits.
    overlap = []
    for group in pd.unique(groups):
        labels = set(split[groups == group])
        if len(labels) > 1:
            overlap.append((group, sorted(labels)))
    if overlap:
        raise RuntimeError(f"Groupes présents dans plusieurs splits : {overlap[:10]}")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for name in ("train", "validation", "test"):
        mask = split == name
        part = df.loc[mask].reset_index(drop=True)
        part_meta = part[META_COLS].copy()
        part_meta["split"] = name
        part_meta["fold_3way"] = folds[mask]
        X = part[feature_cols].astype(np.uint8)
        y_part = pd.DataFrame({"y": part["y"].astype(np.int8)})

        suffix = "val" if name == "validation" else name
        X.to_parquet(args.out / f"X_{suffix}.parquet", index=False)
        y_part.to_parquet(args.out / f"y_{suffix}.parquet", index=False)
        part_meta.to_parquet(args.out / f"metadata_{suffix}.parquet", index=False)

        summaries[name] = {
            "n": int(mask.sum()),
            "n_positive": int(part["y"].sum()),
            "positive_rate": float(part["y"].mean()),
            "n_groups": int(len(pd.unique(groups[mask]))),
            "n_unknown": int((part["mlst_group"].astype(str) == "unknown").sum()),
        }

    summary = {
        "source": str(args.data),
        "out": str(args.out),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_snp_features": int(sum(c.startswith("SNP_") for c in feature_cols)),
        "n_splits": int(args.n_splits),
        "seed": int(args.seed),
        "val_fold": int(val_fold),
        "test_fold": int(test_fold),
        "overall_positive_rate": overall_rate,
        "target_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "splits": summaries,
        "unknown_handling": "Chaque isolat 'unknown' est traité comme un groupe unique (sra).",
        "group_integrity": "Aucun groupe MLST connu n'apparaît dans plusieurs splits.",
        "candidate_fold_pairs": table.head(10).to_dict(orient="records"),
    }
    (args.out / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Fichiers écrits dans {args.out}")


if __name__ == "__main__":
    main()
