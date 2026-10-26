#!/usr/bin/env python3
"""Crée data/splits_snp/ à partir du jeu enrichi avec les SNP.

Le découpage train/test existant (data/splits/) est réutilisé tel quel afin de
comparer les modèles « pangénome seul » et « pangénome + SNP » à split constant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = ["sra", "Genome.Name", "mlst_group", "y"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/ecoli_ciprofloxacin_snp.parquet"),
    )
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    parser.add_argument("--out", type=Path, default=Path("data/splits_snp"))
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    meta_train = pd.read_parquet(args.splits / "metadata_train.parquet")
    meta_test = pd.read_parquet(args.splits / "metadata_test.parquet")

    feature_cols = [c for c in df.columns if c not in META_COLS]
    indexed = df.set_index("sra", drop=False)

    missing_train = set(meta_train["sra"]) - set(indexed.index)
    missing_test = set(meta_test["sra"]) - set(indexed.index)
    if missing_train or missing_test:
        raise ValueError(
            f"sra absents du jeu enrichi : train={len(missing_train)}, "
            f"test={len(missing_test)}"
        )

    train = indexed.loc[meta_train["sra"]].reset_index(drop=True)
    test = indexed.loc[meta_test["sra"]].reset_index(drop=True)

    y_train = pd.DataFrame({"y": train["y"].astype(np.int8)})
    y_test = pd.DataFrame({"y": test["y"].astype(np.int8)})

    meta_train_out = meta_train.copy()
    meta_test_out = meta_test.copy()
    for col in ("sra", "Genome.Name", "mlst_group", "y"):
        meta_train_out[col] = train[col].values
        meta_test_out[col] = test[col].values

    X_train = train[feature_cols].astype(np.uint8)
    X_test = test[feature_cols].astype(np.uint8)

    # Contrôles d'intégrité.
    assert X_train.shape[1] == X_test.shape[1] == len(feature_cols)
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert (y_train["y"].values == meta_train_out["y"].values).all()
    assert (y_test["y"].values == meta_test_out["y"].values).all()

    args.out.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(args.out / "X_train.parquet", index=False)
    X_test.to_parquet(args.out / "X_test.parquet", index=False)
    y_train.to_parquet(args.out / "y_train.parquet", index=False)
    y_test.to_parquet(args.out / "y_test.parquet", index=False)
    meta_train_out.to_parquet(args.out / "metadata_train.parquet", index=False)
    meta_test_out.to_parquet(args.out / "metadata_test.parquet", index=False)

    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"Features : {len(feature_cols)}")
    print(f"y_train : {y_train['y'].value_counts().to_dict()}")
    print(f"y_test  : {y_test['y'].value_counts().to_dict()}")
    print(f"Écrit dans {args.out}")


if __name__ == "__main__":
    main()
