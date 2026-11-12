#!/usr/bin/env python3
"""Matérialise le split train / validation / test du jeu *E. coli* × 50 antibiotiques.

Le fichier ML contient déjà les colonnes `split` et `fold_3way`, issues du split
3-way groupé par MLST du jeu ciprofloxacine. Ce script découpe ce fichier en
fichiers X / y / metadata prêts à l'emploi, vérifie l'intégrité des groupes MLST
et écrit un résumé JSON.

Exemple :
    python src/make_3way_splits_all_antibiotics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

META_COLS = ["sra", "Genome.Name", "mlst_group", "split", "fold_3way"]
SPLIT_ORDER = ["train", "validation", "test"]
SUFFIX = {"train": "train", "validation": "val", "test": "test"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "data/processed/ecoli_all_antibiotics/ecoli_all_antibiotics_ml.parquet"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("data/splits_all_antibiotics"))
    args = parser.parse_args()

    df = pd.read_parquet(args.data)

    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {args.data} : {missing}")

    label_cols = [c for c in df.columns if c.startswith("y_")]
    feature_cols = [
        c for c in df.columns if c not in META_COLS and c not in label_cols
    ]
    if not label_cols:
        raise ValueError("Aucun label y_* trouvé.")
    if not feature_cols:
        raise ValueError("Aucune feature trouvée.")

    if df["sra"].duplicated().any():
        raise ValueError("Des sra sont dupliqués.")
    if df["split"].isna().any():
        raise ValueError("Des lignes n'ont pas de split.")
    unexpected = set(df["split"].unique()) - set(SPLIT_ORDER)
    if unexpected:
        raise ValueError(f"Valeurs de split inattendues : {sorted(unexpected)}")

    # Aucun groupe MLST connu ne doit traverser deux splits.
    known = df[df["mlst_group"].astype(str) != "unknown"]
    groups_per_split = known.groupby("mlst_group")["split"].nunique()
    overlap = groups_per_split[groups_per_split > 1]
    if len(overlap):
        raise RuntimeError(
            f"Groupes présents dans plusieurs splits : {list(overlap.index[:10])}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for name in SPLIT_ORDER:
        part = df[df["split"] == name].reset_index(drop=True)
        suffix = SUFFIX[name]

        X = part[feature_cols].astype("uint8")
        y = part[label_cols].astype("Int8")
        metadata = part[META_COLS]

        X.to_parquet(args.out / f"X_{suffix}.parquet", index=False)
        y.to_parquet(args.out / f"y_{suffix}.parquet", index=False)
        metadata.to_parquet(args.out / f"metadata_{suffix}.parquet", index=False)

        mlst = part["mlst_group"].astype(str)
        summaries[name] = {
            "n": int(len(part)),
            "n_groups": int(part.loc[mlst != "unknown", "mlst_group"].nunique()),
            "n_unknown": int((mlst == "unknown").sum()),
            "fold_3way": sorted(int(f) for f in part["fold_3way"].unique()),
            "n_observed_labels": int(part[label_cols].notna().sum().sum()),
            "n_resistant": int((part[label_cols] == 1).sum().sum()),
            "n_susceptible": int((part[label_cols] == 0).sum().sum()),
        }

    per_drug = []
    for col in label_cols:
        values = df[col]
        n_observed = int(values.notna().sum())
        n_resistant = int((values == 1).sum())
        per_drug.append(
            {
                "label": col,
                "n_observed": n_observed,
                "n_resistant": n_resistant,
                "n_susceptible": int((values == 0).sum()),
                "prevalence_R": (
                    float(n_resistant / n_observed) if n_observed else None
                ),
                "per_split": {
                    name: {
                        "n_observed": int(df.loc[df["split"] == name, col].notna().sum()),
                        "n_resistant": int(
                            (df.loc[df["split"] == name, col] == 1).sum()
                        ),
                    }
                    for name in SPLIT_ORDER
                },
            }
        )

    summary = {
        "source": str(args.data),
        "out": str(args.out),
        "n_rows": int(len(df)),
        "n_features": int(len(feature_cols)),
        "n_labels": int(len(label_cols)),
        "split_source": (
            "Colonnes split / fold_3way du fichier ML, issues du split 3-way "
            "groupé par MLST du jeu ciprofloxacine."
        ),
        "group_integrity": "Aucun groupe MLST connu n'apparaît dans plusieurs splits.",
        "unknown_handling": "Chaque isolat 'unknown' est traité comme un groupe unique (sra).",
        "splits": summaries,
        "per_drug": per_drug,
    }
    (args.out / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(json.dumps({k: summary[k] for k in ("n_rows", "n_features", "n_labels")}, indent=2))
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Fichiers écrits dans {args.out}")


if __name__ == "__main__":
    main()
