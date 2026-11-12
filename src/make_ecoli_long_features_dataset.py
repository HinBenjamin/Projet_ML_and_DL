#!/usr/bin/env python3
"""Assemble le jeu *E. coli* « tous antibiotiques » en un seul fichier long.

Le fichier produit rassemble, dans **une seule table Parquet** :

- une ligne par couple **génome × antibiotique** dont le phénotype binaire est
  observé (Résistant ou Sensible) ;
- les métadonnées du génome (``Genome.Name``, ``sra``, ``mlst_group``) ;
- le split train / validation / test et le fold d'origine ;
- l'antibiotique explicitement nommé (``antibiotic_label`` et ``antibiotic``),
  sa classe pharmacologique et la résistance binaire (``resistance``) ;
- les 20 929 features pangénome + SNP QRDR.

Entrée :
``data/processed/ecoli_all_antibiotics/ecoli_all_antibiotics_ml.parquet``
(5 935 génomes × 50 labels × 20 929 features).

Sortie :
``data/processed/ecoli_all_antibiotics/ecoli_all_antibiotics_long_features.parquet``

Le fichier est écrit par groupes de lignes (un antibiotique par groupe) pour
limiter l'empreinte mémoire. Les combinaisons non testées (``NA``) et les
intermédiaires sont exclus : ils ne doivent pas être transformés en sensibles.

Exemple :
    python src/make_ecoli_long_features_dataset.py

Pour ne garder que les antibiotiques suffisamment observés :
    python src/make_ecoli_long_features_dataset.py --min-observations 30
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecoli_all_antibiotics"
    / "ecoli_all_antibiotics_ml.parquet"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecoli_all_antibiotics"
    / "ecoli_all_antibiotics_drug_summary.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecoli_all_antibiotics"
    / "ecoli_all_antibiotics_long_features.parquet"
)

META_COLUMNS = ["Genome.Name", "sra", "mlst_group", "split", "fold_3way"]
LABEL_COLUMNS = [
    "antibiotic_label",
    "antibiotic",
    "drug_class",
    "resistance",
    "raw_label",
]
RAW_LABELS = {1: "Resistant", 0: "Susceptible"}


def _sha256(path: Path) -> str:
    """Empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _drug_metadata(summary_path: Path) -> dict[str, dict[str, str]]:
    """Charge le libellé lisible et la classe de chaque antibiotique."""
    summary = pd.read_csv(summary_path)
    required = {"antibiotic", "antibiotic_label", "drug_class"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {summary_path.name} : {sorted(missing)}")
    return {
        row["antibiotic"]: {
            "antibiotic_label": row["antibiotic_label"],
            "drug_class": row["drug_class"],
        }
        for _, row in summary.iterrows()
    }


def _write_manifest(
    output_path: Path,
    input_path: Path,
    counts: dict,
    per_drug: pd.DataFrame,
    feature_columns: list[str],
) -> Path:
    """Écrit un manifeste de traçabilité à côté du fichier long."""
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/make_ecoli_long_features_dataset.py",
        "input": {
            "file": str(input_path),
            "sha256": _sha256(input_path),
        },
        "output": {
            "file": output_path.name,
            "bytes": int(output_path.stat().st_size),
            "sha256": _sha256(output_path),
        },
        "counts": counts,
        "columns": {
            "metadata": META_COLUMNS,
            "labels": LABEL_COLUMNS,
            "n_features": len(feature_columns),
            "n_snp_features": sum(c.startswith("SNP_") for c in feature_columns),
            "n_pangenome_features": sum(
                not c.startswith("SNP_") for c in feature_columns
            ),
        },
        "per_drug": per_drug.to_dict(orient="records"),
        "notes": [
            "Une ligne = un couple génome × antibiotique avec phénotype binaire observé.",
            "resistance : 1 = Resistant, 0 = Susceptible. Les intermédiaires et les "
            "combinaisons non testées sont exclus (NA ≠ 0).",
            "antibiotic_label est le nom lisible, antibiotic le nom snake_case.",
            "Les features sont dupliquées pour chaque antibiotique du même génome ; "
            "pour l'entraînement, préférez le fichier large "
            "ecoli_all_antibiotics_ml.parquet et ses colonnes y_*.",
            "Le split est le split 3-way groupé par MLST du projet.",
        ],
    }
    manifest_path = output_path.with_name(output_path.stem + "_manifest.json")
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def build_long_features(
    input_path: Path = DEFAULT_INPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    output_path: Path = DEFAULT_OUTPUT,
    min_observations: int = 0,
) -> dict:
    """Construit le fichier long unique et retourne le manifeste."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ml = pd.read_parquet(input_path)
    y_columns = [column for column in ml.columns if column.startswith("y_")]
    if not y_columns:
        raise ValueError(f"Aucune colonne y_* dans {input_path.name}")

    missing_meta = set(META_COLUMNS).difference(ml.columns)
    if missing_meta:
        raise ValueError(f"Colonnes de métadonnées absentes : {sorted(missing_meta)}")

    feature_columns = [
        column
        for column in ml.columns
        if column not in set(META_COLUMNS) and not column.startswith("y_")
    ]
    if not feature_columns:
        raise ValueError(f"Aucune feature dans {input_path.name}")

    drugs = _drug_metadata(summary_path)

    # Métadonnées et labels restent petits ; les features sont converties en
    # uint8 (0/1) pour diviser l'empreinte mémoire par huit.
    meta = ml[META_COLUMNS].reset_index(drop=True)
    labels = ml[y_columns].reset_index(drop=True)
    features = ml[feature_columns].to_numpy(dtype=np.uint8)
    del ml

    selection = []
    for y_column in y_columns:
        drug = y_column[len("y_") :]
        n_observed = int(labels[y_column].notna().sum())
        if n_observed >= min_observations and n_observed > 0:
            selection.append((y_column, drug, n_observed))

    if not selection:
        raise ValueError("Aucun antibiotique ne satisfait le filtre demandé.")

    total_rows = sum(n for _, _, n in selection)
    print(
        f"{len(selection)} antibiotiques, {total_rows:,} lignes, "
        f"{len(feature_columns):,} features"
    )

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    per_drug_rows: list[dict] = []
    position = 0

    try:
        for y_column, drug, n_observed in selection:
            mask = labels[y_column].notna().to_numpy()
            indices = np.flatnonzero(mask)
            n_rows = int(len(indices))
            if n_rows == 0:
                continue

            values = labels[y_column].to_numpy()[indices].astype(np.int8)
            drug_info = drugs.get(
                drug, {"antibiotic_label": drug, "drug_class": "non classé"}
            )

            batch = pd.DataFrame(
                features[indices], columns=feature_columns, copy=False
            )
            batch.insert(0, "raw_label", [RAW_LABELS[int(v)] for v in values])
            batch.insert(0, "resistance", values)
            batch.insert(0, "drug_class", drug_info["drug_class"])
            batch.insert(0, "antibiotic", drug)
            batch.insert(0, "antibiotic_label", drug_info["antibiotic_label"])
            for column in reversed(META_COLUMNS):
                batch.insert(0, column, meta[column].to_numpy()[indices])

            table = pa.Table.from_pandas(
                batch.loc[:, META_COLUMNS + LABEL_COLUMNS + feature_columns],
                preserve_index=False,
            )
            if schema is None:
                schema = table.schema
                writer = pq.ParquetWriter(
                    output_path, schema, compression="zstd"
                )
            elif not table.schema.equals(schema):
                table = table.cast(schema)
            writer.write_table(table)
            del batch, table

            n_resistant = int(values.sum())
            per_drug_rows.append(
                {
                    "antibiotic": drug,
                    "antibiotic_label": drug_info["antibiotic_label"],
                    "drug_class": drug_info["drug_class"],
                    "n_rows": n_rows,
                    "n_resistant": n_resistant,
                    "n_susceptible": n_rows - n_resistant,
                    "prevalence_R": round(n_resistant / n_rows, 6),
                }
            )
            position += n_rows
            if position % 10000 < n_rows:
                print(f"  {position:,}/{total_rows:,} lignes écrites")
    finally:
        if writer is not None:
            writer.close()

    if position != total_rows:
        raise RuntimeError(
            f"Incohérence de comptage : {position} lignes écrites pour {total_rows} attendues"
        )

    per_drug = pd.DataFrame(per_drug_rows)
    per_drug.to_csv(
        output_path.with_name(output_path.stem + "_index.csv"),
        index=False,
    )

    counts = {
        "n_drugs": int(len(per_drug)),
        "n_rows": int(total_rows),
        "n_genomes": int(len(meta)),
        "n_features": int(len(feature_columns)),
        "n_snp_features": int(
            sum(column.startswith("SNP_") for column in feature_columns)
        ),
        "n_pangenome_features": int(
            sum(not column.startswith("SNP_") for column in feature_columns)
        ),
        "n_resistant": int(per_drug["n_resistant"].sum()),
        "n_susceptible": int(per_drug["n_susceptible"].sum()),
    }

    manifest_path = _write_manifest(
        output_path, input_path, counts, per_drug, feature_columns
    )
    print(f"Fichier écrit : {output_path}")
    print(f"Manifeste     : {manifest_path}")
    print(
        f"{counts['n_rows']:,} lignes | {counts['n_drugs']} antibiotiques | "
        f"{counts['n_genomes']:,} génomes"
    )
    return {
        "output": str(output_path),
        "manifest": str(manifest_path),
        "counts": counts,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-observations",
        type=int,
        default=0,
        help="Ne conserver que les antibiotiques avec au moins N labels observés.",
    )
    args = parser.parse_args(argv)

    build_long_features(
        input_path=args.input,
        summary_path=args.summary,
        output_path=args.output,
        min_observations=args.min_observations,
    )


if __name__ == "__main__":
    main()
