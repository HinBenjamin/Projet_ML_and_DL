#!/usr/bin/env python3
"""Exporte le fichier long *E. coli* « génome × antibiotique » en CSV.

Le fichier Parquet source contient 62 666 lignes × 20 939 colonnes. Le CSV
correspondant pèse environ 2,6 Go. Il est écrit par lots pour éviter de charger
toute la table en mémoire.

Exemple :
    python src/export_ecoli_long_features_csv.py

Sortie par défaut :
    data/processed/ecoli_all_antibiotics/ecoli_all_antibiotics_long_features.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ecoli_all_antibiotics"
    / "ecoli_all_antibiotics_long_features.parquet"
)
DEFAULT_CSV = DEFAULT_PARQUET.with_suffix(".csv")


def export_csv(
    parquet_path: Path = DEFAULT_PARQUET,
    csv_path: Path = DEFAULT_CSV,
    batch_size: int = 2000,
) -> dict:
    """Écrit le Parquet en CSV et retourne quelques statistiques."""
    parquet_path = parquet_path.resolve()
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(parquet_path)
    total_rows = parquet_file.metadata.num_rows
    n_columns = parquet_file.metadata.num_columns

    started = time.time()
    written = 0
    first_batch = True
    try:
        with csv_path.open("wb") as sink:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                pacsv.write_csv(
                    table,
                    sink,
                    write_options=pacsv.WriteOptions(include_header=first_batch),
                )
                first_batch = False
                written += batch.num_rows
                if written % 10000 < batch_size:
                    elapsed = time.time() - started
                    print(f"  {written:,}/{total_rows:,} lignes ({elapsed:.0f} s)")
    except Exception:
        # Ne pas laisser un CSV partiel à moitié écrit.
        if csv_path.exists():
            csv_path.unlink()
        raise

    if written != total_rows:
        raise RuntimeError(
            f"CSV incomplet : {written} lignes écrites pour {total_rows} attendues"
        )

    elapsed = time.time() - started
    size = csv_path.stat().st_size
    print(f"CSV écrit : {csv_path}")
    print(
        f"{written:,} lignes × {n_columns:,} colonnes | "
        f"{size / 1024**3:.2f} Go | {elapsed:.0f} s"
    )
    return {
        "csv_path": str(csv_path),
        "n_rows": written,
        "n_columns": n_columns,
        "bytes": int(size),
        "elapsed_seconds": round(elapsed, 1),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Nombre de lignes lues/écrites par lot (défaut : 2000).",
    )
    args = parser.parse_args(argv)

    export_csv(
        parquet_path=args.parquet,
        csv_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
