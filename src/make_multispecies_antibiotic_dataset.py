#!/usr/bin/env python3
"""Assemble le jeu de données multi-espèces « tous antibiotiques » à partir des fichiers bruts.

Le dossier ``data/raw/genome_id`` contient un fichier par couple
(espèce, antibiotique), avec deux colonnes : ``genome_id`` et
``resistant_phenotype`` (0 = sensible, 1 = résistant). Ce script :

1. lit et valide les 80 fichiers ;
2. produit une table **longue** (une observation = un génome × un antibiotique),
   au format Parquet (format principal) et CSV compressé (interopérabilité) ;
3. produit une matrice **large** (génomes × antibiotiques) avec des valeurs
   manquantes explicites (``Int8`` nullable), globale et par espèce ;
4. écrit les tableaux de synthèse, le dictionnaire des antibiotiques et un
   manifeste de traçabilité (comptages, contrôles, empreintes SHA-256).

Exemple :
    python src/make_multispecies_antibiotic_dataset.py

Les sorties sont écrites par défaut dans
``data/processed/amr_multispecies/`` (dossier ignoré par Git).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "genome_id"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "amr_multispecies"

# Espèces présentes dans le dépôt. La liste sert à découper les noms de
# fichiers sans ambiguïté (les noms d'antibiotiques contiennent aussi des « _ »).
SPECIES = [
    "Acinetobacter_baumannii",
    "Campylobacter_jejuni",
    "Enterococcus_faecium",
    "Escherichia_coli",
    "Klebsiella_pneumoniae",
    "Mycobacterium_tuberculosis",
    "Neisseria_gonorrhoeae",
    "Pseudomonas_aeruginosa",
    "Salmonella_enterica",
    "Staphylococcus_aureus",
    "Streptococcus_pneumoniae",
]

# Synonymes / fautes de frappe repérés dans les noms de fichiers.
# La clé est le nom brut normalisé (minuscules, « - » -> « _ »).
ANTIBIOTIC_ALIASES = {
    "ethiomide": "ethionamide",
    "cotrimoxazole": "trimethoprim_sulfamethoxazole",
}

# Classes pharmacologiques, pour les analyses par famille d'antibiotiques.
DRUG_CLASSES = {
    "amikacin": "aminoglycoside",
    "amoxicillin": "beta-lactam",
    "amoxicillin_clavulanic_acid": "beta-lactam / inhibiteur",
    "ampicillin": "beta-lactam",
    "ampicillin_sulbactam": "beta-lactam / inhibiteur",
    "azithromycin": "macrolide",
    "aztreonam": "beta-lactam (monobactame)",
    "beta_lactam": "beta-lactam (classe)",
    "capreomycin": "polypeptide (anti-tuberculeux)",
    "cefepime": "beta-lactam (céphalosporine)",
    "cefixime": "beta-lactam (céphalosporine)",
    "cefotaxime": "beta-lactam (céphalosporine)",
    "cefoxitin": "beta-lactam (céphamycine)",
    "ceftazidime": "beta-lactam (céphalosporine)",
    "ceftiofur": "beta-lactam (céphalosporine)",
    "ceftriaxone": "beta-lactam (céphalosporine)",
    "cefuroxime": "beta-lactam (céphalosporine)",
    "chloramphenicol": "amphénicol",
    "ciprofloxacin": "fluoroquinolone",
    "clindamycin": "lincosamide",
    "erythromycin": "macrolide",
    "ethambutol": "anti-tuberculeux",
    "ethionamide": "anti-tuberculeux (thioamide)",
    "fusidic_acid": "acide fusidique",
    "gentamicin": "aminoglycoside",
    "imipenem": "carbapénème",
    "isoniazid": "anti-tuberculeux",
    "kanamycin": "aminoglycoside",
    "levofloxacin": "fluoroquinolone",
    "meropenem": "carbapénème",
    "methicillin": "beta-lactam (pénicilline)",
    "nalidixic_acid": "quinolone",
    "ofloxacin": "fluoroquinolone",
    "penicillin": "beta-lactam (pénicilline)",
    "piperacillin_tazobactam": "beta-lactam / inhibiteur",
    "pyrazinamide": "anti-tuberculeux",
    "rifampicin": "rifamycine",
    "streptomycin": "aminoglycoside",
    "sulfisoxazole": "sulfamide",
    "tetracycline": "tétracycline",
    "tobramycin": "aminoglycoside",
    "trimethoprim": "diaminopyrimidine",
    "trimethoprim_sulfamethoxazole": "diaminopyrimidine / sulfamide",
    "vancomycin": "glycopeptide",
}

# Libellés lisibles pour les tableaux et les figures.
DISPLAY_NAMES = {
    "amoxicillin_clavulanic_acid": "amoxicillin/clavulanic acid",
    "ampicillin_sulbactam": "ampicillin/sulbactam",
    "beta_lactam": "beta-lactam",
    "fusidic_acid": "fusidic acid",
    "nalidixic_acid": "nalidixic acid",
    "piperacillin_tazobactam": "piperacillin/tazobactam",
    "trimethoprim_sulfamethoxazole": "trimethoprim/sulfamethoxazole",
}

LONG_COLUMNS = [
    "genome_id",
    "species",
    "antibiotic",
    "antibiotic_std",
    "drug_class",
    "phenotype",
    "cohort",
    "source_file",
    "source_row",
]


def _sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier sans le charger en mémoire."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_name(antibiotic_std: str) -> str:
    """Retourne un libellé lisible pour un antibiotique standardisé."""
    if antibiotic_std in DISPLAY_NAMES:
        return DISPLAY_NAMES[antibiotic_std]
    return antibiotic_std.replace("_", " ")


def _parse_filename(path: Path) -> tuple[str, str]:
    """Découpe ``Data_<espèce>_<antibiotique>.txt``.

    Retourne ``(espèce, antibiotique_brut)``. L'antibiotique brut garde la
    graphie du fichier (par exemple ``beta-lactam`` ou ``ethiomide``).
    """
    stem = path.stem
    if not stem.startswith("Data_"):
        raise ValueError(f"Nom de fichier inattendu : {path.name!r}")

    payload = stem[len("Data_") :]
    for species in SPECIES:
        prefix = f"{species}_"
        if payload.startswith(prefix):
            antibiotic = payload[len(prefix) :]
            if not antibiotic:
                raise ValueError(f"Antibiotique vide dans {path.name!r}")
            return species, antibiotic

    raise ValueError(
        f"Espèce inconnue dans {path.name!r}. Ajoutez-la à la constante SPECIES."
    )


def _standardize_antibiotic(antibiotic: str) -> str:
    """Normalise un nom d'antibiotique en snake_case sans synonyme."""
    normalized = antibiotic.strip().lower().replace("-", "_")
    return ANTIBIOTIC_ALIASES.get(normalized, normalized)


def _load_one_file(path: Path, species: str, antibiotic: str) -> pd.DataFrame:
    """Lit et valide un fichier (espèce, antibiotique)."""
    frame = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]

    required = {"genome_id", "resistant_phenotype"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path.name} : {sorted(missing)}")

    frame = frame.loc[:, ["genome_id", "resistant_phenotype"]].copy()
    frame["genome_id"] = frame["genome_id"].str.strip()
    frame["resistant_phenotype"] = frame["resistant_phenotype"].str.strip()

    if frame["genome_id"].isna().any() or (frame["genome_id"] == "").any():
        raise ValueError(f"Identifiant de génome vide dans {path.name}")
    if frame["resistant_phenotype"].isna().any():
        raise ValueError(f"Phénotype manquant dans {path.name}")

    phenotype = pd.to_numeric(frame["resistant_phenotype"], errors="raise")
    invalid = ~phenotype.isin([0, 1])
    if invalid.any():
        values = sorted(phenotype[invalid].unique().tolist())
        raise ValueError(f"Phénotypes hors {{0, 1}} dans {path.name} : {values}")

    if frame["genome_id"].duplicated().any():
        duplicated = frame.loc[frame["genome_id"].duplicated(), "genome_id"].head()
        raise ValueError(f"Identifiants dupliqués dans {path.name} : {duplicated.tolist()}")

    antibiotic_std = _standardize_antibiotic(antibiotic)
    result = pd.DataFrame(
        {
            "genome_id": frame["genome_id"].astype("string"),
            "species": species,
            "antibiotic": antibiotic,
            "antibiotic_std": antibiotic_std,
            "drug_class": DRUG_CLASSES.get(antibiotic_std, "non classé"),
            "phenotype": phenotype.astype("int8"),
            "cohort": f"{species}::{antibiotic}",
            "source_file": path.name,
            "source_row": np.arange(1, len(frame) + 1, dtype=np.int32),
        }
    )
    return result.loc[:, LONG_COLUMNS]


def _build_long(input_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Lit tous les fichiers et empile les observations."""
    files = sorted(input_dir.glob("Data_*.txt"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier Data_*.txt dans {input_dir}")

    frames = []
    sources = []
    for path in files:
        species, antibiotic = _parse_filename(path)
        frame = _load_one_file(path, species, antibiotic)
        frames.append(frame)
        sources.append(
            {
                "file": path.name,
                "species": species,
                "antibiotic": antibiotic,
                "antibiotic_std": frame["antibiotic_std"].iat[0],
                "n_observations": int(len(frame)),
                "n_resistant": int(frame["phenotype"].sum()),
                "sha256": _sha256(path),
            }
        )

    long = pd.concat(frames, ignore_index=True)
    for column in [
        "genome_id",
        "species",
        "antibiotic",
        "antibiotic_std",
        "drug_class",
        "cohort",
        "source_file",
    ]:
        long[column] = long[column].astype("string")
    long = long.sort_values(
        ["species", "antibiotic", "source_row"], kind="mergesort"
    ).reset_index(drop=True)
    return long, sources


def _quality_checks(long: pd.DataFrame) -> dict:
    """Contrôles de cohérence et statistiques de base."""
    duplicates = int(long.duplicated(["cohort", "genome_id"]).sum())
    if duplicates:
        raise ValueError(f"{duplicates} observations dupliquées (cohorte, génome).")

    # Après standardisation, deux cohortes synonymes pourraient couvrir le même
    # génome. On vérifie que cela ne crée pas de conflit de phénotype.
    duplicated_std = long.duplicated(
        ["species", "genome_id", "antibiotic_std"], keep=False
    )
    conflicting_std = 0
    if duplicated_std.any():
        conflicts = (
            long.loc[duplicated_std]
            .groupby(["species", "genome_id", "antibiotic_std"])["phenotype"]
            .nunique()
        )
        conflicting_std = int((conflicts > 1).sum())
        if conflicting_std:
            raise ValueError(
                f"{conflicting_std} conflits de phénotype après standardisation."
            )

    phenotype_counts = {
        "susceptible_0": int((long["phenotype"] == 0).sum()),
        "resistant_1": int((long["phenotype"] == 1).sum()),
    }
    return {
        "n_observations": int(len(long)),
        "n_unique_genomes": int(long["genome_id"].nunique()),
        "n_species": int(long["species"].nunique()),
        "n_cohorts": int(long["cohort"].nunique()),
        "n_antibiotics_raw": int(long["antibiotic"].nunique()),
        "n_antibiotics_std": int(long["antibiotic_std"].nunique()),
        "phenotype_counts": phenotype_counts,
        "prevalence": float(long["phenotype"].mean()),
        "duplicate_observations": duplicates,
        "conflicting_std_phenotypes": conflicting_std,
        "missing_values": int(long.isna().sum().sum()),
    }


def _make_wide(long: pd.DataFrame) -> pd.DataFrame:
    """Construit la matrice génomes × antibiotiques standardisés (Int8)."""
    # Les synonymes peuvent créer des doublons ; _quality_checks a déjà vérifié
    # qu'ils ne sont pas conflictuels.
    deduplicated = long.drop_duplicates(
        ["species", "genome_id", "antibiotic_std"], keep="first"
    )
    wide = deduplicated.pivot(
        index=["species", "genome_id"],
        columns="antibiotic_std",
        values="phenotype",
    )
    wide = wide.astype("Int8").reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["species", "genome_id"], kind="mergesort").reset_index(
        drop=True
    )
    return wide


def _drug_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par couple (espèce, antibiotique brut)."""
    summary = (
        long.groupby(
            ["species", "antibiotic", "antibiotic_std", "drug_class", "source_file"],
            sort=True,
        )
        .agg(
            n_total=("phenotype", "size"),
            n_resistant=("phenotype", "sum"),
            n_genomes=("genome_id", "nunique"),
        )
        .reset_index()
    )
    summary["n_susceptible"] = summary["n_total"] - summary["n_resistant"]
    summary["prevalence"] = summary["n_resistant"] / summary["n_total"]
    return summary.loc[
        :,
        [
            "species",
            "antibiotic",
            "antibiotic_std",
            "drug_class",
            "source_file",
            "n_total",
            "n_resistant",
            "n_susceptible",
            "n_genomes",
            "prevalence",
        ],
    ].sort_values(["species", "antibiotic"], kind="mergesort")


def _species_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Statistiques par espèce."""
    summary = (
        long.groupby("species", sort=True)
        .agg(
            n_observations=("phenotype", "size"),
            n_genomes=("genome_id", "nunique"),
            n_cohorts=("cohort", "nunique"),
            n_antibiotics_raw=("antibiotic", "nunique"),
            n_antibiotics_std=("antibiotic_std", "nunique"),
            n_resistant=("phenotype", "sum"),
        )
        .reset_index()
    )
    summary["n_susceptible"] = summary["n_observations"] - summary["n_resistant"]
    summary["prevalence"] = summary["n_resistant"] / summary["n_observations"]
    return summary


def _antibiotic_mapping(long: pd.DataFrame) -> pd.DataFrame:
    """Table de correspondance brut -> standardisé -> classe."""
    mapping = (
        long.groupby(["antibiotic", "antibiotic_std", "drug_class"], sort=True)
        .agg(
            n_species=("species", "nunique"),
            n_cohorts=("cohort", "nunique"),
            n_observations=("phenotype", "size"),
        )
        .reset_index()
    )
    mapping["antibiotic_label"] = mapping["antibiotic_std"].map(_display_name)
    return mapping.loc[
        :,
        [
            "antibiotic",
            "antibiotic_std",
            "antibiotic_label",
            "drug_class",
            "n_species",
            "n_cohorts",
            "n_observations",
        ],
    ].sort_values("antibiotic_std", kind="mergesort")


def _manifest(
    input_dir: Path,
    output_dir: Path,
    sources: list[dict],
    checks: dict,
    outputs: list[Path],
    by_species: list[Path],
) -> dict:
    """Construit le manifeste JSON de traçabilité."""
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/make_multispecies_antibiotic_dataset.py",
        "source": {
            "directory": str(input_dir.relative_to(PROJECT_ROOT))
            if input_dir.is_relative_to(PROJECT_ROOT)
            else str(input_dir),
            "n_files": len(sources),
            "files": sources,
        },
        "counts": checks,
        "schema": {
            "amr_long": {
                "description": "Une ligne par observation génome × antibiotique.",
                "columns": {
                    "genome_id": "Identifiant BV-BRC du génome, conservé en texte.",
                    "species": "Espèce bactérienne (11 valeurs).",
                    "antibiotic": "Nom brut de l'antibiotique, tel que dans le fichier source.",
                    "antibiotic_std": "Nom standardisé en snake_case (synonymes fusionnés).",
                    "drug_class": "Classe pharmacologique.",
                    "phenotype": "0 = sensible, 1 = résistant (int8).",
                    "cohort": "Cohorte source : espèce::antibiotique brut.",
                    "source_file": "Fichier d'origine.",
                    "source_row": "Numéro de ligne dans le fichier d'origine (1-based).",
                },
            },
            "amr_wide": {
                "description": "Matrice génomes × antibiotiques standardisés.",
                "index_columns": ["species", "genome_id"],
                "value_columns": (
                    f"{checks['n_antibiotics_std']} colonnes d'antibiotiques en "
                    "Int8 nullable (NA = non testé)."
                ),
            },
            "amr_by_species": {
                "description": "Sous-matrice large par espèce, colonnes non testées supprimées.",
            },
        },
        "outputs": [
            {
                "file": str(path.relative_to(output_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
        "outputs_by_species": [
            {
                "file": str(path.relative_to(output_dir)),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in by_species
        ],
        "notes": [
            "Les fichiers sources ne contiennent que le phénotype : aucune "
            "caractéristique génomique n'est incluse ici.",
            "Les cohortes sont propres à chaque couple espèce × antibiotique ; "
            "elles ne forment pas un jeu multi-étiquettes complet.",
            "Les synonymes ethiomide/ethionamide et "
            "cotrimoxazole/trimethoprim_sulfamethoxazole sont fusionnés dans "
            "antibiotic_std, mais les cohortes brutes restent distinctes.",
            "beta-lactam est un libellé de classe, pas une molécule unique.",
            "Les valeurs manquantes de la matrice large signifient « non testé », "
            "et non « sensible ».",
        ],
    }


def build_dataset(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_csv: bool = True,
) -> dict:
    """Construit toutes les sorties et retourne le manifeste."""
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    by_species_dir = output_dir / "amr_by_species"
    by_species_dir.mkdir(parents=True, exist_ok=True)

    long, sources = _build_long(input_dir)
    checks = _quality_checks(long)
    wide = _make_wide(long)

    drug_summary = _drug_summary(long)
    species_summary = _species_summary(long)
    mapping = _antibiotic_mapping(long)

    outputs: list[Path] = []
    long_path = output_dir / "amr_long.parquet"
    long.to_parquet(long_path, index=False)
    outputs.append(long_path)

    if write_csv:
        csv_path = output_dir / "amr_long.csv.gz"
        long.to_csv(csv_path, index=False, compression="gzip")
        outputs.append(csv_path)

    wide_path = output_dir / "amr_wide.parquet"
    wide.to_parquet(wide_path, index=False)
    outputs.append(wide_path)

    for name, frame in [
        ("amr_drug_summary.csv", drug_summary),
        ("amr_species_summary.csv", species_summary),
        ("amr_antibiotic_mapping.csv", mapping),
    ]:
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    by_species: list[Path] = []
    for species, frame in wide.groupby("species", sort=True):
        columns = ["genome_id"]
        columns += [
            column
            for column in frame.columns
            if column not in {"species", "genome_id"}
            and frame[column].notna().any()
        ]
        path = by_species_dir / f"{species}.parquet"
        frame.loc[:, columns].to_parquet(path, index=False)
        by_species.append(path)

    manifest = _manifest(
        input_dir=input_dir,
        output_dir=output_dir,
        sources=sources,
        checks=checks,
        outputs=outputs,
        by_species=by_species,
    )
    manifest_path = output_dir / "amr_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Assemble le jeu de données total de phénotypes d'antibiorésistance."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Dossier des fichiers Data_*.txt (défaut : {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Dossier de sortie (défaut : {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Ne pas écrire la version CSV compressée (Parquet uniquement).",
    )
    args = parser.parse_args(argv)

    manifest = build_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        write_csv=not args.no_csv,
    )
    counts = manifest["counts"]
    print("Jeu de données multi-espèces construit.")
    print(f"  observations      : {counts['n_observations']:,}")
    print(f"  génomes uniques   : {counts['n_unique_genomes']:,}")
    print(f"  espèces           : {counts['n_species']}")
    print(f"  antibiotiques     : {counts['n_antibiotics_raw']} bruts, "
          f"{counts['n_antibiotics_std']} standardisés")
    print(f"  prévalence        : {counts['prevalence']:.4f}")
    print(f"  sortie            : {args.output_dir}")


if __name__ == "__main__":
    main()
