#!/usr/bin/env python3
"""Construit le jeu de données *E. coli* « 50 antibiotiques ».

Source phénotypique :
``data/raw/Ecoli_AMR_GenotypePhenotype/phenotype_matrix/phenotype_matrix_08302024.csv``
(6 983 génomes × 50 antibiotiques, valeurs Resistant / Susceptible /
Intermediate / manquant).

Source des caractéristiques génomiques :
``data/processed/ecoli_ciprofloxacin_snp.parquet`` (5 935 génomes × 20 929
features pangénome + SNP QRDR), déjà utilisé par le projet.

Le script produit :

- ``ecoli_all_antibiotics_labels.parquet`` : 6 983 génomes × 50 labels
  binaires (R=1, S=0, Intermediate et manquant = NA) + masque d'observation ;
- ``ecoli_all_antibiotics_labels_ns.parquet`` : même chose avec la définition
  « non sensible » (Resistant ou Intermediate = 1) ;
- ``ecoli_all_antibiotics_long.parquet`` : une ligne par label observé, avec
  le libellé brut (R/S/I) pour audit ;
- ``ecoli_all_antibiotics_ml.parquet`` : les 5 935 génomes avec features,
  métadonnées et 50 labels prêts pour l'apprentissage ;
- des tableaux de synthèse et un manifeste de traçabilité.

Exemple :
    python src/make_ecoli_all_antibiotics_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHENOTYPE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Ecoli_AMR_GenotypePhenotype"
    / "phenotype_matrix"
    / "phenotype_matrix_08302024.csv"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Ecoli_AMR_GenotypePhenotype"
    / "strains_metadata.csv"
)
DEFAULT_FEATURES = (
    PROJECT_ROOT / "data" / "processed" / "ecoli_ciprofloxacin_snp.parquet"
)
DEFAULT_SPLITS = PROJECT_ROOT / "data" / "splits_snp_3way"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ecoli_all_antibiotics"

EXPECTED_N_DRUGS = 50
LABEL_MAP = {"Resistant": 1, "Susceptible": 0}
NON_SUSCEPTIBLE_MAP = {"Resistant": 1, "Intermediate": 1, "Susceptible": 0}
VALID_RAW_LABELS = {"Resistant", "Susceptible", "Intermediate"}

# Classes pharmacologiques, indexées par nom standardisé (snake_case).
DRUG_CLASSES = {
    "amikacin": "aminoglycoside",
    "amoxicillin": "beta-lactam",
    "amoxicillin_clavulanic_acid": "beta-lactam / inhibiteur",
    "ampicillin": "beta-lactam",
    "ampicillin_sulbactam": "beta-lactam / inhibiteur",
    "azithromycin": "macrolide",
    "aztreonam": "beta-lactam (monobactame)",
    "cefalexin": "beta-lactam (céphalosporine 1re génération)",
    "cefalotin": "beta-lactam (céphalosporine 1re génération)",
    "cefazolin": "beta-lactam (céphalosporine 1re génération)",
    "cefepime": "beta-lactam (céphalosporine 4e génération)",
    "cefotaxime": "beta-lactam (céphalosporine 3e génération)",
    "cefotaxime_clavulanic_acid": "beta-lactam / inhibiteur",
    "cefotetan": "beta-lactam (céphamycine)",
    "cefoxitin": "beta-lactam (céphamycine)",
    "cefpodoxime": "beta-lactam (céphalosporine 3e génération)",
    "ceftazidime": "beta-lactam (céphalosporine 3e génération)",
    "ceftriaxone": "beta-lactam (céphalosporine 3e génération)",
    "cefuroxime": "beta-lactam (céphalosporine 2e génération)",
    "cephalothin": "beta-lactam (céphalosporine 1re génération)",
    "chloramphenicol": "amphénicol",
    "ciprofloxacin": "fluoroquinolone",
    "colistin": "polymyxine",
    "doripenem": "carbapénème",
    "doxycycline": "tétracycline",
    "ertapenem": "carbapénème",
    "fosfomycin": "fosfomycine",
    "gentamicin": "aminoglycoside",
    "imipenem": "carbapénème",
    "kanamycin": "aminoglycoside",
    "levofloxacin": "fluoroquinolone",
    "meropenem": "carbapénème",
    "minocycline": "tétracycline",
    "moxifloxacin": "fluoroquinolone",
    "nalidixic_acid": "quinolone",
    "nitrofurantoin": "nitrofurane",
    "norfloxacin": "fluoroquinolone",
    "ofloxacin": "fluoroquinolone",
    "piperacillin": "beta-lactam (pénicilline)",
    "piperacillin_tazobactam": "beta-lactam / inhibiteur",
    "streptomycin": "aminoglycoside",
    "sulbactam": "inhibiteur de bêta-lactamase",
    "sulfamethoxazole": "sulfamide",
    "sulfamethoxazole_trimethoprim": "sulfamide / diaminopyrimidine",
    "tetracycline": "tétracycline",
    "ticarcillin_clavulanic_acid": "beta-lactam / inhibiteur",
    "tigecycline": "tétracycline (glycylcycline)",
    "tobramycin": "aminoglycoside",
    "trimethoprim": "diaminopyrimidine",
    "trimethoprim_sulfamethoxazole": "diaminopyrimidine / sulfamide",
}


def _sha256(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snake(name: str) -> str:
    """Transforme un libellé d'antibiotique en nom de colonne snake_case."""
    normalized = name.strip().lower()
    normalized = normalized.replace("/", "_").replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _read_phenotypes(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Lit et valide la matrice de phénotypes E. coli."""
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [str(column).strip() for column in frame.columns]

    if "Genome.Name" not in frame.columns:
        raise ValueError(f"Colonne Genome.Name absente de {path.name}")

    # La première colonne est un index de ligne sans nom.
    unnamed = [column for column in frame.columns if column.startswith("Unnamed")]
    frame = frame.drop(columns=unnamed)
    frame["Genome.Name"] = frame["Genome.Name"].str.strip()

    if frame["Genome.Name"].isna().any() or (frame["Genome.Name"] == "").any():
        raise ValueError(f"Genome.Name manquant dans {path.name}")
    if frame["Genome.Name"].duplicated().any():
        duplicated = frame.loc[frame["Genome.Name"].duplicated(), "Genome.Name"].head()
        raise ValueError(f"Genome.Name dupliqué dans {path.name} : {duplicated.tolist()}")

    drugs = [column for column in frame.columns if column != "Genome.Name"]
    if len(drugs) != EXPECTED_N_DRUGS:
        raise ValueError(
            f"{len(drugs)} antibiotiques trouvés dans {path.name}, "
            f"{EXPECTED_N_DRUGS} attendus"
        )

    for drug in drugs:
        frame[drug] = frame[drug].str.strip()
        values = set(frame[drug].dropna().unique())
        invalid = values.difference(VALID_RAW_LABELS)
        if invalid:
            raise ValueError(
                f"Valeurs inattendues pour {drug!r} dans {path.name} : {sorted(invalid)}"
            )

    return frame, drugs


def _read_metadata(path: Path) -> pd.DataFrame:
    """Lit les métadonnées BV-BRC et normalise les noms de colonnes."""
    frame = pd.read_csv(
        path,
        usecols=["Genome.ID", "Genome.Name", "SRA.Accession", "MLST"],
        dtype=str,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(
        columns={
            "Genome.ID": "bvbrc_genome_id",
            "SRA.Accession": "sra",
            "MLST": "mlst",
        }
    )
    frame["Genome.Name"] = frame["Genome.Name"].str.strip()
    frame["sra"] = frame["sra"].str.strip()
    frame["bvbrc_genome_id"] = frame["bvbrc_genome_id"].str.strip()
    frame["mlst"] = frame["mlst"].fillna("unknown").str.strip()

    if frame["Genome.Name"].duplicated().any():
        raise ValueError(f"Genome.Name dupliqué dans {path.name}")
    return frame


def _build_labels(
    phenotypes: pd.DataFrame,
    drugs: list[str],
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construit les matrices de labels, de masque et de non-sensibilité."""
    merged = metadata.merge(
        phenotypes[["Genome.Name"] + drugs],
        on="Genome.Name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(phenotypes):
        raise ValueError(
            "Tous les génomes de la matrice de phénotypes n'ont pas de métadonnées"
        )
    merged = merged.sort_values("Genome.Name", kind="mergesort").reset_index(drop=True)

    id_columns = ["Genome.Name", "sra", "bvbrc_genome_id", "mlst"]
    labels = merged[id_columns].copy()
    non_susceptible = merged[id_columns].copy()
    mask = merged[id_columns].copy()

    tested = merged[id_columns].copy()
    for drug in drugs:
        standard = _snake(drug)
        raw = merged[drug]
        binary = raw.map(LABEL_MAP)
        labels[f"y_{standard}"] = binary.astype("Int8")
        non_susceptible[f"y_ns_{standard}"] = raw.map(NON_SUSCEPTIBLE_MAP).astype(
            "Int8"
        )
        # Masque binaire : TRUE seulement si R ou S (Intermediate = FALSE).
        mask[f"obs_{standard}"] = binary.notna().astype("boolean")
        # « Testé » inclut les intermédiaires, contrairement au masque binaire.
        tested[f"tested_{standard}"] = raw.notna().astype("boolean")

    drug_columns = [f"y_{_snake(drug)}" for drug in drugs]
    mask_columns = [f"obs_{_snake(drug)}" for drug in drugs]
    ns_columns = [f"y_ns_{_snake(drug)}" for drug in drugs]
    tested_columns = [f"tested_{_snake(drug)}" for drug in drugs]

    labels["n_tested"] = tested[tested_columns].sum(axis=1).astype("int16")
    labels["n_observed"] = mask[mask_columns].sum(axis=1).astype("int16")
    labels["n_resistant"] = labels[drug_columns].sum(axis=1).astype("int16")
    non_susceptible["n_non_susceptible"] = (
        non_susceptible[ns_columns].sum(axis=1).astype("int16")
    )

    labels = pd.concat([labels, mask[mask_columns]], axis=1)
    return labels, non_susceptible, mask


def _build_long(
    phenotypes: pd.DataFrame,
    drugs: list[str],
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Table longue des labels observés (une ligne par génome × antibiotique)."""
    long = phenotypes.melt(
        id_vars=["Genome.Name"],
        value_vars=drugs,
        var_name="antibiotic_label",
        value_name="raw_label",
    )
    long = long.loc[long["raw_label"].notna()].copy()
    long["antibiotic"] = long["antibiotic_label"].map(_snake)
    long["drug_class"] = long["antibiotic"].map(DRUG_CLASSES).fillna("non classé")
    long["phenotype"] = long["raw_label"].map(LABEL_MAP).astype("Int8")
    long["phenotype_ns"] = long["raw_label"].map(NON_SUSCEPTIBLE_MAP).astype("Int8")
    long = long.merge(
        metadata[["Genome.Name", "sra", "bvbrc_genome_id", "mlst"]],
        on="Genome.Name",
        how="left",
        validate="many_to_one",
    )
    long = long.sort_values(
        ["Genome.Name", "antibiotic"], kind="mergesort"
    ).reset_index(drop=True)
    return long.loc[
        :,
        [
            "Genome.Name",
            "sra",
            "bvbrc_genome_id",
            "mlst",
            "antibiotic_label",
            "antibiotic",
            "drug_class",
            "raw_label",
            "phenotype",
            "phenotype_ns",
        ],
    ]


def _drug_summary(
    phenotypes: pd.DataFrame,
    drugs: list[str],
    feature_genomes: set[str],
) -> pd.DataFrame:
    """Statistiques par antibiotique, sur tous les génomes et sur le sous-ensemble features."""
    rows = []
    for drug in drugs:
        raw = phenotypes[drug]
        observed = raw.notna()
        raw_features = raw.loc[phenotypes["Genome.Name"].isin(feature_genomes)]

        rows.append(
            {
                "antibiotic_label": drug,
                "antibiotic": _snake(drug),
                "drug_class": DRUG_CLASSES.get(_snake(drug), "non classé"),
                "n_genomes": int(len(raw)),
                "n_observed": int(observed.sum()),
                "n_resistant": int((raw == "Resistant").sum()),
                "n_susceptible": int((raw == "Susceptible").sum()),
                "n_intermediate": int((raw == "Intermediate").sum()),
                "n_missing": int(raw.isna().sum()),
                "prevalence_R": (
                    float((raw == "Resistant").sum() / observed.sum())
                    if observed.any()
                    else np.nan
                ),
                "prevalence_NS": (
                    float(
                        ((raw == "Resistant") | (raw == "Intermediate")).sum()
                        / observed.sum()
                    )
                    if observed.any()
                    else np.nan
                ),
                "n_observed_with_features": int(raw_features.notna().sum()),
                "n_resistant_with_features": int((raw_features == "Resistant").sum()),
                "n_missing_with_features": int(raw_features.isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("antibiotic", kind="mergesort")


def _read_splits(splits_path: Path) -> pd.DataFrame:
    """Charge le split 3-way groupé par MLST (métadonnées train/val/test)."""
    frames = []
    for split in ["train", "val", "test"]:
        path = splits_path / f"metadata_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Fichier de split introuvable : {path}")
        frames.append(
            pd.read_parquet(
                path, columns=["sra", "mlst_group", "split", "fold_3way"]
            )
        )
    splits = pd.concat(frames, ignore_index=True)
    if splits["sra"].duplicated().any():
        raise ValueError("sra dupliqué entre les splits train/val/test")
    return splits


def _build_ml(
    features_path: Path,
    labels: pd.DataFrame,
    splits_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Fusionne les labels avec les features génomiques des 5 935 génomes."""
    features = pd.read_parquet(features_path)
    required = {"sra", "Genome.Name", "mlst_group", "y"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {features_path.name} : {sorted(missing)}")

    if features["Genome.Name"].duplicated().any():
        raise ValueError(f"Genome.Name dupliqué dans {features_path.name}")

    splits = _read_splits(splits_path)
    label_columns = [column for column in labels.columns if column.startswith("y_")]
    merged = features.merge(
        labels[["Genome.Name"] + label_columns],
        on="Genome.Name",
        how="left",
        validate="one_to_one",
    )

    mlst_check = features[["sra", "mlst_group"]].merge(
        splits[["sra", "mlst_group"]],
        on="sra",
        how="inner",
        suffixes=("_features", "_split"),
    )
    mlst_mismatches = int(
        (mlst_check["mlst_group_features"] != mlst_check["mlst_group_split"]).sum()
    )
    if mlst_mismatches:
        raise ValueError(
            f"{mlst_mismatches} groupes MLST incohérents entre les features et le split"
        )

    merged = merged.merge(
        splits[["sra", "split", "fold_3way"]],
        on="sra",
        how="left",
        validate="one_to_one",
    )

    cipro = "y_ciprofloxacin"
    if cipro not in merged.columns:
        raise ValueError("Label ciprofloxacine absent du jeu 50 antibiotiques")

    comparable = merged["y"].notna() & merged[cipro].notna()
    mismatches = int((merged.loc[comparable, "y"] != merged.loc[comparable, cipro]).sum())
    if mismatches:
        raise ValueError(
            f"{mismatches} labels ciprofloxacine incohérents entre le jeu historique "
            "et la matrice 50 antibiotiques"
        )

    feature_columns = [
        column
        for column in features.columns
        if column not in {"sra", "Genome.Name", "mlst_group", "y"}
    ]
    merged = merged.loc[
        :,
        ["sra", "Genome.Name", "mlst_group", "split", "fold_3way"]
        + label_columns
        + feature_columns,
    ]
    checks = {
        "n_feature_genomes": int(len(features)),
        "n_feature_genomes_matched": int(merged[label_columns].notna().any(axis=1).sum()),
        "n_cipro_comparable": int(comparable.sum()),
        "cipro_mismatches": mismatches,
        "n_split_genomes": int(len(splits)),
        "n_split_genomes_matched": int(merged["split"].notna().sum()),
        "mlst_group_mismatches": mlst_mismatches,
    }
    return merged, checks


def _manifest(
    output_dir: Path,
    sources: dict[str, Path],
    counts: dict,
    checks: dict,
    outputs: list[Path],
) -> dict:
    """Manifeste de traçabilité du jeu E. coli 50 antibiotiques."""
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "src/make_ecoli_all_antibiotics_dataset.py",
        "source": {
            "phenotype_matrix": str(sources["phenotypes"]),
            "metadata": str(sources["metadata"]),
            "features": str(sources["features"]),
            "splits": str(sources["splits"]),
            "split_files": {
                path.name: _sha256(path)
                for path in sorted(sources["splits"].glob("metadata_*.parquet"))
            },
            "sha256": {
                name: _sha256(path)
                for name, path in sources.items()
                if path.is_file()
            },
        },
        "counts": counts,
        "checks": checks,
        "schema": {
            "labels": (
                "Genome.Name, sra, bvbrc_genome_id, mlst, n_tested, n_observed, "
                "n_resistant, 50 colonnes y_<antibiotique> (Int8) et 50 colonnes "
                "obs_<antibiotique> (TRUE si R ou S, FALSE si Intermediate ou manquant)."
            ),
            "labels_ns": (
                "Mêmes métadonnées + 50 colonnes y_ns_<antibiotique> "
                "(1 = Resistant ou Intermediate, 0 = Susceptible, NA = manquant)."
            ),
            "long": (
                "Une ligne par label observé : Genome.Name, sra, bvbrc_genome_id, "
                "mlst, antibiotic_label, antibiotic, drug_class, raw_label, "
                "phenotype, phenotype_ns."
            ),
            "ml": (
                "5 935 génomes avec sra, Genome.Name, mlst_group, split, "
                "fold_3way, 50 labels y_* et les 20 929 features pangénome + SNP QRDR."
            ),
        },
        "outputs": [
            {
                "file": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
            for path in outputs
        ],
        "notes": [
            "Définition principale : Resistant = 1, Susceptible = 0, "
            "Intermediate = NA (exclu).",
            "Définition alternative : y_ns_* regroupe Resistant et Intermediate "
            "comme non sensibles.",
            "Les 1 048 génomes sans features génomiques restent disponibles dans "
            "les fichiers de labels et la table longue.",
            "obs_* est le masque binaire (R ou S) à utiliser pour la perte ; "
            "n_tested compte aussi les intermédiaires, n_observed non.",
            "Les labels sont très creux : de nombreux antibiotiques ne sont testés "
            "que sur quelques dizaines de génomes.",
            "Le fichier ML reprend le split 3-way groupé par MLST déjà utilisé pour "
            "la ciprofloxacine (colonnes split et fold_3way).",
            "Ce jeu E. coli est distinct du jeu multi-espèces : les cohortes et "
            "les définitions de phénotype ne doivent pas être mélangées sans contrôle.",
        ],
    }


def build_dataset(
    phenotype_path: Path = DEFAULT_PHENOTYPE,
    metadata_path: Path = DEFAULT_METADATA,
    features_path: Path = DEFAULT_FEATURES,
    splits_path: Path = DEFAULT_SPLITS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_csv: bool = True,
) -> dict:
    """Construit toutes les sorties et retourne le manifeste."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phenotypes, drugs = _read_phenotypes(phenotype_path)
    metadata = _read_metadata(metadata_path)
    labels, non_susceptible, mask = _build_labels(phenotypes, drugs, metadata)
    long = _build_long(phenotypes, drugs, metadata)
    feature_genomes = set(
        pd.read_parquet(features_path, columns=["Genome.Name"])["Genome.Name"]
    )
    summary = _drug_summary(phenotypes, drugs, feature_genomes)
    ml, feature_checks = _build_ml(features_path, labels, splits_path)

    outputs: list[Path] = []

    def _write(frame: pd.DataFrame, name: str) -> None:
        path = output_dir / name
        frame.to_parquet(path, index=False)
        outputs.append(path)

    _write(labels, "ecoli_all_antibiotics_labels.parquet")
    _write(non_susceptible, "ecoli_all_antibiotics_labels_ns.parquet")
    _write(mask, "ecoli_all_antibiotics_mask.parquet")
    _write(long, "ecoli_all_antibiotics_long.parquet")
    _write(ml, "ecoli_all_antibiotics_ml.parquet")

    if write_csv:
        for frame, name in [
            (labels, "ecoli_all_antibiotics_labels.csv.gz"),
            (long, "ecoli_all_antibiotics_long.csv.gz"),
        ]:
            path = output_dir / name
            frame.to_csv(path, index=False, compression="gzip")
            outputs.append(path)

    summary_path = output_dir / "ecoli_all_antibiotics_drug_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    counts = {
        "n_genomes": int(len(phenotypes)),
        "n_genomes_with_features": int(feature_checks["n_feature_genomes"]),
        "n_drugs": int(len(drugs)),
        "n_long_observations": int(len(long)),
        "n_binary_labels": int(long["phenotype"].notna().sum()),
        "n_resistant": int((long["raw_label"] == "Resistant").sum()),
        "n_susceptible": int((long["raw_label"] == "Susceptible").sum()),
        "n_intermediate": int((long["raw_label"] == "Intermediate").sum()),
        "n_missing_labels": int(
            len(phenotypes) * len(drugs) - len(long)
        ),
        "n_genomes_with_at_least_one_label": int(
            (labels[[c for c in labels.columns if c.startswith("y_")]]
             .notna().any(axis=1)).sum()
        ),
        "mean_drugs_tested_per_genome": float(labels["n_tested"].mean()),
        "median_drugs_tested_per_genome": float(labels["n_tested"].median()),
    }
    checks = {
        "duplicate_genome_names": 0,
        "invalid_raw_labels": 0,
        **feature_checks,
    }
    manifest = _manifest(
        output_dir=output_dir,
        sources={
            "phenotypes": phenotype_path,
            "metadata": metadata_path,
            "features": features_path,
            "splits": splits_path,
        },
        counts=counts,
        checks=checks,
        outputs=outputs,
    )
    manifest_path = output_dir / "ecoli_all_antibiotics_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(
        description="Construit le jeu E. coli 50 antibiotiques avec features génomiques."
    )
    parser.add_argument(
        "--phenotypes",
        type=Path,
        default=DEFAULT_PHENOTYPE,
        help=f"Matrice de phénotypes (défaut : {DEFAULT_PHENOTYPE}).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help=f"Métadonnées BV-BRC (défaut : {DEFAULT_METADATA}).",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES,
        help=f"Features génomiques (défaut : {DEFAULT_FEATURES}).",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=DEFAULT_SPLITS,
        help=f"Dossier du split 3-way (défaut : {DEFAULT_SPLITS}).",
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
        help="Ne pas écrire les CSV compressés (labels et table longue).",
    )
    args = parser.parse_args(argv)

    manifest = build_dataset(
        phenotype_path=args.phenotypes,
        metadata_path=args.metadata,
        features_path=args.features,
        splits_path=args.splits,
        output_dir=args.output_dir,
        write_csv=not args.no_csv,
    )
    counts = manifest["counts"]
    print("Jeu de données E. coli 50 antibiotiques construit.")
    print(f"  génomes (labels)        : {counts['n_genomes']:,}")
    print(f"  génomes (features)      : {counts['n_genomes_with_features']:,}")
    print(f"  antibiotiques           : {counts['n_drugs']}")
    print(f"  labels observés         : {counts['n_long_observations']:,}")
    print(f"  résistants / sensibles  : {counts['n_resistant']:,} / "
          f"{counts['n_susceptible']:,}")
    print(f"  intermédiaires          : {counts['n_intermediate']:,}")
    print(f"  sortie                  : {args.output_dir}")


if __name__ == "__main__":
    main()
