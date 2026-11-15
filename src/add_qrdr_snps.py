#!/usr/bin/env python3
"""Ajoute des features SNP ciblées (QRDR / régulateurs) au jeu E. coli.

Le script lit un VCF (éventuellement gzip) issu du variant calling pangénomique,
garde les variants des contigs d'intérêt, encode chaque allèle alternatif en 0/1
pour chaque isolat SRA, puis fusionne le résultat avec le CSV/Parquet existant.

Le VCF produit par le pipeline est de type « présence de variant » : un génotype
`.` signifie « pas de variant détecté » (référence), pas « donnée manquante ».
On encode donc :

- 0 : allèle de référence / pas de variant détecté ;
- 1 : allèle alternatif détecté.

Les sites multi-alléliques sont éclatés : chaque ALT devient une feature distincte
avec sa propre annotation protéique.

Exemple :
    python src/add_qrdr_snps.py \
        --vcf data/raw/Ecoli_AMR_GenotypePhenotype/vcf_files/qrdr/qrdr_non_silent.vcf \
        --csv data/processed/ecoli_ciprofloxacin.csv \
        --out-csv data/processed/ecoli_ciprofloxacin_snp.csv \
        --out-parquet data/processed/ecoli_ciprofloxacin_snp.parquet \
        --min-carriers 5
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Contigs du pangénome associés aux mécanismes de résistance aux quinolones.
# Les noms de gènes sont ceux du pangénome (Roary), pas les noms canoniques.
TARGET_CONTIGS: Dict[str, str] = {
    "LMHECDEF_04343": "gyrA",
    "LMHECDEF_04466": "gyrB",
    "LMHECDEF_01124": "parC",
    "LMHECDEF_01105": "parE",
    "LMHECDEF_00662": "marR",
    "LMHECDEF_00326": "acrR_1",
    "LMHECDEF_00443": "acrR_2",
    "LMHECDEF_03751": "soxR",
}

SNP_COLUMNS = ["name", "contig", "pos", "ref", "alt", "effect", "hgvs_p", "carriers"]


def _open_text(path: Path):
    """Ouvre un fichier texte, gzip ou non."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def _parse_ann_by_allele(info: str) -> Dict[str, Tuple[str, str, str]]:
    """Retourne {allèle ALT: (effet, gène, HGVS.p)} depuis INFO/ANN de snpEff."""
    match = re.search(r"(?:^|;)ANN=([^;]+)", info)
    if not match:
        return {}
    annotations: Dict[str, Tuple[str, str, str]] = {}
    for ann in match.group(1).split(","):
        fields = ann.split("|")
        if len(fields) < 11:
            continue
        allele = fields[0]
        effect = fields[1] or ""
        gene = fields[3] or ""
        hgvs_p = fields[10] or ""
        annotations.setdefault(allele, (effect, gene, hgvs_p))
    return annotations


def _sanitize(value: str) -> str:
    """Rend un libellé utilisable comme nom de colonne."""
    value = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value)
    return value.strip("_")


def _read_samples(vcf_path: Path) -> List[str]:
    """Lit la ligne #CHROM du VCF et retourne les noms d'échantillons."""
    with _open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                return line.rstrip("\n").split("\t")[9:]
    raise ValueError(f"Ligne #CHROM introuvable dans {vcf_path}")


def _iter_variants(
    vcf_path: Path,
    sample_index: Dict[str, int],
    sample_ids: List[str],
    target_contigs: Dict[str, str],
    min_carriers: int,
    max_carriers: int,
) -> Iterable[Tuple[str, int, str, str, str, str, str, int, np.ndarray]]:
    """Parcourt le VCF et yield les allèles ALT ciblés encodés en 0/1.

    Yields:
        (contig, pos, ref, alt, effet, hgvs_p, carriers, valeurs)
    """
    n_samples = len(sample_ids)
    with _open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue
            chrom = cols[0]
            if chrom not in target_contigs:
                continue

            pos = int(cols[1])
            ref = cols[3]
            alts = cols[4].split(",")
            ann_by_allele = _parse_ann_by_allele(cols[7])

            fmt = cols[8].split(":")
            gt_index = fmt.index("GT") if "GT" in fmt else 0

            # Génotypes des seuls échantillons présents dans le CSV.
            genotypes: List[str] = []
            for sra in sample_ids:
                j = sample_index.get(sra)
                if j is None:
                    genotypes.append(".")
                    continue
                genotypes.append(cols[9 + j].split(":")[gt_index])

            for allele_index, alt in enumerate(alts, start=1):
                effect, _, hgvs_p = ann_by_allele.get(alt, ("", "", ""))
                # Le VCF « non_silent » peut contenir des ALT synonymes sur un
                # site multi-allélique : on les écarte.
                if effect == "synonymous_variant":
                    continue

                values = np.zeros(n_samples, dtype=np.uint8)
                for i, gt in enumerate(genotypes):
                    if gt in ("", "."):
                        continue
                    alleles = re.split(r"[/|]", gt)
                    if str(allele_index) in alleles:
                        values[i] = 1

                carriers = int(values.sum())
                if carriers < min_carriers or carriers > max_carriers:
                    continue

                yield chrom, pos, ref, alt, effect, hgvs_p, carriers, values


def _build_feature_names(
    records: List[Tuple[str, int, str, str, str, str, str, int, np.ndarray]],
) -> List[str]:
    """Construit des noms uniques du type SNP_gyrA_Ser83Leu."""
    base_names = []
    for contig, _pos, _ref, _alt, _effect, hgvs_p, _carriers, _values in records:
        gene = TARGET_CONTIGS[contig]
        if hgvs_p and hgvs_p not in (".", ""):
            label = hgvs_p[2:] if hgvs_p.startswith("p.") else hgvs_p
        else:
            label = "variant"
        base_names.append(f"SNP_{_sanitize(gene)}_{_sanitize(label)}")

    counts = Counter(base_names)
    names: List[str] = []
    for base, record in zip(base_names, records):
        if counts[base] == 1:
            names.append(base)
        else:
            contig, pos, _ref, alt, _effect, _hgvs_p, _carriers, _values = record
            gene = TARGET_CONTIGS[contig]
            names.append(f"SNP_{_sanitize(gene)}_{_sanitize(base.split('_', 2)[-1])}_{pos}_{_sanitize(alt)}")

    # Sécurité : deux ALT différents au même site peuvent produire le même nom.
    seen: Dict[str, int] = {}
    unique: List[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name}_{seen[name]}")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vcf", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-parquet", required=True, type=Path)
    parser.add_argument("--min-carriers", type=int, default=5,
                        help="Nombre minimal d'isolats porteurs (défaut : 5).")
    parser.add_argument("--max-carriers", type=int, default=0,
                        help="0 = n_samples - 1 (aucune borne haute).")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.vcf.exists():
        raise FileNotFoundError(args.vcf)
    if not args.csv.exists():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    if "sra" not in df.columns:
        raise ValueError("Le CSV doit contenir une colonne 'sra'")
    sample_ids = df["sra"].astype(str).tolist()
    sample_set = set(sample_ids)

    vcf_samples = _read_samples(args.vcf)
    sample_index = {name: i for i, name in enumerate(vcf_samples)}
    missing_samples = [s for s in sample_ids if s not in sample_index]
    extra_samples = [s for s in vcf_samples if s not in sample_set]

    max_carriers = args.max_carriers or (len(sample_ids) - 1)
    records = list(
        _iter_variants(
            args.vcf,
            sample_index,
            sample_ids,
            TARGET_CONTIGS,
            min_carriers=max(1, args.min_carriers),
            max_carriers=max_carriers,
        )
    )

    if not records:
        raise RuntimeError(
            "Aucun variant ciblé trouvé. Vérifie le nom des contigs dans le VCF "
            f"(attendu : {', '.join(TARGET_CONTIGS)})."
        )

    names = _build_feature_names(records)
    data = {name: r[7] for name, r in zip(names, records)}
    snp_df = pd.DataFrame(data)
    snp_df.insert(0, "sra", sample_ids)

    out = df.merge(snp_df, on="sra", how="left", validate="one_to_one")
    snp_cols = [c for c in snp_df.columns if c != "sra"]
    out[snp_cols] = out[snp_cols].fillna(0).astype(np.uint8)

    details = []
    for name, record in zip(names, records):
        contig, pos, ref, alt, effect, hgvs_p, carriers, _values = record
        details.append({
            "name": name,
            "contig": contig,
            "gene": TARGET_CONTIGS[contig],
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "effect": effect,
            "hgvs_p": hgvs_p,
            "carriers": carriers,
        })

    if args.dry_run:
        print(f"CSV : {df.shape}")
        print(f"Variants SNP retenus : {len(snp_cols)}")
        print(f"Échantillons CSV absents du VCF : {len(missing_samples)}")
        print(f"Échantillons VCF hors CSV : {len(extra_samples)}")
        for detail in details[:40]:
            print(
                f"{detail['name']}\t{detail['contig']}:{detail['pos']}\t"
                f"{detail['ref']}>{detail['alt']}\t{detail['effect']}\t"
                f"carriers={detail['carriers']}"
            )
        if len(details) > 40:
            print(f"... {len(details) - 40} autres variants")
        return

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    out.to_parquet(args.out_parquet, index=False)

    manifest = {
        "source_vcf": str(args.vcf),
        "source_csv": str(args.csv),
        "output_csv": str(args.out_csv),
        "output_parquet": str(args.out_parquet),
        "n_rows": int(out.shape[0]),
        "n_columns_total": int(out.shape[1]),
        "n_snp_features": int(len(snp_cols)),
        "n_pangenome_features": int(out.shape[1] - len(snp_cols) - 4),
        "snp_features": snp_cols,
        "snp_details": details,
        "target_contigs": TARGET_CONTIGS,
        "missing_csv_samples_in_vcf": missing_samples,
        "vcf_samples_not_in_csv": extra_samples,
        "min_carriers": int(args.min_carriers),
        "max_carriers": int(max_carriers),
        "encoding": "0 = reference / no variant call; 1 = alternate allele detected",
        "source_vcf_note": (
            "VCF de présence de variants : le génotype '.' signifie absence de "
            "variant détecté (référence), pas une donnée manquante."
        ),
    }
    manifest_path = args.manifest or args.out_parquet.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: manifest[k] for k in (
        "n_rows", "n_columns_total", "n_snp_features",
        "n_pangenome_features", "missing_csv_samples_in_vcf",
        "min_carriers",
    )}, indent=2, ensure_ascii=False))
    print(f"Manifeste : {manifest_path}")


if __name__ == "__main__":
    main()
