"""Module utilitaire pour charger rapidement les données préparées.

Exemples d'utilisation :
    from src.data_loader import load_train, load_test

    # Split 2-way d'origine
    X_train, y_train, meta_train = load_train()
    X_test, y_test, meta_test = load_test()

    # Split 2-way avec SNP
    X_train, y_train, meta_train = load_train(dataset="snp")

    # Split 3-way groupé par MLST
    X_train, y_train, meta_train = load_train(dataset="snp_3way")
    X_val, y_val, meta_val = load_validation(dataset="snp_3way")
    X_test, y_test, meta_test = load_test(dataset="snp_3way")

    # Jeu E. coli 50 antibiotiques (avec features)
    from src.data_loader import load_ecoli_antibiotics_labels, load_ecoli_antibiotics_ml
    labels = load_ecoli_antibiotics_labels()
    ml = load_ecoli_antibiotics_ml()

    # Fichier unique : une ligne par couple génome × antibiotique
    from src.data_loader import load_ecoli_antibiotics_long_features
    tout = load_ecoli_antibiotics_long_features()
    cipro = load_ecoli_antibiotics_long_features("ciprofloxacin")

    # Jeu multi-espèces (phénotypes seuls)
    from src.data_loader import (
        load_multispecies_antibiotics_long,
        load_multispecies_antibiotics_wide,
    )
    long = load_multispecies_antibiotics_long()
    ecoli = load_multispecies_antibiotics_wide(species="Escherichia_coli")
"""

from pathlib import Path
from typing import Tuple
import pandas as pd


_SPLIT_FOLDERS = {
    "base": "splits",
    "snp": "splits_snp",
    "snp_3way": "splits_snp_3way",
}

_MULTISPECIES_FOLDER = Path("data") / "processed" / "amr_multispecies"
_ECOLI_FOLDER = Path("data") / "processed" / "ecoli_all_antibiotics"


def _get_splits_dir(dataset: str = "base") -> Path:
  """Détecte l'emplacement du dossier de splits quel que soit le dossier courant.

  Args:
      dataset: "base" (data/splits), "snp" (data/splits_snp) ou
          "snp_3way" (data/splits_snp_3way).
  """
  if dataset not in _SPLIT_FOLDERS:
    raise ValueError(
        f"dataset inconnu : {dataset!r}. Choix possibles : {sorted(_SPLIT_FOLDERS)}"
    )
  folder = _SPLIT_FOLDERS[dataset]
  for candidate in [
      Path("data") / folder,
      Path("../data") / folder,
      Path("../../data") / folder,
  ]:
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
      f"Le dossier data/{folder} n'a pas été trouvé. Lancez d'abord le script"
      " de préparation des splits."
  )


def load_train(dataset: str = "base") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Charge les données d'entraînement prêtes à l'emploi.

  Args:
      dataset: "base" (pangénome seul) ou "snp" (pangénome + SNP QRDR).

  Returns:
      X_train (pd.DataFrame): 4 748 isolats x features (uint8).
      y_train (pd.Series): Phénotype cible (0 ou 1, int8).
      meta_train (pd.DataFrame): Métadonnées (sra, mlst_group, folds de CV pré-calculés).
  """
  splits_dir = _get_splits_dir(dataset)
  X_train = pd.read_parquet(splits_dir / "X_train.parquet")
  y_train = pd.read_parquet(splits_dir / "y_train.parquet")["y"]
  meta_train = pd.read_parquet(splits_dir / "metadata_train.parquet")
  return X_train, y_train, meta_train


def load_validation(dataset: str = "snp_3way") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Charge le jeu de validation du split 3-way groupé par MLST.

  Args:
      dataset: "snp_3way" (par défaut). Les autres jeux n'ont pas de validation.

  Returns:
      X_val (pd.DataFrame): 932 isolats x features (uint8).
      y_val (pd.Series): Phénotype cible (0 ou 1, int8).
      meta_val (pd.DataFrame): Métadonnées + split + fold_3way.
  """
  splits_dir = _get_splits_dir(dataset)
  X_val = pd.read_parquet(splits_dir / "X_val.parquet")
  y_val = pd.read_parquet(splits_dir / "y_val.parquet")["y"]
  meta_val = pd.read_parquet(splits_dir / "metadata_val.parquet")
  return X_val, y_val, meta_val


def load_test(dataset: str = "base") -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Charge les données de test (hold-out final).

  Args:
      dataset: "base" (pangénome seul) ou "snp" (pangénome + SNP QRDR).

  Returns:
      X_test (pd.DataFrame): 1 187 isolats x features (uint8).
      y_test (pd.Series): Phénotype cible réel (0 ou 1, int8).
      meta_test (pd.DataFrame): Métadonnées (sra, Genome.Name, mlst_group).
  """
  splits_dir = _get_splits_dir(dataset)
  X_test = pd.read_parquet(splits_dir / "X_test.parquet")
  y_test = pd.read_parquet(splits_dir / "y_test.parquet")["y"]
  meta_test = pd.read_parquet(splits_dir / "metadata_test.parquet")
  return X_test, y_test, meta_test


def _get_multispecies_dir() -> Path:
  """Détecte le dossier du jeu multi-espèces tous antibiotiques."""
  for candidate in [
      _MULTISPECIES_FOLDER,
      Path("..") / _MULTISPECIES_FOLDER,
      Path("../..") / _MULTISPECIES_FOLDER,
  ]:
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
      "Le dossier data/processed/amr_multispecies n'a pas été trouvé."
      " Lancez d'abord : python src/make_multispecies_antibiotic_dataset.py"
  )


def load_multispecies_antibiotics_long() -> pd.DataFrame:
  """Charge la table longue du jeu multi-espèces tous antibiotiques.

  Returns:
      pd.DataFrame: une ligne par observation génome × antibiotique, avec les
      colonnes genome_id, species, antibiotic, antibiotic_std, drug_class,
      phenotype, cohort, source_file et source_row.
  """
  return pd.read_parquet(_get_multispecies_dir() / "amr_long.parquet")


def load_multispecies_antibiotics_wide(species: str | None = None) -> pd.DataFrame:
  """Charge la matrice large génomes × antibiotiques.

  Args:
      species: si renseigné, charge uniquement la sous-matrice de cette espèce
          (par exemple "Escherichia_coli"). Sinon, charge la matrice globale.

  Returns:
      pd.DataFrame: colonnes genome_id + antibiotiques standardisés, en Int8
      nullable (NA = non testé). Pour une espèce, la colonne species est
      absente et les antibiotiques jamais testés sont retirés.
  """
  amr_dir = _get_multispecies_dir()
  if species is None:
    return pd.read_parquet(amr_dir / "amr_wide.parquet")

  path = amr_dir / "amr_by_species" / f"{species}.parquet"
  if not path.exists():
    available = sorted(
        item.stem for item in (amr_dir / "amr_by_species").glob("*.parquet")
    )
    raise FileNotFoundError(
        f"Espèce inconnue : {species!r}. Espèces disponibles : {available}"
    )
  return pd.read_parquet(path)


def _get_ecoli_antibiotics_dir() -> Path:
  """Détecte le dossier du jeu E. coli 50 antibiotiques."""
  for candidate in [
      _ECOLI_FOLDER,
      Path("..") / _ECOLI_FOLDER,
      Path("../..") / _ECOLI_FOLDER,
  ]:
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
      "Le dossier data/processed/ecoli_all_antibiotics n'a pas été trouvé."
      " Lancez d'abord : python src/make_ecoli_all_antibiotics_dataset.py"
  )


def load_ecoli_antibiotics_labels() -> pd.DataFrame:
  """Charge les labels binaires E. coli (6 983 génomes × 50 antibiotiques).

  Définition principale : Resistant = 1, Susceptible = 0, Intermediate et
  manquant = NA. Le fichier contient aussi les colonnes ``obs_*`` (masque
  binaire, FALSE pour un intermédiaire), ``n_tested`` (inclut les
  intermédiaires), ``n_observed`` (R ou S) et ``n_resistant``.
  """
  return pd.read_parquet(
      _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_labels.parquet"
  )


def load_ecoli_antibiotics_labels_ns() -> pd.DataFrame:
  """Charge les labels E. coli « non sensible » (Resistant ou Intermediate = 1)."""
  return pd.read_parquet(
      _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_labels_ns.parquet"
  )


def load_ecoli_antibiotics_mask() -> pd.DataFrame:
  """Charge le masque binaire (TRUE = label R ou S observé, FALSE = I ou manquant)."""
  return pd.read_parquet(
      _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_mask.parquet"
  )


def load_ecoli_antibiotics_long() -> pd.DataFrame:
  """Charge la table longue des labels E. coli observés (67 060 lignes)."""
  return pd.read_parquet(
      _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_long.parquet"
  )


def load_ecoli_antibiotics_ml() -> pd.DataFrame:
  """Charge le jeu E. coli prêt pour l'apprentissage (5 935 génomes).

  Returns:
      pd.DataFrame: métadonnées (sra, Genome.Name, mlst_group), split et
      fold_3way, 50 labels ``y_*`` et les 20 929 features pangénome + SNP QRDR.
  """
  return pd.read_parquet(
      _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_ml.parquet"
  )


def load_ecoli_antibiotics_long_features(
    antibiotic: str | None = None,
) -> pd.DataFrame:
  """Charge le fichier unique « une ligne par génome × antibiotique ».

  Args:
      antibiotic: nom snake_case d'un antibiotique (par exemple
          ``"ciprofloxacin"``). Si renseigné, seules les lignes de cet
          antibiotique sont lues. Sinon, charge les 62 666 lignes.

  Returns:
      pd.DataFrame: métadonnées (Genome.Name, sra, mlst_group, split,
      fold_3way), colonnes antibiotic_label, antibiotic, drug_class,
      resistance, raw_label, puis les 20 929 features pangénome + SNP.
  """
  path = _get_ecoli_antibiotics_dir() / "ecoli_all_antibiotics_long_features.parquet"
  if antibiotic is None:
      return pd.read_parquet(path)
  return pd.read_parquet(path, filters=[("antibiotic", "==", antibiotic)])

