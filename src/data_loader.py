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
"""

from pathlib import Path
from typing import Tuple
import pandas as pd


_SPLIT_FOLDERS = {
    "base": "splits",
    "snp": "splits_snp",
    "snp_3way": "splits_snp_3way",
}


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
