"""Module utilitaire pour charger rapidement les données préparées.

Exemple d'utilisation :
    from src.data_loader import load_train, load_test

    X_train, y_train, meta_train = load_train()
    X_test, y_test, meta_test = load_test()
"""

from pathlib import Path
from typing import Tuple
import pandas as pd


def _get_splits_dir() -> Path:
  """Détecte l'emplacement du dossier data/splits quel que soit le dossier courant."""
  for candidate in [
      Path("data/splits"),
      Path("../data/splits"),
      Path("../../data/splits"),
  ]:
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
      "Le dossier data/splits n'a pas été trouvé. Lancez d'abord le script de"
      " préparation des splits."
  )


def load_train() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Charge les données d'entraînement prêtes à l'emploi.

  Returns:
      X_train (pd.DataFrame): 4 748 isolats x 20 731 features (uint8).
      y_train (pd.Series): Phénotype cible (0 ou 1, int8).
      meta_train (pd.DataFrame): Métadonnées (sra, mlst_group, folds de CV pré-calculés).
  """
  splits_dir = _get_splits_dir()
  X_train = pd.read_parquet(splits_dir / "X_train.parquet")
  y_train = pd.read_parquet(splits_dir / "y_train.parquet")["y"]
  meta_train = pd.read_parquet(splits_dir / "metadata_train.parquet")
  return X_train, y_train, meta_train


def load_test() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
  """Charge les données de test (hold-out final).

  Returns:
      X_test (pd.DataFrame): 1 187 isolats x 20 731 features (uint8).
      y_test (pd.Series): Phénotype cible réel (0 ou 1, int8).
      meta_test (pd.DataFrame): Métadonnées (sra, Genome.Name, mlst_group).
  """
  splits_dir = _get_splits_dir()
  X_test = pd.read_parquet(splits_dir / "X_test.parquet")
  y_test = pd.read_parquet(splits_dir / "y_test.parquet")["y"]
  meta_test = pd.read_parquet(splits_dir / "metadata_test.parquet")
  return X_test, y_test, meta_test
