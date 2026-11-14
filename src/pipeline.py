"""Pipeline réutilisable d'entraînement / évaluation par antibiotique.

Le jeu de données `data/splits_all_snp_3way` contient :

- ``X_{split}.parquet`` : 34 744 features binaires (gènes, SNP, IGR) ;
- ``Y_{split}.parquet`` : 48 phénotypes d'antibiotiques, avec ``NA`` = non testé.

Chaque antibiotique est donc traité comme un problème binaire séparé :
``1`` = résistant, ``0`` = sensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SPLITS = ("train", "val", "test")


def prepare_data_for_atb(
    atb: str,
    data: dict[str, pd.DataFrame],
    min_samples: int = 100,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Filtre les individus testés pour ``atb`` sur les trois splits.

    ``data`` doit contenir les clés ``X_train``, ``Y_train``, ``X_val``,
    ``Y_val``, ``X_test`` et ``Y_test``.

    Returns
    -------
    dict
        ``{"train": (X_train_atb, y_train_atb), "val": (...), "test": (...)}``.
    """
    prepared: dict[str, tuple[pd.DataFrame, pd.Series]] = {}

    for split in SPLITS:
        X = data[f"X_{split}"]
        Y = data[f"Y_{split}"]

        if atb not in Y.columns:
            raise KeyError(
                f"'{atb}' absent de Y_{split}. "
                f"Colonnes disponibles : {Y.columns.tolist()}"
            )
        if len(X) != len(Y):
            raise ValueError(
                f"X_{split} ({len(X)} lignes) et Y_{split} ({len(Y)} lignes) "
                "n'ont pas la même longueur."
            )
        if not X.index.equals(Y.index):
            raise ValueError(
                f"X_{split} et Y_{split} n'ont pas le même index : "
                "impossible de filtrer les deux de façon fiable."
            )

        y = Y[atb]
        mask = y.notna()
        n = int(mask.sum())

        if n < min_samples:
            raise ValueError(
                f"{atb}/{split} : seulement {n} individus testés "
                f"(< min_samples={min_samples})."
            )

        X_atb = X.loc[mask].reset_index(drop=True)
        y_atb = y.loc[mask].astype(int).reset_index(drop=True)

        if y_atb.nunique() < 2:
            raise ValueError(
                f"{atb}/{split} : une seule classe présente "
                f"({sorted(y_atb.unique().tolist())})."
            )

        prepared[split] = (X_atb, y_atb)

    return prepared


def _best_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    metric: str = "f1",
    min_precision: float = 0.90,
) -> tuple[float, float, float, float]:
    """Cherche le seuil qui maximise une métrique sur la validation."""
    prec, rec, thr = precision_recall_curve(y_true, proba)

    if metric == "f1":
        scores = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    elif metric == "recall_at_precision":
        scores = np.where(prec >= min_precision, rec, -1.0)
    else:
        raise ValueError(
            "metric doit être 'f1' ou 'recall_at_precision', "
            f"reçu : {metric!r}"
        )

    if len(thr) == 0:
        return 0.5, 0.0, 0.0, 0.0

    # precision_recall_curve renvoie len(thr) + 1 points ; on aligne sur thr.
    scores = scores[:-1]
    best_idx = int(np.argmax(scores))

    return (
        float(thr[best_idx]),
        float(scores[best_idx]),
        float(prec[best_idx]),
        float(rec[best_idx]),
    )


def _metrics_dict(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Calcule les métriques binaires pour un seuil donné."""
    pred = (proba >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "threshold": float(threshold),
    }


@dataclass
class AntibioticResult:
    """Résultat complet du pipeline pour un antibiotique."""

    atb: str
    n_train: int
    n_val: int
    n_test: int
    prevalence_train: float
    prevalence_val: float
    prevalence_test: float
    models: dict[str, object]
    metrics_val: dict[str, dict[str, float]]
    metrics_test: dict[str, dict[str, float]]
    best_model_name: str
    best_threshold: float
    best_model: object
    shap_values: Optional[object] = None
    feature_importance: Optional[pd.DataFrame] = None


def run_antibiotic_pipeline(
    atb: str,
    data: dict[str, pd.DataFrame],
    models_to_try: Iterable[str] = ("xgb", "lr_l1"),
    threshold_metric: str = "f1",
    compute_shap: bool = True,
    shap_background: int = 1000,
    save_dir: Optional[Path] = None,
    random_state: int = 42,
    verbose: bool = True,
) -> AntibioticResult:
    """Entraîne, évalue et explique un modèle pour un antibiotique.

    Étapes :

    1. filtre les individus testés pour ``atb`` sur train / val / test ;
    2. entraîne XGBoost et/ou une régression logistique L1 ;
    3. choisit le modèle et le seuil qui maximisent la PR-AUC de validation ;
    4. évalue tous les modèles sur le test avec leur propre seuil ;
    5. calcule SHAP sur le test si le meilleur modèle est XGBoost ;
    6. sauvegarde le modèle et les importances si ``save_dir`` est fourni.
    """
    splits = prepare_data_for_atb(atb, data)

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    if verbose:
        print(f"\n=== {atb} ===")
        print(
            f"n_train={len(X_train)} | n_val={len(X_val)} | "
            f"n_test={len(X_test)}"
        )
        print(
            "prévalence train/val/test = "
            f"{y_train.mean():.2%} / {y_val.mean():.2%} / "
            f"{y_test.mean():.2%}"
        )

    pos_weight = float(
        (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    )

    factories = {
        "xgb": lambda: XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            tree_method="hist",
            eval_metric="aucpr",
            scale_pos_weight=pos_weight,
            random_state=random_state,
            n_jobs=-1,
        ),
        "lr_l1": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="liblinear",
                        l1_ratio=1.0,
                        C=0.05,
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    fitted: dict[str, object] = {}
    metrics_val: dict[str, dict[str, float]] = {}

    for name in models_to_try:
        if name not in factories:
            raise ValueError(
                f"Modèle inconnu : {name!r}. "
                f"Choix possibles : {sorted(factories)}"
            )

        model = factories[name]()
        model.fit(X_train, y_train)

        proba_val = model.predict_proba(X_val)[:, 1]
        threshold, *_ = _best_threshold(
            y_val, proba_val, metric=threshold_metric
        )
        metrics_val[name] = _metrics_dict(y_val, proba_val, threshold)
        fitted[name] = model

        if verbose:
            print(
                f"  {name:6s} | val PR-AUC={metrics_val[name]['pr_auc']:.3f} "
                f"| val F1={metrics_val[name]['f1']:.3f} "
                f"@ seuil={threshold:.3f}"
            )

    # Sélection sur la PR-AUC de validation : adaptée au déséquilibre.
    best_name = max(metrics_val, key=lambda key: metrics_val[key]["pr_auc"])
    best_threshold = metrics_val[best_name]["threshold"]
    best_model = fitted[best_name]

    if verbose:
        print(
            f"  → meilleur modèle : {best_name} "
            f"(seuil validation = {best_threshold:.3f})"
        )

    # Chaque modèle est évalué avec son propre seuil de validation.
    metrics_test: dict[str, dict[str, float]] = {}
    for name, model in fitted.items():
        proba_test = model.predict_proba(X_test)[:, 1]
        metrics_test[name] = _metrics_dict(
            y_test,
            proba_test,
            metrics_val[name]["threshold"],
        )

    if verbose:
        test = metrics_test[best_name]
        print(
            f"  TEST   | AUC={test['roc_auc']:.3f} "
            f"| PR-AUC={test['pr_auc']:.3f} "
            f"| F1={test['f1']:.3f} "
            f"| P={test['precision']:.3f} "
            f"| R={test['recall']:.3f}"
        )

    shap_values = None
    feature_importance = None

    if compute_shap and best_name == "xgb":
        import shap

        explainer = shap.TreeExplainer(best_model)
        n_background = min(shap_background, len(X_test))
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_test), size=n_background, replace=False)

        shap_values = explainer(X_test.iloc[idx])
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

        feature_importance = (
            pd.DataFrame(
                {
                    "feature": X_test.columns,
                    "mean_abs_shap": mean_abs_shap,
                }
            )
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = atb.replace("/", "_").replace(" ", "_")

        joblib.dump(
            {
                "model": best_model,
                "threshold": best_threshold,
                "atb": atb,
                "best_model_name": best_name,
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
            },
            save_dir / f"{safe_name}.joblib",
        )

        if feature_importance is not None:
            feature_importance.to_csv(
                save_dir / f"{safe_name}_importance.csv",
                index=False,
            )

    return AntibioticResult(
        atb=atb,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        prevalence_train=float(y_train.mean()),
        prevalence_val=float(y_val.mean()),
        prevalence_test=float(y_test.mean()),
        models=fitted,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        best_model_name=best_name,
        best_threshold=best_threshold,
        best_model=best_model,
        shap_values=shap_values,
        feature_importance=feature_importance,
    )
