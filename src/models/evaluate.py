"""Evaluacion de modelos: metricas para clases desbalanceadas + interpretabilidad.

Uso:
    python src/models/evaluate.py

Pasos:
    1. Metricas por modelo (dummy incluido como piso de referencia).
    2. Ajuste del umbral de decision: se busca en CV el umbral que maximiza
       F1 de la clase fatal y luego se evalua en test con ese umbral.
    3. Curvas Precision-Recall comparativas.
    4. Interpretabilidad: importancia por permutacion.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE = 42
DROP_COLS = ["latitud", "longitud", "id", "fecha", "anio", "gravedad"]


def load_splits():
    """Carga el dataset y devuelve (X_train, y_train), (X_val, y_val),
    (X_test, y_test) con validacion temporal."""
    from src.features.splits import temporal_split

    path = PROJECT_ROOT / "data" / "processed" / "siniestros_features.parquet"
    df = pd.read_parquet(path)
    tr_df, va_df, te_df = temporal_split(df)
    splits = []
    for part in (tr_df, va_df, te_df):
        y = part.pop("target")
        X = part.drop(columns=[c for c in DROP_COLS if c in part.columns])
        splits.append((X, y))
    return splits[0], splits[1], splits[2]


def evaluate_model(name: str, X_test, y_test, threshold: float | None = None) -> dict:
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        print(f"[skip] {name}: no entrenado")
        return {}
    pipe = joblib.load(model_path)["pipeline"]

    scores = pipe.predict_proba(X_test)[:, 1]
    thr = threshold if threshold is not None else 0.5
    y_pred = (scores >= thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "threshold": float(thr),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "recall_fatal": cm[1, 1] / max(cm[1].sum(), 1),
        "precision_fatal": cm[1, 1] / max(cm[1, 1] + cm[0, 1], 1),
        "auc_pr": average_precision_score(y_test, scores),
    }
    try:
        metrics["auc_roc"] = roc_auc_score(y_test, scores)
    except ValueError:
        pass

    print(f"\n=== {name} (umbral={thr:.3f}) ===")
    for k, v in metrics.items():
        if k != "threshold":
            print(f"  {k}: {v:.3f}")
    print(classification_report(y_test, y_pred, digits=3))
    return metrics


def tune_threshold(name: str, X_val, y_val, plot: bool = True) -> float:
    """Busca el umbral que maximiza F1 de la clase fatal usando un set de validacion.

    El umbral optimo depende del costo relativo: priorizar recall (no dejar
    pasar fatales) o precision (evitar alarmas falsas). Aqui se usa F1 como
    compromiso; ajustar beta en fbeta_score si el negocio lo requiere.
    """
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return 0.5
    pipe = joblib.load(path)["pipeline"]
    scores = pipe.predict_proba(X_val)[:, 1]

    prec, rec, thresholds = precision_recall_curve(y_val, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_fatal = np.where(
            (prec + rec) > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0
        )
    # El ultimo punto de prec/rec no tiene umbral asociado
    best_idx = int(np.argmax(f1_fatal[:-1]))
    best_thr = float(thresholds[best_idx])

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, f1_fatal[:-1], label="F1 clase fatal")
        ax.plot(thresholds, prec[:-1], "--", alpha=0.6, label="Precision")
        ax.plot(thresholds, rec[:-1], "--", alpha=0.6, label="Recall")
        ax.axvline(best_thr, color="red", ls=":",
                   label=f"Optimo={best_thr:.3f}")
        ax.set_xlabel("Umbral de decision")
        ax.set_ylabel("Metrica")
        ax.set_title(f"Ajuste de umbral: {name} (set de validacion)")
        ax.legend()
        plt.tight_layout()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        dest = FIGURES_DIR / f"threshold_{name}.png"
        fig.savefig(dest, dpi=120)
        plt.close(fig)
        print(f"Grafico de umbrales guardado en {dest}")

    print(f"[{name}] umbral optimo (F1 fatal): {best_thr:.3f} "
          f"(default=0.500)")
    return best_thr


def plot_pr_curves(names: list[str], X_test, y_test) -> None:
    plt.figure(figsize=(8, 6))
    for name in names:
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            continue
        pipe = joblib.load(path)["pipeline"]
        scores = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, scores)
        ap = average_precision_score(y_test, scores)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    # Piso teorico: prevalencia de la clase positiva
    prevalence = float(np.mean(y_test))
    plt.axhline(prevalence, color="gray", ls=":", lw=1,
                label=f"Piso (prevalencia={prevalence:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall: clase fatal (desbalanceada)")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dest = FIGURES_DIR / "pr_curves.png"
    plt.savefig(dest, dpi=120)
    print(f"Curvas PR guardadas en {dest}")


def feature_importance(name: str, X_test, y_test, top_n: int = 15) -> None:
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        return
    bundle = joblib.load(path)
    pipe, cols = bundle["pipeline"], bundle["X_columns"]
    result = permutation_importance(pipe, X_test, y_test, n_repeats=5,
                                    random_state=RANDOM_STATE, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1][:top_n]
    print(f"\nTop variables ({name}):")
    for i in order:
        print(f"  {cols[i]}: {result.importances_mean[i]:.4f}")


def main() -> None:
    names = ["dummy", "logreg", "hgb", "mlp"]
    # Validacion temporal: modelos entrenados <=2023, umbral ajustado en 2024,
    # evaluacion final en 2025.
    _, (X_val, y_val), (X_test, y_test) = load_splits()

    # 1) Ajuste de umbral en validacion para modelos con senal real
    thresholds: dict[str, float] = {}
    for name in ["logreg", "hgb"]:
        thresholds[name] = tune_threshold(name, X_val, y_val)

    # 2) Evaluacion final en test con el umbral optimo de cada modelo
    all_metrics: dict[str, dict] = {}
    for name in names:
        m = evaluate_model(name, X_test, y_test, threshold=thresholds.get(name))
        if m:
            all_metrics[name] = m

    plot_pr_curves(names, X_test, y_test)
    if "hgb" in all_metrics:
        feature_importance("hgb", X_test, y_test)


if __name__ == "__main__":
    main()
