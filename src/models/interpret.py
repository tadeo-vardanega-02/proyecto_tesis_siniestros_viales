"""Interpretabilidad con SHAP del modelo principal (HistGradientBoosting).

Uso:
    python src/models/interpret.py

Genera:
    - reports/figures/shap_bar.png      importancia global por variable
    - reports/figures/shap_beeswarm.png direccion y magnitud del efecto
    - Impresion de las top variables con su impacto medio.
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
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RANDOM_STATE = 42
MODEL_NAME = "hgb"
N_SAMPLES = 2000


def load_shap_inputs():
    """Muestra del set de test temporal (2025) para explicar predicciones
    sobre datos que el modelo nunca vio."""
    from src.models.evaluate import load_splits

    (_, _), (_, _), (X_test, y_test) = load_splits()
    sample_idx = X_test.sample(min(N_SAMPLES, len(X_test)),
                               random_state=RANDOM_STATE).index

    bundle = joblib.load(MODELS_DIR / f"{MODEL_NAME}.joblib")
    pipe = bundle["pipeline"]
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    X_trans = pre.transform(X_test.loc[sample_idx])
    feat_names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
    X_df = pd.DataFrame(X_trans, columns=feat_names, index=sample_idx)
    return X_df, clf


def main() -> None:
    X_df, clf = load_shap_inputs()
    print(f"SHAP sobre {len(X_df)} muestras, {X_df.shape[1]} features")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Importancia global (media del |SHAP|)
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_df, plot_type="bar", show=False,
                      max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_bar.png", dpi=120)
    plt.close("all")

    # Direccion del efecto
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_df, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=120)
    plt.close("all")

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    print("\nTop 15 variables por impacto medio |SHAP|:")
    for i in order:
        print(f"  {X_df.columns[i]}: {mean_abs[i]:.4f}")

    print(f"\nFiguras guardadas en {FIGURES_DIR}")


if __name__ == "__main__":
    main()
