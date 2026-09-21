"""Entrenamiento de modelos para predecir la gravedad de siniestros.

Modelos:
    0. Dummy: clasificador trivial (referencia de piso)
    1. Baseline: Regresion Logistica
    2. Principal: HistGradientBoostingClassifier (boosting tipo XGBoost)
    3. Comparacion: MLP (red neuronal)

El problema esta desbalanceado; se usa class_weight='balanced' y las
metricas se calculan en evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
DROP_COLS = ["latitud", "longitud", "id", "fecha", "año", "gravedad"]


def load_dataset() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "processed" / "siniestros_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} no existe. Ejecutar primero src/features/build_features.py"
        )
    return pd.read_parquet(path)


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = [
        c for c in X.columns if str(X[c].dtype) in ("object", "category", "str")
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=0.01,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "dummy": Pipeline(
            [
                ("pre", None),
                (
                    "clf",
                    DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "logreg": Pipeline(
            [
                ("pre", None),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hgb": Pipeline(
            [
                ("pre", None),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_STATE, class_weight="balanced"
                    ),
                ),
            ]
        ),
        # Comparacion opcional con deep learning
        "mlp": Pipeline(
            [
                ("pre", None),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train(models: dict[str, Pipeline] | None = None) -> dict[str, float]:
    from src.features.splits import temporal_split

    df = load_dataset()
    train_df, _, _ = temporal_split(df)
    y_train = train_df.pop("target")
    X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])

    if models is None:
        models = build_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, float] = {}

    for name, pipe in models.items():
        pipe.set_params(pre=make_preprocessor(X_train))
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
        pipe.fit(X_train, y_train)
        results[name] = float(scores.mean())
        print(f"{name}: F1-macro CV={scores.mean():.3f} (+/- {scores.std():.3f})")

    MODELS_DIR.mkdir(exist_ok=True)
    for name, pipe in models.items():
        joblib.dump(
            {"pipeline": pipe, "X_columns": list(X_train.columns)},
            MODELS_DIR / f"{name}.joblib",
        )
    print(f"\nModelos guardados en {MODELS_DIR}")
    return results


if __name__ == "__main__":
    train()
