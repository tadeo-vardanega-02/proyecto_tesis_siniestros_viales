"""Splits de datos para entrenamiento y evaluacion.

Validacion temporal (por defecto en el proyecto):
    - Train:      años <= TRAIN_MAX_YEAR
    - Validacion: VAL_YEAR (tuning del umbral)
    - Test:       TEST_YEAR (evaluacion final)

Es mas honesto que un split aleatorio: simula el uso real del modelo
(predecir el futuro con datos pasados) y evita fuga temporal.
"""

from __future__ import annotations

import pandas as pd

TRAIN_MAX_YEAR = 2023
VAL_YEAR = 2024
TEST_YEAR = 2025


def temporal_split(
    df: pd.DataFrame, year_col: str = "año"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if year_col not in df.columns:
        raise ValueError(f"Falta la columna '{year_col}' (ejecutar ETL completo)")
    train = df[df[year_col] <= TRAIN_MAX_YEAR]
    val = df[df[year_col] == VAL_YEAR]
    test = df[df[year_col] == TEST_YEAR]
    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"Splits vacios con {year_col}: "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )
    print(
        f"Split temporal -> train<={TRAIN_MAX_YEAR}: {len(train):,} | "
        f"val={VAL_YEAR}: {len(val):,} | test={TEST_YEAR}: {len(test):,}"
    )
    return train, val, test


def xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa target y elimina columnas no-modelables."""
    from src.models.train_model import DROP_COLS

    y = df["target"]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return X, y
