"""Construye el dataset final de modelado a partir del ETL.

Uso:
    python src/features/build_features.py

Ejecuta el pipeline completo: raw CSV -> limpieza -> features -> parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.transform import (  # noqa: E402
    build_features,
    load_and_clean,
    save_processed,
)


def main() -> None:
    raw_csv = PROJECT_ROOT / "data" / "raw" / "caba_hechos.csv"
    if not raw_csv.exists():
        print(
            "No se encontro data/raw/caba_victimas.csv.\n"
            "Ejecutar primero: python src/etl/extract.py"
        )
        sys.exit(1)

    df_clean = load_and_clean(raw_csv)
    df_features = build_features(df_clean)
    save_processed(df_features)

    print("\nDistribucion del target:")
    print(df_features["target"].value_counts(normalize=True).round(3))
    print(f"\nColumnas finales: {list(df_features.columns)}")


if __name__ == "__main__":
    main()
