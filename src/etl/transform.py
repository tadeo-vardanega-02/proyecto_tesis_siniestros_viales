"""Limpieza y feature engineering de siniestros viales (CABA - datos abiertos).

Uso:
    python src/etl/transform.py

Fuentes que integra:
    - caba_hechos.csv: un registro por siniestro (tabla principal).
    - caba_victimas.csv: un registro por victima; se agrega por siniestro.
      IMPORTANTE: no se usan GRAVEdad_victima ni fecha_fallecimiento_victima
      porque determinan el target (fuga de informacion).
    - clima_caba.csv: clima diario (Open-Meteo); se une por fecha.

Target binario: gravedad MORTAL vs resto.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILE = RAW_DIR / "caba_hechos.csv"
VICTIMAS_FILE = RAW_DIR / "caba_victimas.csv"
CLIMA_FILE = RAW_DIR / "clima_caba.csv"

# Renombres al esquema estandar del proyecto
RENAME_MAP: dict[str, str] = {
    "id_siniestro": "id",
    "fecha_siniestro": "fecha",
    "hora_siniestro": "hora",
    "comuna_siniestro": "comuna",
    "tipo_de_via_siniestro": "tipo_calle",
    "latitud_siniestro": "latitud",
    "longitud_siniestro": "longitud",
    "participantes_siniestro": "participantes",
    "contraparte_siniestro": "contraparte",
    "gravedad_siniestro": "gravedad",
}

FEATURE_CANDIDATES = [
    # hechos
    "tipo_calle", "comuna", "participantes", "contraparte",
    "hora_num", "mes", "dia_semana", "es_fin_de_semana", "franja_horaria",
    # agregados de victimas
    "n_victimas", "edad_promedio", "pct_moto", "pct_peaton", "pct_sexo_m",
    # clima
    "lluvia_mm", "temp_max",
]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or RAW_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe. Ejecutar src/etl/extract.py")
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(columns=RENAME_MAP)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["hora_num"] = (
        df["hora"].astype(str).str.split(":").str[0]
        .pipe(pd.to_numeric, errors="coerce")
        .clip(0, 23)
    )
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=lunes
    df["anio"] = df["fecha"].dt.year
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)

    bins = [-1, 5, 11, 17, 24]
    labels = ["madrugada", "manana", "tarde", "noche"]
    df["franja_horaria"] = pd.cut(df["hora_num"], bins=bins, labels=labels)
    return df


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_cols = [c for c in ["tipo_calle", "comuna", "participantes",
                             "contraparte", "gravedad"] if c in df.columns]
    for col in text_cols:
        df[col] = (
            df[col].astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "SD": np.nan, "": np.nan})
        )
    # Comunas: 'Comuna 8' -> 8 (numerico)
    if "comuna" in df.columns:
        df["comuna"] = df["comuna"].str.extract(r"(\d+)").astype(float)
    return df


def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    in_caba = df["latitud"].between(-35, -34) & df["longitud"].between(-59, -58)
    df.loc[~in_caba & df[["latitud", "longitud"]].notna().all(axis=1),
           ["latitud", "longitud"]] = np.nan
    return df


def load_victimas_agg(path: Path | None = None) -> pd.DataFrame:
    """Agrega la tabla de victimas a nivel siniestro.

    Genera: n_victimas, edad_promedio, pct_moto, pct_peaton, pct_sexo_m.
    Excluye deliberadamente columnas que determinan el target.
    """
    path = path or VICTIMAS_FILE
    v = pd.read_csv(path, sep=";", low_memory=False)

    # 'SD' y valores implausibles -> NaN
    edad = pd.to_numeric(v["edad_victima"], errors="coerce")
    edad = edad.where(edad.between(0, 100))
    sexo_m = (v["sexo_victima"].astype(str).str.upper() == "M").astype(int)
    es_moto = (v["modo_desplazamiento_victima"].astype(str) == "MOTO").astype(int)
    es_peaton = (v["modo_desplazamiento_victima"].astype(str) == "PEATON").astype(int)

    agg = pd.DataFrame({
        "id": v["id_siniestro"],
        "edad_num": edad,
        "sexo_m": sexo_m,
        "es_moto": es_moto,
        "es_peaton": es_peaton,
    })
    out = agg.groupby("id").agg(
        n_victimas=("id", "size"),
        edad_promedio=("edad_num", "mean"),
        pct_sexo_m=("sexo_m", "mean"),
        pct_moto=("es_moto", "mean"),
        pct_peaton=("es_peaton", "mean"),
    ).reset_index()
    return out


def load_clima(path: Path | None = None) -> pd.DataFrame:
    """Carga el clima diario y lo prepara para el join por fecha."""
    path = path or CLIMA_FILE
    c = pd.read_csv(path)
    c = c.rename(columns={
        "time": "fecha",
        "precipitation_sum": "lluvia_mm",
        "temperature_2m_max": "temp_max",
    })
    c["fecha"] = pd.to_datetime(c["fecha"])
    return c[["fecha", "lluvia_mm", "temp_max"]]


def load_and_clean(path: Path | None = None,
                   with_enrichment: bool = True) -> pd.DataFrame:
    df = load_raw(path)
    df = parse_datetime(df)
    df = clean_categories(df)
    df = validate_coordinates(df)

    # Regla de negocio del target: fatal si la gravedad registrada es MORTAL
    df["target"] = (df["gravedad"] == "MORTAL").astype(int)

    if with_enrichment:
        try:
            victimas = load_victimas_agg()
            df = df.merge(victimas, on="id", how="left")
            print(f"Enriquecido con victimas: {len(victimas):,} siniestros con datos agregados")
        except FileNotFoundError as exc:
            print(f"[warn] victimas no disponible: {exc}")
        try:
            clima = load_clima()
            df["fecha_dia"] = df["fecha"].dt.normalize()
            df = df.merge(clima.rename(columns={"fecha": "fecha_dia"}),
                          on="fecha_dia", how="left")
            df = df.drop(columns=["fecha_dia"])
            cobertura = df["lluvia_mm"].notna().mean()
            print(f"Enriquecido con clima: {cobertura:.1%} de cobertura por fecha")
        except FileNotFoundError as exc:
            print(f"[warn] clima no disponible: {exc}")

    print(f"Limpieza OK: {len(df):,} filas, {len(df.columns)} columnas")
    print("Distribucion del target:", df['target'].value_counts(normalize=True).round(4).to_dict())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in FEATURE_CANDIDATES if c in df.columns] + ["target"]
    extra = [c for c in ["latitud", "longitud", "id", "fecha", "anio",
                         "gravedad"] if c in df.columns]
    return df[cols + extra].copy()


def save_processed(df: pd.DataFrame) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = PROCESSED_DIR / "siniestros_features.parquet"
    df.to_parquet(dest, index=False)
    print(f"Dataset procesado guardado en {dest}")
    return dest


if __name__ == "__main__":
    raw = load_and_clean()
    feats = build_features(raw)
    save_processed(feats)
