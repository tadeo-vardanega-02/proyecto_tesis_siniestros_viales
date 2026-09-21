"""Descarga de datos crudos de siniestros viales y clima (fuentes publicas).

Uso:
    python src/etl/extract.py

Fuentes:
    - CABA (hechos y victimas): portal de datos abiertos.
    - Clima diario CABA: API historica de Open-Meteo (gratuita, sin key).
      Documentacion: https://open-meteo.com/en/docs/historical-weather-api

Las URLs se centralizan en DATA_SOURCES: si una fuente cambia su endpoint,
actualizar solo ese diccionario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# URLs de datasets abiertos. Verificar vigencia en el portal de cada fuente.
DATA_SOURCES: dict[str, str] = {
    # CABA: victimas de siniestros viales (georreferenciado)
    "caba_victimas": "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte-y-obras-publicas/victimas-siniestros-viales/siniestros_viales_victimas.csv",
    # CABA: hechos de siniestros viales
    "caba_hechos": "https://cdn.buenosaires.gob.ar/datosabiertos/datasets/transporte-y-obras-publicas/victimas-siniestros-viales/siniestros_viales_hechos.csv",
}

# Clima: punto representativo de CABA + rango que cubre los anios del dataset.
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
CLIMA_PARAMS: dict = {
    "latitude": -34.61,
    "longitude": -58.44,
    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
    "timezone": "America/Argentina/Buenos_Aires",
}
CLIMA_CHUNKS: list[tuple[str, str]] = [
    ("2019-01-01", "2020-12-31"),
    ("2021-01-01", "2022-12-31"),
    ("2023-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
]

TIMEOUT_SECONDS = 120


def download(name: str, url: str) -> Path | None:
    """Descarga un dataset a data/raw/. Devuelve la ruta o None si falla."""
    dest = RAW_DIR / f"{name}.csv"
    if dest.exists():
        print(f"[skip] {name} ya existe en {dest}")
        return dest
    print(f"[get ] descargando {name} ...")
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        size_mb = dest.stat().st_size / 1_000_000
        print(f"[ok  ] {name}: {size_mb:.1f} MB -> {dest}")
        return dest
    except requests.RequestException as exc:
        print(f"[fail] {name}: {exc}", file=sys.stderr)
        return None


def load_raw(path: Path) -> pd.DataFrame:
    """Carga un CSV crudo con tolerancia a encoding/sep variables."""
    for sep in [",", ";"]:
        for encoding in ["utf-8", "latin-1"]:
            try:
                return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
            except UnicodeDecodeError:
                continue
            except pd.errors.ParserError:
                continue
    raise ValueError(f"No se pudo leer {path}")


def download_climate() -> Path | None:
    """Descarga el clima diario de CABA (Open-Meteo) en chunks por rango."""
    dest = RAW_DIR / "clima_caba.csv"
    if dest.exists():
        print(f"[skip] clima_caba ya existe en {dest}")
        return dest
    print("[get ] descargando clima CABA (Open-Meteo) ...")
    frames: list[pd.DataFrame] = []
    try:
        for start, end in CLIMA_CHUNKS:
            params = dict(CLIMA_PARAMS, start_date=start, end_date=end)
            resp = requests.get(OPEN_METEO_URL, params=params,
                                timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            daily = resp.json()["daily"]
            frames.append(pd.DataFrame(daily))
            print(f"[ok  ] chunk {start}..{end}: {len(daily['time'])} dias")
        df = pd.concat(frames, ignore_index=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        print(f"[ok  ] clima guardado: {len(df):,} filas -> {dest}")
        return dest
    except requests.RequestException as exc:
        print(f"[fail] clima: {exc}", file=sys.stderr)
        return None


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {n: download(n, u) for n, u in DATA_SOURCES.items()}
    downloaded["clima_caba"] = download_climate()
    ok = [(n, p) for n, p in downloaded.items() if p]
    if not ok:
        print("No se descargo ningun dataset. Revisar URLs en DATA_SOURCES.")
        sys.exit(1)
    for name, path in ok:
        df = load_raw(path)
        print(f"{name}: {len(df):,} filas x {len(df.columns)} columnas")
        print(f"  columnas: {list(df.columns)[:15]}")


if __name__ == "__main__":
    main()
