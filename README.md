# Siniestros Viales en Argentina: Predicción de Gravedad

Proyecto de tesis de fin de carrera. Objetivo: **predecir la gravedad de un
siniestro vial** a partir de las condiciones del hecho (tipo de vía, franja
horaria, vehículos involucrados, factores climáticos) y **identificar zonas y
horarios críticos**, usando datos públicos argentinos.

## Problema

Argentina registra alrededor de 4.000 muertes viales por año (~10 por día,
fuente ANSV). Un modelo capaz de anticipar qué condiciones se asocian a
siniestros graves aporta evidencia para políticas públicas (controles,
infraestructura, campañas) priorizando recursos donde más impactan.

## Preguntas de investigación

1. ¿Qué variables explican mejor la gravedad de un siniestro vial?
2. ¿Es posible clasificar la gravedad (fatal / no fatal) con métricas útiles?
3. ¿Dónde y cuándo se concentran los siniestros graves? (análisis geoespacial)

## Datos (fuentes públicas)

| Fuente | Contenido | Uso |
|---|---|---|
| ANSV - Base LUCHAS | Siniestros fatales a nivel nacional | Extensión futura |
| CABA - Portal de Datos Abiertos | Hechos + víctimas de siniestros (2019-2025) | Dataset principal |
| Open-Meteo (API histórica) | Clima diario CABA: lluvia, temperatura | Feature engineering |

> Las URLs de descarga pueden cambiar. Se centralizan en `src/etl/extract.py`
> (`DATA_SOURCES`, `CLIMA_PARAMS`) para actualizarlas en un solo lugar.

## Estructura del proyecto

```
proyecto-siniestros-viales/
├── data/
│   ├── raw/                  # datos crudos descargados (hechos, victimas, clima)
│   └── processed/            # dataset limpio e integrado (parquet)
├── docs/
│   └── informe_tesis.md      # documentación formal con resultados
├── notebooks/
│   ├── 01_etl_exploracion.ipynb        # pipeline end-to-end (Colab-ready)
│   ├── 02_eda.ipynb                    # EDA + hotspots geoespaciales
│   └── 03_modelado_evaluacion.ipynb    # entrenamiento, umbral, métricas, SHAP
├── src/
│   ├── etl/
│   │   ├── extract.py        # descarga siniestros (CABA) + clima (Open-Meteo)
│   │   └── transform.py      # limpieza, integración de fuentes, features
│   ├── features/
│   │   ├── build_features.py # orquesta el ETL -> parquet final
│   │   └── splits.py         # validación temporal (train/val/test por año)
│   └── models/
│       ├── train_model.py    # dummy + logreg + HGB + MLP
│       ├── evaluate.py       # métricas desbalanceadas + tuning de umbral
│       └── interpret.py      # SHAP del modelo principal
├── reports/figures/          # gráficos generados (PR curves, SHAP, mapa)
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

### Opción 1: Notebooks (Google Colab)

Abrir las notebooks en orden (`01` → `03`). La 02 genera el mapa interactivo
de hotspots (`reports/figures/mapa_hotspots.html`).

### Opción 2: Scripts locales

```bash
pip install -r requirements.txt

python src/etl/extract.py                 # descarga siniestros + clima
python src/features/build_features.py     # limpia + integra + arma dataset
python src/models/train_model.py          # entrena y guarda modelos
python src/models/evaluate.py             # umbral óptimo + métricas finales
python src/models/interpret.py            # SHAP (importancia global)
```

## Metodología

1. **ETL**: integración de 3 fuentes (hechos + víctimas agregadas + clima),
   normalización de categorías, validación geográfica.
2. **Validación temporal**: train ≤2023 · validación 2024 · test 2025.
   Simula el uso real (predecir el futuro) sin fuga temporal.
3. **Modelado**: clasificación binaria desbalanceada (~1% fatales).
   - Piso: DummyClassifier · Baseline: Regresión Logística
   - Principal: HistGradientBoosting · Comparación: MLP
4. **Umbral de decisión**: ajustado en 2024 para maximizar F1 de la clase fatal.
5. **Métricas**: AUC-PR vs prevalencia, precision/recall/F1 de la clase fatal.
6. **Interpretabilidad**: SHAP + importancia por permutación.

## Alcance

Versión mínima viable: una jurisdicción (ej. CABA o una provincia).
Escalable a nivel nacional combinando fuentes.

## Autor / Estado

Tesis de fin de carrera — en desarrollo.
