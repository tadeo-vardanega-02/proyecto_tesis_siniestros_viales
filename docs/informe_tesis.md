# Informe de Tesis: Prediccion de Gravedad de Siniestros Viales en Argentina

> Estado: borrador. Este documento es el esqueleto formal del informe; cada
> seccion se completa con los resultados de las notebooks.

## 1. Introduccion

### 1.1 Contexto y motivacion
Los siniestros viales constituyen un problema grave de salud publica en
Argentina, con aproximadamente 4.000 fallecidos anuales (~10 por dia,
Agencia Nacional de Seguridad Vial). La disponibilidad de datos abiertos
permite aplicar tecnicas de ciencia de datos para identificar patrones de
riesgo y aportar evidencia al diseno de politicas publicas.

### 1.2 Objetivo general
Construir un modelo predictivo que clasifique la gravedad de un siniestro
vial a partir de las condiciones del hecho, e identificar zonas y franjas
horarias criticas.

### 1.3 Objetivos especificos
- Consolidar y limpiar fuentes de datos abiertas (ETL).
- Caracterizar la distribucion temporal y espacial de los siniestros (EDA).
- Entrenar y comparar modelos de clasificacion (baseline vs. ensemble vs. red neuronal).
- Evaluar con metricas apropiadas para clases desbalanceadas.
- Interpretar las variables mas influyentes (SHAP / permutacion).

## 2. Datos

| Fuente | Periodo | Contenido | Acceso |
|---|---|---|---|
| CABA - Portal de Datos Abiertos (hechos) | 2019-2025 | Siniestros con victimas, georreferenciados | abierto |
| CABA - Portal de Datos Abiertos (victimas) | 2019-2025 | Registro por victima: modo, sexo, edad | abierto |
| Open-Meteo (API historica) | 2019-2025 | Clima diario CABA: lluvia, temp. max/min | abierto, sin key |
| ANSV - Base LUCHAS | 2015+ | Siniestros fatales nacionales (extension futura) | abierto |

### 2.1 Integracion de fuentes
- **Victimas -> siniestro**: agregacion por id (n_victimas, edad_promedio,
  pct_moto, pct_peaton, pct_sexo_m). Se excluyen GRAVEdad_victima y
  fecha_fallecimiento_victima por determinar directamente el target
  (fuga de informacion).
- **Clima -> siniestro**: join por fecha; cobertura 100%.

### 2.2 Diccionario de variables (dataset procesado)
- `tipo_calle`, `comuna`: via y comuna del hecho.
- `participantes`, `contraparte`: vehiculos involucrados.
- `hora_num`, `mes`, `dia_semana`, `es_fin_de_semana`, `franja_horaria`: temporales derivados.
- `n_victimas`, `edad_promedio`, `pct_moto`, `pct_peaton`, `pct_sexo_m`: agregados de victimas.
- `lluvia_mm`, `temp_max`: clima del dia.
- `latitud`, `longitud`: coordenadas.
- **`target`**: 1 = fatal (gravedad MORTAL), 0 = resto.

### 2.3 Validacion temporal
- Train: 2019-2023 (43.497) | Validacion: 2024 (10.569) | Test: 2025 (11.752).
- Simula el uso real (predecir el futuro) y evita fuga temporal.

## 3. Metodologia

### 3.1 ETL
- Normalizacion de nombres de columnas y categorias inconsistentes entre fuentes (`src/etl/transform.py`).
- Parseo robusto de fechas/horas; derivacion de variables temporales.
- Validacion geografica (filtrado de coordenadas fuera de Argentina).

### 3.2 Analisis exploratorio (hallazgos principales)
Notebook `02_eda.ipynb`. Resumen:
- **Letalidad ≠ volumen**: la madrugada concentra pocos siniestros pero es
  3,3x mas letal que la tarde; las autopistas son 8,6x mas letales que las
  calles (velocidad de impacto).
- **Colisiones criticas**: peaton-camion (9,1% letal), moto contra objeto
  fijo (8,1%), peaton-transporte publico (8,0%).
- **Perfil de victima**: en siniestros fatales predominan peatones (39% vs
  12%), motos, mayor edad (45 vs 38 anos) y varones (75% vs 51%).
- **Clima sin efecto sobre severidad**: correlaciones < 0,005.
- **Geografia desigual**: comunas del sur (8, 4, 10) con tasas de letalidad
  hasta 2,7x mayores; mapa de densidad en `reports/figures/`.

### 3.3 Modelado
Problema de clasificacion binaria desbalanceada (~1% fatales):
- Piso de referencia: DummyClassifier (estratificado) — obliga a comparar contra el azar.
- Baseline: Regresion Logistica (ponderada por clase).
- Principal: HistGradientBoostingClassifier.
- Comparacion: MLP (red neuronal multicapa).

Preprocesamiento: imputacion, escalado numerico, one-hot encoding con
agrupacion de categorias raras.

### 3.4 Ajuste de umbral y protocolo de evaluacion
Con validacion temporal (train <=2023, val 2024, test 2025):
- **Validacion (2024)**: se busca el umbral de decision que maximiza el F1
  de la clase fatal (el default 0.5 no es apropiado con clases tan
  desbalanceadas).
- **Test (2025)**: evaluacion final con el umbral optimo de cada modelo.
- El modelo nunca ve datos posteriores a su entrenamiento.

### 3.5 Metricas
- F1-macro como referencia, pero **no** como criterio unico: con ~1% de
  positivos, un clasificador aleatorio obtiene ~0.50 por estructura del dato.
- Metricas decisivas: precision/recall/F1 de la clase fatal, y AUC-PR
  comparada contra la prevalencia (piso teorico = 0.0105).
- Matrices de confusion y curvas Precision-Recall.

## 4. Resultados

> Corrida end-to-end con datos de CABA (2019-2025, 65.818 siniestros,
> enriquecidos con victimas y clima). Validacion temporal: train <=2023,
> umbral ajustado en 2024, evaluacion final en 2025 (n=11.752, 90 fatales).

| Modelo | Umbral | F1-macro | AUC-PR | Recall fatal | Precision fatal |
|---|---|---|---|---|---|
| Dummy (azar) | 0.50 | 0.496 | 0.008 (≈prevalencia) | 0.000 | - |
| Regresion Logistica | 0.970 | 0.539 | 0.055 | 0.089 | 0.081 |
| **HistGradientBoosting** | **0.896** | **0.573** | **0.074** | **0.189** | **0.130** |
| MLP | 0.50 | 0.523 | 0.057 | 0.033 | 0.100 |

### 4.1 Lectura de los resultados
- Con validacion temporal (mas exigente que el split aleatorio), HGB es el
  mejor modelo: detecta ~19% de los fatales de 2025 con precision ~13%
  (~13x la prevalencia), entrenando solo con datos hasta 2023.
- AUC-ROC de HGB = 0.892 vs 0.495 del azar: el ranking de riesgo tiene senal
  fuerte; el desafio esta en el corte operativo (umbral).
- La regresion logistica degrada mas en el tiempo: las relaciones lineales
  fijas de 2019-2023 generalizan peor que los arboles.
- Comparado con el split aleatorio anterior, todas las metricas bajan: es el
  costo honesto de predecir un año futuro no visto.

### 4.2 Interpretabilidad (SHAP sobre HGB, test 2025)
Variables de mayor impacto medio |SHAP|:
1. `pct_moto` — proporcion de motos involucradas: principal predictor
2. `edad_promedio` — edad media de las victimas
3. `pct_sexo_m` — proporcion de victimas varones
4. `contraparte_AUTO` / `contraparte_CAMION` — vehiculo oponente pesa
5. `pct_peaton` — peatones implicados elevan el riesgo fatal
6. `participantes_SD-SD` — datos faltantes informativos (ver 4.3)
7. `hora_num`, `temp_max`, `lluvia_mm` — temporales y clima aportan menos

### 4.3 Hallazgos metodologicos
- **Datos faltantes informativos (MNAR)**: "SD" sigue siendo influyente pero
  perdio peso relativo frente a las variables de victima reales.
- **El clima aporta poco**: temperatura maxima aparece entre las top-10 pero
  con impacto marginal; la lluvia casi no influye sobre la gravedad
  (si acaso sobre la frecuencia, no la severidad).
- **El perfil de la victima domina**: composicion por modo de transporte
  (moto/peaton) y demografia explican mas que infraestructura o tiempo.

### 4.1 Interpretabilidad
> Variables mas influyentes segun SHAP/permutacion.

## 5. Discusion y limitaciones
- **Desbalanceo extremo (~1%)**: accuracy es inutil; se trabaja con metricas
  por clase y AUC-PR contra prevalencia. La precision del modelo (13%) limita
  su uso como "prediccion individual", pero sirve como ranking de riesgo.
- **Validacion temporal mas exigente**: las metricas bajan respecto a un
  split aleatorio porque 2025 no comparte patrones exactos con el pasado;
  es la estimacion honesta del rendimiento en produccion.
- **Datos faltantes informativos (MNAR)**: registros "SD" se concentran en
  hechos graves; parte del poder predictivo puede ser artefacto de registro.
- **Sesgo de reporte**: CABA registra con criterios propios; no extrapolable
  directamente al resto del pais sin armonizacion (base LUCHAS).
- **Clima**: sin senal sobre severidad; podria si aportar sobre frecuencia.

## 6. Conclusiones y recomendaciones

### 6.1 Conclusiones tecnicas
1. Es posible anticipar la gravedad de un siniestro con senal estadistica
   clara: HGB logra AUC-ROC 0,89 y detecta ~19% de los fatales del año
   siguiente con precision 13x superior al azar.
2. El perfil de la victima (modo de transporte, edad) domina sobre
   infraestructura y clima: la vulnerabilidad del cuerpo, no el entorno,
   explica principalmente el desenlace fatal.
3. El ajuste de umbral y la validacion temporal son pasos obligatorios en
   problemas desbalanceados: sin ellos las metricas enganan.

### 6.2 Recomendaciones de politica publica (derivadas de los datos)
- **Priorizar proteccion de peatones y motociclistas**: separacion fisica de
  trafico pesado (peaton-camion 9% letal), infraestructura motociclista.
- **Intervenir autopistas y madrugada**: control de velocidad nocturno,
  donde cada siniestro es 3x mas grave.
- **Focalizar en zona sur**: comunas 8, 4 y 10 concentran letalidad
  desproporcionada - asignacion geografica de recursos.
- **Mejorar captura de datos**: la calidad de registro cae justamente en los
  casos graves (MNAR); capacitar relevamiento in situ.

## Referencias
- ANSV - Observatorio Nacional de Seguridad Vial.
- Portal de Datos Abiertos CABA.
- Documentacion: scikit-learn, SHAP.
