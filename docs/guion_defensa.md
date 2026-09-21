# Guion de Defensa de Tesis

> Proyecto: Prediccion de gravedad de siniestros viales en CABA con datos publicos.
> Todos los numeros citados provienen de corridas reales del proyecto.

---

## 1. Pitch de apertura (~2 min)

"Argentina registra alrededor de 4.000 muertes viales por año. Este proyecto
usa datos abiertos de la Ciudad de Buenos Aires -65.818 siniestros entre 2019
y 2025- para responder dos preguntas: que condiciones hacen mas letal a un
siniestro, y es posible anticiparlo con machine learning.

El pipeline integra tres fuentes publicas: hechos de siniestros, registro por
victima y clima diario. Con ese dataset se entrenaron cuatro clasificadores,
con validacion temporal: los modelos aprenden de 2019-2023 y son evaluados
sobre 2025, un año que nunca vieron.

Resultado principal: el modelo detecta ~19% de los siniestros fatales del año
siguiente con precision 13 veces superior al azar, y el analisis interpretable
muestra que lo que mas predice la gravedad no es la via ni el clima, sino el
perfil de la victima: peatones y motociclistas."

## 2. Estructura sugerida de exposicion (15-20 min)

| Min | Seccion | Mensaje clave |
|---|---|---|
| 0-2 | Problema y motivacion | Muertes evitables + datos abiertos = oportunidad |
| 2-5 | Datos y ETL | Integracion de 3 fuentes, decisiones de limpieza |
| 5-8 | Hallazgos del EDA | Letalidad != volumen; perfil de victima domina |
| 8-13 | Modelado y evaluacion | Protocolo riguroso: dummy, umbral, validacion temporal |
| 13-16 | Interpretabilidad | SHAP: que aprendio el modelo y por que tiene sentido |
| 16-20 | Conclusiones y politica publica | Recomendaciones accionables |

---

## 3. Preguntas probables del jurado (con respuestas)

### Metodologicas

**P: Por que no reportan accuracy?**
R: Con 99% de clase negativa, predecir siempre "no fatal" da 98.9% de accuracy.
Es una metrica inutil aqui. Trabajamos con precision/recall/F1 de la clase
fatal y AUC-PR comparada contra la prevalencia (0.9%), que es el piso teorico.
Incluimos ademas un DummyClassifier como referencia explicita.

**P: Por que validacion temporal y no split aleatorio?**
R: Dos razones. Primera, honestidad: el modelo se usaria para predecir el
futuro, no el pasado remezclado. Segunda, evidencia: al comparar, las metricas
con split aleatorio eran notablemente mejores (AUC-PR 0.14 vs 0.074); esa caida
es el costo real de generalizar a un año no visto. Reportar solo el numero
optimista seria enganoso.

**P: Como eligieron el umbral de decision? No es trampa ajustarlo?**
R: El default 0.5 es arbitrario y malo con clases desbalanceadas. Ajustamos el
umbral maximizando F1 de la clase fatal en un set de VALIDACION (2024)
completamente separado del test (2025). Es el protocolo estandar: tuning en
validacion, evaluacion unica en test. Ademas el umbral es un parametro de
negocio: un sistema de alerta priorizaria recall (no dejar pasar fatales).

**P: Hay fuga de informacion (leakage)?**
R: Lo revisamos expresamente. La tabla de victimas contiene GRAVEdad_victima y
fecha_fallecimiento_victima, que DETERMINAN el target: las excluimos
deliberadamente. Solo usamos agregados ex-ante: cantidad de victimas, edad
promedio, proporciones de moto/peaton/sexo. El clima se une por fecha, conocida
el mismo dia.

**P: Que modelo usaron y por que gano?**
R: HistGradientBoosting supero a regresion logistica y a una red neuronal MLP.
Con datos tabulares, tamano moderado (~43 mil filas de entrenamiento) y senal
concentrada en pocas variables, los metodos basados en arboles suelen ganar:
capturan interacciones no lineales sin requerir ingenieria de features
manual. Es consistente con la literatura (por eso XGBoost/LightGBM dominan
competencias de tabular).

### Sobre los hallazgos

**P: Ustedes dicen que el clima no importa. Estan seguros?**
R: Sobre la SEVERIDAD, si: correlaciones < 0.005 y SHAP marginal. Es un
hallazgo negativo documentado. Ojo: no afirmamos que el clima no afecte la
CANTIDAD de siniestros - eso seria otro modelo (de frecuencia). Severidad y
frecuencia son problemas distintos.

**P: La variable "sin dato" aparece entre las mas influyentes. No es un error?**
R: Es uno de nuestros hallazgos metodologicos centrales. Los registros con
datos faltantes se concentran en hechos graves: es missing not at random.
Dos lecturas: (a) parte del poder predictivo puede ser artefacto de registro,
lo reconocemos como limitacion; (b) en si mismo es relevante para gestion
publica: la calidad de captura cae justamente en los casos mas importantes.

**P: Como pasan de correlacion a recomendacion de politica publica?**
R: Con cautela y transparencia. Las recomendaciones salen de asociaciones
fuertes, consistentes y con mecanismo plausible (velocidad de impacto explica
autopistas letales; vulnerabilidad corporal explica peaton/moto). No
afirmamos causalidad; recomendamos pilotos focalizados y medicion posterior.

### Tecnicas / de implementacion

**P: Como manejan el desbalanceo?**
R: Tres capas: class_weight='balanced' en los modelos, eleccion de metricas
insensibles al desbalanceo, y umbral ajustado en validacion. Evaluamos SMOTE
como trabajo futuro; con boosting ponderado suele aportar poco y puede distorsionar
las probabilidades calibradas.

**P: Es reproducible?**
R: Todo: fuentes con URLs versionadas en codigo, seeds fijos, scripts ordenados
(extract -> build_features -> train -> evaluate -> interpret), notebooks Colab
ejecutables end-to-end, y requirements.txt. Cualquiera re-ejecuta el pipeline
desde cero.

**P: Por que no deep learning si la carrera es de IA?**
R: Usamos una red neuronal (MLP) como comparacion: perdio contra HGB. La
eleccion del modelo es empirica, no ideologica. DL brilla en imagenes, texto y
secuencias; en tabular pequeno, boosting es estado del arte. Decimos esto con
evidencia dentro del propio proyecto.

**P: Limitaciones principales?**
R: (1) Solo CABA: no extrapolamos al pais sin armonizar con LUCHAS. (2)
Precision baja para uso individual: es un ranking de riesgo, no un oraculo.
(3) Posible sesgo de reporte entre jurisdicciones/años. (4) MNAR discutido
antes. (5) Variables que no tenemos: densidad de trafico, infraestructura vial,
alcoholemia.

---

## 4. Frases para cerrar

- "El valor de este proyecto no esta en el modelo, esta en el protocolo:
  cualquier jurisdiccion con datos abiertos puede replicarlo."
- "En seguridad vial, saber DONDE mirar ya es una intervencion: el modelo
  convierte 65 mil registros en prioridades accionables."
- "Documentamos tambien lo que NO funciono - el clima - porque un hallazgo
  negativo honesto vale mas que un resultado inflado."

## 5. Checklist antes de defender

- [ ] Notebooks ejecutadas top-to-bottom en Colab, outputs visibles
- [ ] Repo en GitHub publico + enlace en el informe
- [ ] Ensayar el pitch cronometrado (max 20 min)
- [ ] Tener a mano: mapa_hotspots.html, shap_bar.png, pr_curves.png
- [ ] Repasar significado exacto de cada metrica de la tabla de resultados
