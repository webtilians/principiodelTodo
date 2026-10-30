# INFINITO E10-B — Resultado confirmatorio REAL con LLM y RAG semántico

**30 de octubre de 2026 · Confirmación completada · Hipótesis primaria RECHAZADA dentro del protocolo.**

La ejecución GitHub Actions pasó **técnicamente**, pero la propuesta de superioridad de INFINITO 3.0 frente al RAG semántico **NO pasó científicamente**.

**Referencias verificables**
- [Ejecución GitHub Actions #41109251106 — SUCCESS](https://github.com/webtilians/principiodelTodo/actions/runs/41109251106), commit al iniciar la ejecución \`96af13a6dd218217df4eb8bbdc7b074c2272ee19\`.
- [Protocolo E10-B congelado antes de los experimentos](./INFINITO_E10B_LIVE_SEMANTIC_PROTOCOL.md).
- [JSON agregado inmutable (resultados de seis familias, métricas y criterio)](./INFINITO_E10B_RESULTS_2026-10-30.json).
- [Artifact original CONFIRM: 24 pares, todas las respuestas y contextos](https://github.com/webtilians/principiodelTodo/actions/runs/41109251106/artifacts/13002599280) (retención GitHub Actions limitada).
- [Artifact original DEV](https://github.com/webtilians/principiodelTodo/actions/runs/41109251106/artifacts/13004065821).
- [PR #29 (borrador, sin fusionar)](https://github.com/webtilians/principiodelTodo/pull/29).

## Pregunta científica y método

E10-A había producido 128/128 pares correctos usando una regla de decisión **offline**, tres patrones de hechos estructurados preexistentes y un baseline RAG de vectores **hash**, NO embeddings semánticos. Ese experimento validaba la entrega de evidencia al lector bajo condiciones favorables para INFINITO, **no** su superioridad general.

E10-B preregistró una evaluación distinta, más exigente:
- Mismo LLM **real** \`gpt-5.6-luna\`, \`reasoning_effort=none\`; mismas instrucciones para todos los brazos y \`max_output_tokens=80\`.
- Embeddings semánticos reales **\`text-embedding-3-small\`**, compartidos entre brazos; no vector hash. \`top_k=8\`, contexto máximo **320 tokens estimados** (límite, no obligación de utilizar todo el presupuesto), historial corto invisible en decisión.
- \`no_memory\`; \`semantic_rag\` que guarda **todos** los mensajes de configuración, busca semánticamente y entrega evidencia en orden **cronológico**, incluyendo actualizaciones; \`infinito3\` original inalterado (gate, SQLite, recuperación y GeneralizedContextBuilder); \`infinito3_gate_open\`, ablación que persiste declaraciones sin cambiar el resto del núcleo.
- Seis familias: cuatro **nuevas** para el extractor programado (avisos, formatos, escalado de incidencias, cadencia de revisiones) y dos controles (color favorito, ciudad).
- Cada par tiene dos historias de hechos invertidos, 12 distractores, decisión, corrección al valor opuesto, otros 12 distractores, reinicio de historial y persistencia SQLite, segunda decisión. Acreditar éxito exige **cuatro elecciones correctas** por par, no solo escoger un brazo o repetir una respuesta.
- **DEV** \`1700000+\`: 6 pares, completados antes de abrir confirmación. **CONFIRM** \`1800000+index*1000+j\`, cuatro índices ×seis familias =**24 pares nuevos**, **96 acciones por brazo**, completamente separados de E10-A/F9.
- No se ajustó el núcleo tras observar DEV. Ni las reglas, ni los embeddings, ni la evaluación, ni el modelo se eligieron usando la nueva confirmación. Holdout histórico F9 \`900000+\` **sin abrir**.
- La ingestión de las observaciones utiliza \`CognitiveEngine.process\`, no llamadas al LLM para cada evento. El **LLM real** participa en las decisiones. No es una trayectoria abierta agente-entorno con acciones que alteren futuros datos.

## Desarrollo (6 pares) — primera señal, no resultado confirmatorio

| Brazo | Pares 4/4 |
|---|---:|
| Sin memoria | 0/6 |
| RAG semántico cronológico | **6/6** |
| INFINITO 3.0 original | 1/6 |
| INFINITO con gate abierto | 3/6 |

DEV terminó correctamente; los datos confirmatorios permanecieron sellados hasta superar el gate de procedencia del proveedor real y los tests.

## Confirmación (24 pares NUEVOS) — resultado definitivo del protocolo

| Brazo | Pares con **las cuatro** elecciones correctas | Decisiones individuales | Pares con ambas inversiones contrafactuales |
|---|---:|---:|---:|
| Sin memoria | 0/24 (0%) | 0/96 | 0/24 |
| **RAG semántico cronológico** | **24/24 (100%)** | **96/96** | **24/24** |
| **INFINITO 3.0 original** | **2/24 (8,33%)** | **25/96** | **2/24** |
| **INFINITO 3.0 con gate abierto** | **14/24 (58,33%)** | **72/96** | **14/24** |

Las seis preguntas de control **sin dato histórico** produjeron abstención \`ABSTENER\` en **6/6 casos de cada brazo**. Son seis controles negativos por brazo, insuficientes para estimar una tasa general de alucinación.

### Desglose por familia, en pares 4/4 correctos (cuatro mundos por familia)

| Familia | RAG semántico | INFINITO original | INFINITO gate abierto |
|---|---:|---:|---:|
| Canal de avisos | 4/4 | 0/4 | 3/4 |
| Formato de expediente | 4/4 | 0/4 | 4/4 |
| Escalado de incidencias | 4/4 | 0/4 | 4/4 |
| Frecuencia de revisión | 4/4 | 0/4 | 0/4 |
| Color favorito | 4/4 | 1/4 | 3/4 |
| Ciudad actual | 4/4 | 1/4 | 0/4 |
| **Total** | **24/24** | **2/24** | **14/24** |

**Criterio preregistrado**: ventaja INFINITO original ≥10 puntos porcentuales respecto a RAG semántico, diferencia positiva ≥4/6 familias, intervalo bootstrap por familias con límite inferior >0 y no mayor alucinación negativa. **NO ALCANZADO**.

- Diferencia INFINITO original − RAG semántico: **−91,67 puntos porcentuales** en pares completos.
- Diferencias por familia: **[−100, −100, −100, −100, −75, −75] puntos porcentuales** (en orden avisos/formato/escalado/cadencia/color/ciudad).
- **0/6 familias** positivas para INFINITO.
- Intervalo bootstrap exploratorio por familias **[−100; −83,33] puntos**. Las seis familias de plantillas no constituyen una muestra representativa de dominios reales, por lo que **no extrapolar** ese intervalo a tareas abiertas.
- La ablación de captura mejoró INFINITO de 2/24 a **14/24** (+50 puntos porcentuales), pero siguió 10 pares por debajo del RAG semántico.

### Uso y presupuesto de contexto

| Brazo | Contexto medio en tokens *estimados* | Tokens de entrada reportados por API |
|---|---:|---:|
| Sin memoria | 0 | 16.424 |
| RAG semántico cronológico | **145,36** | **32.875** |
| INFINITO original | **35,92** | **18.520** |
| INFINITO gate abierto | **55,69** | **23.053** |

Los cuatro brazos tenían máximo **320 tokens estimados** de evidencia, pero el contexto **utilizado** no fue equivalente. RAG aportó mucha más información y aprovechó explícitamente el orden temporal; INFINITO filtró más material. Por ello el contraste mide una **diferencia de arquitectura con el mismo techo**, no el efecto aislado de un mecanismo a igualdad de material realmente entregado.

Se registraron **134 llamadas de embedding semántico** en confirmación, con caché común para textos idénticos. Los tokens de entrada API tienen en cuenta todos los mensajes enviados al modelo, no solo el contexto añadido.

## Qué podemos concluir, y qué no

**1. El resultado de E10-A NO generaliza a este banco E10-B.** En la prueba con LLM real y tareas menos alineadas al extractor, el núcleo congelado queda muy por debajo de un RAG semántico que almacena y presenta todos los hechos seleccionados con orden temporal. No existe evidencia de una ventaja general de INFINITO 3.0 frente a este baseline; el ensayo prueba lo contrario **en estos escenarios**.

**2. Hay evidencia diagnóstica de un cuello de botella en la CAPTURA.** Abrir únicamente el gate de almacenamiento eleva los pares 4/4 de 2 a 14. La diferencia es causal en esta ablación de arquitectura, pero **no** cuantifica por sí sola la contribución exclusiva de la captura: cambia qué entra en la base y por tanto qué puede elegir el Context Builder más adelante. No altera la política de lectura/selección.

**3. La captura NO es la única limitación.** Con el gate abierto persisten fallos (10/24); destacan 0/4 en cadencia y ciudad. Posibles mecanismos: extracción del hecho vigente, selección de candidatos, pérdida de contexto temporal, filtros de precisión y decisiones del LLM. **No podemos asignar una causa concreta sin auditar las respuestas/contextos originales del artifact**; no inventar una explicación única.

**4. El baseline semántico tiene ventajas de diseño importantes.** Almacena todo y conserva turnos ordenados; no está equiparado en tokens realmente utilizados ni en política de actualización estructurada. Es un baseline práctico razonable para decisiones que precisan hechos históricos, pero no es prueba de que RAG sea siempre superior a una arquitectura cognitiva ni un control ortogonal que aísle un módulo.

**5. Ninguna conclusión sobre conciencia o autonomía cognitiva general.** El LLM toma decisiones en preguntas cerradas, no decide libremente qué metas perseguir ni qué recordar. E11 y E12 siguen siendo experimentos distintos.

## Decisión conforme a la hoja de ruta E10→E13

**E10-B no supera su gate y queda cerrado**, sin modificar el núcleo ni reexaminar \`1800000+\` como nuevo conjunto confirmatorio. Para evitar ramificaciones infinitas:

1. Ejecutar **una única auditoría diagnóstica sin nuevas hipótesis**, usando el artifact bruto ya obtenido, descomponiendo: «dato almacenado», «dato recuperado», «dato transmitido al LLM», «decisión». Comparar los fallos de \`infinito3\`, \`infinito3_gate_open\` y RAG usando **los mismos pares**. Esta es **exploración diagnóstica**, no confirmación.
2. Solo si emerge un mecanismo concreto y generalizable, hacer **una** intervención acotada, congelada antes de una **nueva partición** genuinamente independiente. Baselines prácticos con presupuestos igualados y modelo real. **No** ajustar la solución sobre los mismos 24 pares.
3. Si una sola intervención no demuestra ventaja o la complejidad no se justifica, **mantener el baseline semántico sencillo** e integrarlo como referencia para E11/E12. El objetivo de INFINITO sigue siendo mejorar decisiones por recuerdos, no ganar un benchmark construido para favorecer su arquitectura.

**No fusionar PR hasta revisión.** E10-A \`1300000+\` y E10-B \`1800000+\` confirmaciones **consumidas**. Holdout histórico F9 \`900000+\` permanece cerrado.
