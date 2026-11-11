# INFINITO F10-O: observaciones terminales y decisión de integración

**3 de noviembre de 2026. Resultado: F10O_TERMINAL_TRANSFER_FAIL. Ejecución y auditoría completas. Recomendación: NO-GO para integrar el controlador aprendido en INFINITO 3; GO condicionado únicamente para diseñar un piloto de observación aislado, desactivado por defecto.**

## 1. Respuesta científica

Con el algoritmo, información y presupuesto fijados prospectivamente, aprender una representación recurrente de las operaciones a partir de su respuesta terminal y recompensa **no produjo transferencia composicional robusta**. T obtiene 1130/2048 aciertos (55,18% descriptivo); los controles de recompensa y de respuesta permutada, 970/2048 (47,36%); el control privilegiado de efectos instrumentados, 1981/2048 (96,73%). Ninguna réplica cumple G1. No se ha demostrado imposibilidad de aprender efectos con observaciones parciales: se ha refutado la suficiencia de esta configuración.

Una réplica sí muestra una señal prometedora, pero no pasa: r3 alcanza 406/512 (79,30%), propone SRX en 296 casos (57,81%) y RX en 216. Además falla competencia DEV PUBLIC+FLIP1 (71,875%). No se rescata esa réplica ni se relajan umbrales. Las otras tres no consultan memoria en sus planes OOD: SPX, SPX y XPX/XXX.

**Límite de atribución:** T e I difieren tanto en información como en algoritmo/representación. Su diferencia NO identifica por sí sola el efecto causal puro de retirar instrumentación. T frente a R y S controla arquitectura, inicialización, interacciones y actualizaciones, y estima la utilidad de la supervisión terminal auxiliar en esta arquitectura. No es una reproducción del planificador tabular F10-N con una única variable cambiada.

## 2. Cronología y preservación

- Producto revisado: master `f6bd6239bfc1734c04c233a5cf8cef050c6d54c2`, paquete `infinito-memory` 0.6.0; README conserva el rótulo alpha v0.5. Se informa esta discrepancia documental; no se infiere versión desplegada.
- Dependencia de laboratorio congelada: `f1ba7db6364cb92f0443d63a4d13eb527049f60d`. F10-N y F10-N-C quedan sin modificación. F10-N-C conserva su PASS 3/4 y su fallo individual r2.
- [Prerregistro 752e99a](https://github.com/webtilians/principiodelTodo/commit/752e99af6974c7cf853aa4a31b6bd8d8b0bdc2c6): publicado a las 22:22:47 CET antes de implementación y generación de datos.
- [Implementación congelada 0c49df4](https://github.com/webtilians/principiodelTodo/commit/0c49df4e66d3eba0dd0e14947cdc13e571f80c72): publicada a las 22:24:25 CET antes de la única ejecución local CPU. No se lanzó entrenamiento en GitHub Actions ni se atribuye a un run inexistente.
- 20 checkpoints originales F10-L del [run 41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939): autenticados contra manifiestos y G0, sin reentrenarlos. Las nuevas memorias I/B y redes T/R/S son artefactos F10-O, no reemplazos de las memorias F10-N.
- Comparación de referencias remotas antes/después: **123 ramas previas, cero cambios**; solo se añade la rama F10-O, nacida de master. El código `src/infinito3` no tiene diferencias frente a master revisado. No hay merge ni despliegue.

## 3. Qué se ejecutó

Cuatro réplicas, 3072 casos TRAIN por réplica, 16 pases y dos programas por caso: **98.304 episodios por brazo/réplica** sobre un flujo compartido. Es decir, 393.216 ejecuciones únicas, observadas por cinco brazos; no cinco veces más interacciones para T. TRAIN incluye PUBLIC0, PUBLIC1 y MEMORY0; excluye MEMORY1 y el programa SRX. Catálogo de 85 programas, 84 permitidos en entrenamiento. Evaluación: 768 DEV y 512 OOD por réplica; OOD combina MEMORY1, 9/11 informes y cuatro patrones de procedencia. Cuatro variantes correlacionadas por mundo.

Semillas nuevas: TRAIN 27100000+, DEV 27200000+, OOD 27400000+, con desplazamientos por réplica/contexto documentados. 27300000+ permanece reservado. Ninguna partición previa de confirmación se usa como datos nuevos. Es nueva muestra de la misma familia sintética, no entorno externo.

| Brazo | Información y aprendizaje |
|---|---|
| T | GRU con contexto público y secuencia de acciones; predice éxito y respuesta terminal realmente emitida. |
| R | Misma red, solo pérdida de éxito; cabeza auxiliar con peso cero. |
| S | Misma red; respuestas terminales permutadas dentro de cada minibatch. |
| B | Extrapolación aditiva de puntuaciones original F10-N. |
| I | Memoria original F10-N de transiciones before/after, control positivo privilegiado. |

T/R/S reciben exclusivamente `context`, `actions`, `answer`, `success`. No reciben `truth`, `winner`, contenido F4, índice seleccionado, `trace`, `before`, `after`, errores internos ni estados de trabajo. El ejecutor sigue instrumentado para el control I y la auditoría; esa instrumentación no cruza la frontera de aprendizaje de T/R/S. La recompensa binaria y SOURCE/FLIP sí son proporcionados por el entorno diseñado: no son feedback natural ruidoso ni observaciones completamente libres de diseño humano.

GRU: 4.266 parámetros por brazo, embedding 8, estado 32, Adam .003, 512 actualizaciones de batch 192. Inicializaciones idénticas entre T/R/S de cada réplica; sin selección de checkpoint. Los tres evalúan 85 candidatos antes de actuar, maximizando probabilidad de éxito predicha menos .01 por operación. El estado latente se aprende, no se suministra una regla manual de selección/XOR. La cabeza de respuesta es auxiliar: esto no identifica reglas simbólicas de cada operador ni demuestra que la representación interna sea causalmente correcta.

Los modelos y tablas se guardan y sellan con SHA antes de generar DEV/OOD de cada réplica. Los resultados de una réplica no afectan al entrenamiento de otra. No hubo tuning, early stopping, entrenamiento OOD ni nueva corrida tras los resultados.

## 4. Resultados congelados

| Réplica | T terminal | IC95 por mundo | R recompensa | S permutado | B aditivo | I instrumentado | SRX en T |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 51.95% | [47.66; 56.25] | 51.95% | 51.95% | 51.95% | 100.00% | 0/512 |
| 1 | 43.75% | [39.45; 48.05] | 43.75% | 43.75% | 43.75% | 100.00% | 0/512 |
| 2 | 45.70% | [41.41; 49.61] | 45.70% | 45.70% | 45.70% | 86.91% | 0/512 |
| 3 | 79.30% | [74.02; 84.18] | 48.05% | 48.05% | 48.05% | 100.00% | 296/512 |

Intervalos percentiles de 2000 remuestreos de 128 mundos, manteniendo juntas las cuatro variantes. La suma de 2048 casos es descriptiva: no constituye 2048 observaciones independientes ni cuatro validaciones externas.

| Gate preregistrado | Réplicas que pasan | Decisión |
|---|---:|---|
| G0: competencia DEV >=80% en los tres contextos | 3/4 | PASS global |
| G1: precisión OOD >=80%, IC inferior >70%, estratos >=70%, SRX >=90% | 0/4 | FAIL |
| G2: caídas causales F4 y X >=20pp, IC inferior >0 | 1/4 | FAIL global |
| G3: ventaja sobre R, S y B >=10pp, IC inferior >0 | 1/4 | FAIL global |
| G4: control I >=80%, diagnóstico | 4/4 | Control positivo operativo |

Se exigían >=3/4 en cada G0–G3. `science.json` conserva todos los estratos, programas, intervalos y gates, incluidos negativos.

## 5. Causalidad e integridad

Las acciones TRAIN se asignaron uniformemente por RNG, dentro de cada caso compartido. Esto evita atribuir la ventaja de T a exploración adicional. La permutación de respuestas en S conserva la carga auxiliar de entrenamiento, pero destruye su asociación con el episodio. R mantiene capacidad asignada y número de pasos, aunque su cabeza de respuesta no recibe gradiente. La magnitud de pérdida y los FLOPs efectivos no son idénticos; no se afirma igualdad exacta de cómputo con las tablas B/I.

Intervenciones pareadas sobre cada plan congelado: reordenar contenidos F4, desactivar X y cambiar el bit público. r0–r2 tienen caída F4 exactamente cero, coherente con planes que no leen memoria. En r3: caída F4 **29,69pp**, IC95 [22,46;37,11]; caída X **57,81pp**, [49,22;66,41]; cambio público 0. La ventaja T−R/S/B es **31,25pp**, [25,59;36,72], solo en r3; en las otras tres es exactamente cero. Permutar los identificadores de acciones del predictor deja r3 en 0%, diagnóstico de dependencia del catálogo aprendido, no de robustez semántica.

La auditoría posterior:

1. Reconstruye exactamente **393.216 episodios TRAIN**, sus respuestas terminales y recompensas, y ambas memorias B/I desde el generador y ejecutor congelados.
2. Recalcula los planes desde los checkpoints T/R/S congelados y las tablas; reproduce **30.720 filas DEV/OOD** y **122.880 resultados con intervenciones** (incluye control diagnóstico de IDs).
3. Verifica tamaños, exclusión SRX, ausencia MEMORY1/TRAIN, particiones disjuntas, claves permitidas y SHA de modelos/logs/filas. Comprueba los 20 SHA originales antes y después.
4. Recalcula estadísticas desde las filas, sin entrenar ni modificar modelos.

Es un auditor en proceso separado, escrito por el mismo agente, que comparte el ejecutor original: **no es revisión externa ni implementación independiente de la física del mundo**. El prerregistro y código tienen prueba temporal remota; los sellos individuales de modelos se emitieron localmente, no en un servicio de timestamp externo. Durante el desarrollo del replay se corrigió una comparación tuple/list tras serializar JSON; se conservó la corrida, sin cambiar entrenamiento, modelos, datos ni gates. Los tests previos verificaron padding y flujo de gradientes. Los tests existentes del producto dieron **137 passed, 2 skipped**; no equivalen a certificación de producción.

## 6. Por qué todavía no trasladarlo al producto

La observación terminal es insuficiente para distinguir muchas trayectorias internas: distintos programas pueden producir el mismo bit final. Además, MEMORY1 no está entrenado y el modelo puede aprender atajos ligados al bit público. Una red recurrente y una pérdida auxiliar no fuerzan la factorización selección→lectura→transformación. El control I demuestra que las primitivas congeladas y los nuevos mundos siguen siendo utilizables, pero no separa información privilegiada de sesgo inductivo del planificador.

Tampoco se han probado texto libre, percepción, errores de herramientas, feedback retrasado, resultados censurados, concurrencia, objetivos naturales, nuevas operaciones, adaptación online ni planificación reactiva. El éxito binario exacto del simulador suele faltar en producto. Aumentar capacidad o presupuesto podría ayudar, pero sería otro experimento con otro prerregistro, nunca una reparación de este resultado.

## 7. Arquitectura real de INFINITO 3

Revisión de código del commit master indicado, no inspección de un servicio desplegado:

- `CognitiveEngine.process`: seguridad → olvido → recuperación anterior a la escritura → gate/objetivos → contexto → persistencia. `recall` ofrece construcción de contexto sin almacenar el mensaje ni ejecutar olvido.
- `SQLiteCognitiveMemoryStore`: memoria persistente, hechos y versiones, búsqueda léxica/vectorial, refuerzo, sustitución, olvido y purga. `search` registra accesos por defecto: un shadow ingenuo puede alterar contadores; usar copia de base o `record_access=False`.
- `GeneralizedContextBuilder`/`BalancedContextBuilder`: selección y orden temporal bajo presupuesto; no poseen un modelo aprendido de efectos de herramientas.
- `CognitiveLoop` y `LLMAdapter.generate(LLMRequest)`: separación de la lógica cognitiva del proveedor, historial corto y contexto tratado como datos no fiables. `compare` compara con/sin cognición; no sustituye un benchmark de RAG semántico de igual presupuesto.
- MCP `AgentMemory`: `remember`, `recall`, `list_memories`, `forget`; bóveda opcional separada. `recall` muestra todo hasta 40 memorias y luego selecciona 16; `list_memories` limita a las 200 recientes. No exponer automáticamente estos contenidos como entrenamiento. La bóveda debe quedar fuera de cualquier observador aprendido.
- Núcleo solo biblioteca estándar; PyTorch/NumPy del laboratorio no deben convertirse en dependencias obligatorias del paquete. No hay contrato actual para planificar operaciones aprendidas ni aislamiento multiusuario en los argumentos de estas APIs; un piloto necesita aislamiento explícito por base/proceso/usuario.

Fuentes de arquitectura: [engine.py](https://github.com/webtilians/principiodelTodo/blob/f6bd6239bfc1734c04c233a5cf8cef050c6d54c2/src/infinito3/engine.py), [interfaces.py](https://github.com/webtilians/principiodelTodo/blob/f6bd6239bfc1734c04c233a5cf8cef050c6d54c2/src/infinito3/interfaces.py), [persistent_memory.py](https://github.com/webtilians/principiodelTodo/blob/f6bd6239bfc1734c04c233a5cf8cef050c6d54c2/src/infinito3/persistent_memory.py), [mcp_server.py](https://github.com/webtilians/principiodelTodo/blob/f6bd6239bfc1734c04c233a5cf8cef050c6d54c2/src/infinito3/mcp_server.py).

Los hallazgos publicados de producto favorecen una prioridad más directa: INFINITO v0.5 obtiene 112/120 escenarios frente a 117/120 de RAG y reduce 31% los tokens de contexto; 10 de sus 13 decisiones erróneas correspondían a actualizaciones no recuperadas. Son resultados anteriores documentados, no reejecutados aquí. Fuente: [FINDINGS.md del master revisado](https://github.com/webtilians/principiodelTodo/blob/f6bd6239bfc1734c04c233a5cf8cef050c6d54c2/docs/FINDINGS.md). El beneficio marginal más defendible hoy está en recuperación de actualizaciones, selección de contexto y olvido efectivo.

## 8. Integración aislada propuesta — no implementada

**NO-GO:** sustituir el planificador, automatizar escrituras/olvidos, aprender online de usuarios, trasladar el 96,73% sintético a marketing o producto, añadir Torch al núcleo.

**GO condicionado:** diseñar un componente externo de observación y evaluación offline, con flag propuesto `INFINITO_EFFECT_OBSERVER=off|shadow` (no existe actualmente). `off` por defecto, sin carga de modelo ni latencia añadida; `shadow` solo sobre datos sintéticos o copias saneadas autorizadas. No decide ni ejecuta herramientas.

Contrato propuesto, versionado, separado de las APIs existentes:

- `observe(event)`: `schema_version`, ID de episodio seudónimo, secuencia, ID/version de operación, instante, permisos, campos de respuesta pública permitidos, máscara de observación y feedback opcional con procedencia. Ausente se representa como ausente, nunca como éxito cero. No registrar trazas internas, consultas completas, contenidos personales, bóveda ni secretos. Aplicar filtrado antes del log, no después.
- `predict_effects(history, candidate_actions, budget)`: predicciones con incertidumbre, soporte observado, versión del modelo y posibilidad de abstenerse. Sin acceso a SQLite de producción; catálogo permitido y límite de longitud fuera del modelo.
- `purge_lineage(subject_or_episode)`: elimina eventos y derivados; si un modelo ya aprendió de esos datos, invalidar/reentrenar desde datos permitidos. Borrar una fila no borra información de pesos, copias ni logs.

La seguridad actual y las autorizaciones permanecen en el flujo obligatorio. Fallos/timeouts del sidecar devuelven abstención y conservan comportamiento base. No permitir `forget`, `remember` ni operaciones de bóveda en un futuro piloto de planificación sin una revisión específica. Mantener kill switch, cuotas, procedencia y aislamiento; ningún recuerdo recuperado se convierte en instrucción.

## 9. Coste y beneficio

Medido en esta máquina, CPU y un hilo: **47,71 s** sumando las cuatro réplicas (generación, formación de memorias, entrenamiento y evaluación; excluye instalación y auditoría). Cada red se entrenó en aproximadamente 1,52–1,63 s; 4.266 parámetros, checkpoint de unos 20 KB. Las decisiones T sobre ocho contextos distintos tardaron en conjunto 6,2–11,7 ms por réplica (media aproximada 0,8–1,5 ms/contexto); se cachearon, por lo que esto NO es p95 de servicio ni un benchmark de tareas reales. Cero llamadas a LLM o APIs de pago para el experimento; no incluye electricidad, trabajo de implementación ni almacenamiento. Artefactos originales del run previo no se cobran como entrenamiento nuevo aquí.

La búsqueda crece como suma de 4^k: 85 candidatos hasta longitud 3, 1.365 hasta 5, 87.381 hasta 8. Los costes no escalan linealmente con longitud sin poda o planificación distinta. El cuello de botella de producto probablemente sea obtener feedback fiable y evaluar seguridad/beneficio, no estos 20 KB. No hay medida que justifique ahorro monetario de integración.

Coste futuro debe calcularse por separado: `N * (tokens_entrada * tarifa_entrada + tokens_salida * tarifa_salida) / 1e6`, más embeddings, CPU y almacenamiento. No se fijan precios actuales no consultados ni se imputa una tarifa inexistente. Estimación de planificación, no medición: adaptador shadow y borrado de derivados, 3–5 jornadas de ingeniería; evaluación y revisión de privacidad, otras 5–10; no incluye construir un algoritmo nuevo. Priorizar un único ensayo informativo de identificación/observación parcial frente a rondas de tuning sobre este OOD.

## 10. Criterios de una futura decisión GO

Estos criterios son propuestas para otro protocolo, no cambios a F10-O:

1. Confirmación nueva, artefactos congelados y sin ajuste OOD: al menos 3/4 réplicas pasan competencia, transferencia, ventaja sobre controles y dependencia causal. Para separar algoritmos de observabilidad, usar el mismo learner con una escalera de observaciones y controles de información equivalentes.
2. Tareas de producto nuevas y separadas por usuario/episodio: contraste contra INFINITO actual y RAG semántico, idéntico límite de contexto, llamadas y tokens. Ejemplo de gate a preregistrar: mejora >=5pp y límite inferior IC95 >0 frente a ambos; ningún estrato prioritario peor en >2pp. Potencia/tamaño muestral definidos antes; no elegir el conjunto que favorezca el método.
3. Cero filtraciones en suite adversarial predefinida; aislamiento, inyección, permisos y borrado de todos los derivados pasan; ausencia de fallos observados no equivale a riesgo cero. Toda condición incierta se abstiene, sin escritura.
4. Presupuesto propuesto: p95 añadido <=25 ms para observador local, <=10% coste total adicional por tarea, prueba con cardinalidad/concurrencia realistas y rollback verificado. Debe compensar con utilidad medida.
5. Paso shadow → sugerencias de solo lectura → cualquier acción, cada uno evaluado por separado. Ningún PASS científico autoriza automáticamente un merge o despliegue.

**Decisión final:** mantener congelada la investigación previa y esta corrida negativa. Continuar investigación acotada sobre identificabilidad de efectos, feedback observable y sesgo composicional; en producto priorizar los problemas medidos de recuperación y olvido. Un observador aislado puede aportar evidencia futura, pero el controlador F10-O no está listo para integrarse.

## Reproducibilidad y entregables

`science.json`: métricas y gates completos. `replay.json`: reconstrucción de todos los episodios y planes. Por réplica: logs TRAIN comprimidos, checkpoints, tablas, manifiestos congelados y filas de evaluación. Se entrega paquete con código, generador congelado y checkpoints originales, sin credenciales ni datos de usuarios. El README del paquete contiene las órdenes de auditoría/reproducción. Reproducir la corrida es una comprobación numérica, no una confirmación en semillas nuevas.
