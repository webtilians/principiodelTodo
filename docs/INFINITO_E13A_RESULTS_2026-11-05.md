# E13-A — La organización ayuda; la adaptación no supera el gate

**Resultado confirmatorio: E13A_FAIL. No integrar en producto.** La vista
derivada recupera más evidencia que CURRENT y RAG en este lenguaje sintético,
pero no demuestra la ventaja mínima de adaptarse frente a consolidar de forma
periódica. Tampoco ahorra almacenamiento. Se conservan dos fallos de actualidad.

## Trazabilidad y alcance

- Base de producto: master `61a3329e46b797812ca2b01b7509c1615ac0fc83`.
- Rama: `experiment/e13a-adaptive-consolidation-20261105`, desde master.
- Prerregistro publicado **antes del código**: `29482d4`.
- Código y evidencia DEV: `41a8879`; manifiesto congelado publicado: `4254560`.
- Confirmación ejecutada después de ese último commit. Código/modelo cotejados
  contra [el manifiesto](INFINITO_E13A_FREEZE.json) antes de abrir HOLDOUT.
- [Protocolo](INFINITO_E13A_PREREG_2026-11-05.md),
  [correcciones DEV](INFINITO_E13A_DEV_NOTES.md),
  [datos y reproducción](../observatory_results/e13a/README.md),
  [resultado completo](../observatory_results/e13a/holdout/summary.json).
- Se reutiliza la propuesta E13-A existente en INFINITO, sin escribir, borrar,
  consolidar ni entrenar con recuerdos reales. Solo SQLite temporal y corpus
  sintético. Ningún cambio en `src/`, ningún servicio desplegado, ninguna fusión.

Este es un experimento de **recuperación de evidencia con un lector determinista**,
no un ensayo de respuestas de un LLM ni del servidor MCP remoto. Invoca la clase
real `AgentMemory` detrás de las herramientas MCP; no transporta llamadas por
HTTP/stdio. La extracción estructural está programada para una gramática pequeña,
no aprendida a partir de experiencia. No extrapolar los porcentajes a texto libre.

## Qué se comparó

CURRENT usa remember/recall/forget reales y el MeaningRanker local del proyecto.
El adaptador de presupuesto vuelve a ordenar los 20 resultados por similitud
antes de recortar y después presenta lo elegido cronológicamente. Esto evita
castigar el orden de presentación de recall. RAG guarda todas las fuentes y
recupera los 20 textos más cercanos con el mismo modelo.

ADAPTIVE conserva además todas las fuentes en JSONL y una vista JSON reconstruible:
deduplicación factual con fecha/modalidad, etiquetas de entidad/atributo, enlaces
explícitos responsable→herramienta, cronologías e IDs de procedencia completos.
No genera nuevos hechos ni reescribe originales. Reconstruye por escrituras,
densidad de duplicados o uso reciente, sin etiquetas de acierto. PERIODIC usa
exactamente la misma vista y recuperación con disparador fijo cada 24 escrituras.
Tras forget ambos invalidan y reconstruyen inmediatamente desde fuentes vivas.

193 fuentes por mundo, 16 entidades principales, fechas fuera del orden de
ingestión, duplicados literales/paráfrasis, cambios, propuestas y conflictos
simultáneos. 8 mundos DEV reutilizados solo en desarrollo; **32 mundos HOLDOUT
disjuntos**, cuatro réplicas de ocho. 6.176 fuentes de entrada en confirmación.
Cada brazo tiene 640 consultas primarias a 512 tokens estimados, más controles,
presupuesto 256 y borrado/reinicio: **7.424 consultas puntuadas entre los cuatro
brazos**, además de consultas de uso sin etiquetas. Familias compartidas entre
DEV y HOLDOUT; cambian entidades, valores y orden, no la gramática.

## Resultado primario a 512 tokens estimados

| Sistema | Aciertos / 640 | Exactitud |
|---|---:|---:|
| CURRENT | 553 | 86,406 % |
| RAG | 547 | 85,469 % |
| PERIODIC | 632 | 98,750 % |
| ADAPTIVE | 638 | 99,688 % |

Comparaciones pareadas por mundo, IC bootstrap 95%, 5.000 remuestreos:

| ADAPTIVE frente a | Diferencia | IC 95% |
|---|---:|---:|
| CURRENT | +13,281 pp | [+11,719; +15,000] pp |
| RAG | +14,219 pp | [+12,344; +16,094] pp |
| PERIODIC | +0,938 pp | [+0,156; +1,875] pp |

La última diferencia es inferior a los **2 pp preregistrados**, aunque su IC
puntual esté por encima de cero. No sustituir significación por utilidad mínima.

| Familia (128 consultas cada una) | CURRENT | RAG | PERIODIC | ADAPTIVE |
|---|---:|---:|---:|---:|
| Hecho actual | 127 | 126 | 126 | 126 |
| Hecho antiguo | 128 | 127 | 128 | 128 |
| Cronología completa | 126 | 124 | 125 | 128 |
| Relación de dos saltos | 44 | 42 | 126 | 128 |
| Conflicto simultáneo: ambos valores | 128 | 128 | 127 | 128 |

La mejora proviene principalmente de enlaces explícitos, no de resolver mejor
todos los hechos actuales. ADAPTIVE pierde un acierto de actualidad frente a
CURRENT; cae 0,781 pp en esa familia, dentro de la tolerancia preregistrada de 2pp.
Por réplica (de 160): CURRENT 137/134/144/138; RAG 136/132/143/136;
PERIODIC 159/160/158/155; ADAPTIVE 160/160/160/158.

## Fallos, seguridad y procedencia

Los dos errores ADAPTIVE ocurren en `world-3-0.json` y `world-3-5.json`, pregunta
de estado actual. Se conserva la respuesta antigua y la referencia correcta
en cada artefacto. La vista puede estar pendiente de reconstrucción; prioriza
nodos antiguos y relega escrituras pendientes al final del contexto. Una política
de actualización incremental o revalidación de actualidad merece otra prueba,
pero **no se corrigió el código después de abrir HOLDOUT**.

En ADAPTIVE: propuestas tratadas como desconocidas 32/32 en cada presupuesto;
desconocidos 32/32 en cada presupuesto; olvido 128/128 entre comprobación inmediata
y tras reinicio/reconstrucción; supervivientes ajenos al borrado 320/320. No se
observa resurrección ni procedencia inválida. Los otros brazos también pasan
128/128 controles de olvido y los controles de propuesta/desconocido.

El auditor separado reinterpreta contextos sin importar el generador, producto
ni consolidación; coteja IDs, texto factual, modalidad, proyecto, fecha, enlaces,
fuentes supervivientes y ausencia del contenido borrado en filas SQLite
exportadas. Audita 7.424 consultas. El borrado recibe IDs explícitos seleccionados
por el harness: esto NO evalúa si un agente entiende una petición natural de
olvido, ni borra copias ya vistas por un LLM, backups, discos remotos o caches ajenas.

## Espacio y coste

Bytes medios por mundo, antes de borrar. SQLite compactado real; archivo e
índice realmente escritos en disco, UTF-8 sin comprimir. Fuentes originales
incluidas, modelo compartido separado.

| Sistema | Bytes totales medios | CPU total, 32 mundos | p95 consulta |
|---|---:|---:|---:|
| CURRENT | 413.696 | 17,328 s | 4,093 ms |
| RAG | 29.497 | 9,703 s | 0,523 ms |
| PERIODIC | 461.622 | 15,547 s | 1,101 ms |
| ADAPTIVE | 461.001 | 16,188 s | 1,203 ms |

ADAPTIVE suma aproximadamente 29.497 bytes de fuentes y 17.808 de índice a
SQLite: **+11,435 % de almacenamiento frente a CURRENT**, no compresión.
RAG persiste JSONL y recalcula vectores al abrir; no tiene las tablas/embeddings
SQLite de CURRENT, por lo que su menor tamaño no aísla el efecto de deduplicar.
El modelo exportado ocupa 155.569.138 bytes una vez, compartidos por los brazos;
no se cuentan caches RAM como disco ni se atribuye el modelo a cada mundo.

ADAPTIVE hace **416 reconstrucciones y procesa 50.886 filas**, frente a 320 y
39.744 en PERIODIC. CPU de mantenimiento registrada: 0,125 s en ambos; tiempo
de pared: 0,302 s frente a 0,213 s. No hay ahorro de mantenimiento demostrado;
hay más reconstrucciones (+30%) y más filas procesadas (+28%). Esas mediciones
breves de CPU sufren cuantización: en DEV cambiaron el gate entre ejecuciones.
La confirmación y su repetición dan G4 FAIL. No interpretar diferencias pequeñas
de CPU como una estimación robusta de eficiencia o coste económico.

0 llamadas de pago y 0 tokens de API; no se entrenó ningún modelo. Coste de
proveedor por inferencia: cero. Se reporta cómputo local, no una factura ficticia.
La preparación común de vectores se mide y carga por igual a cada brazo; las
consultas usan caché caliente. Carga/descarga inicial de pesos fuera del tiempo
operativo. Hardware: Windows 10.0.26200, Intel64 Family 6 Model 170 Stepping 4,
Python 3.11.9. No son cifras de un servicio concurrente ni de un benchmark de I/O
durable: no se fuerza fsync del JSONL después de cada escritura.

## Sensibilidad, límites y reproducibilidad

A 256 tokens: CURRENT 428/640, RAG 428/640, PERIODIC 632/640, ADAPTIVE 640/640.
**No interpretar el 100% como mejora causada por bajar presupuesto**: el runner
evalúa 256 después de 512 y la política sigue recibiendo uso, por lo que puede
haber reconstruido entre ambas tandas. Es una sensibilidad secuencial, no una
comparación limpia sobre la misma instantánea congelada. El endpoint primario
preregistrado a 512 no depende de esa segunda tanda.

La adaptación combina varios mecanismos; el estudio no separa deduplicación,
etiquetas, vínculos y cronología en ablaciones individuales. No prueba cambios
de responsable, lenguaje ambiguo, nueva gramática, multiusuario, concurrencia,
retención de décadas ni crecimiento de memoria a gran escala. Los identificadores
aleatorios y las plantillas favorecen extracción exacta; el resultado valida
solo esta intervención estructural acotada.

La repetición íntegra de los mismos 32 mundos reproduce exactamente contextos,
predicciones, referencias y aciertos, con tiempos distintos: **no es una nueva
confirmación independiente**. Véanse [auditoría](../observatory_results/e13a/audit-holdout.json)
y [reproducción](../observatory_results/e13a/audit-replay.json). Semillas HOLDOUT
consumidas; no reutilizarlas como una nueva confirmación.

Pruebas locales: 12 contratos E13-A; 155 pruebas del producto aprobadas y 10
omitidas. Se añade workflow para ejecutar contratos y auditar artefactos en PR,
sin descargar el modelo. El estado de CI se consulta en el PR.

## Gates y decisión

| Gate | Resultado | Motivo |
|---|---|---|
| G0 integridad | PASS | Presupuestos, procedencia, disjunción y reproducción auditados |
| G1 recuperación | PASS | >5pp e IC positivo frente a ambos baselines, sin caída >2pp |
| G2 seguridad acotada | PASS | Sin resurrecciones, promoción de hipótesis ni daño a supervivientes en ADAPTIVE |
| G3 coste acotado | PASS | Espacio <=1,5x, CPU <=2x y p95 <=2x CURRENT |
| G4 ventaja adaptativa | **FAIL** | +0,938pp <2pp y sin ahorro >=20% de CPU de mantenimiento |

**No integrar ADAPTIVE en el producto.** La evidencia justifica estudiar vistas
relacionales conservadoras, pero no demuestra que este disparador adaptativo
compense su coste. Antes de otra confirmación: corregir en DEV la revalidación de
escrituras pendientes, separar sensibilidad de presupuesto de adaptación temporal
y medir costes con cargas suficientemente largas. Cualquier siguiente ensayo
requiere otro prerregistro y mundos nuevos. PR en borrador contra master;
la fusión queda exclusivamente al mantenedor.
