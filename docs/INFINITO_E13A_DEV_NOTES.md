# E13-A — Registro DEV y aclaraciones antes de HOLDOUT

Prerregistro publicado: commit `29482d4`, antes del harness. Base intacta.

Primera ejecución DEV (`observatory_results/e13a/dev`): CURRENT 52.50%, RAG
86.25%, ADAPTIVE 100%, PERIODIC 98.75%, endpoint de 160 consultas. G4 falla.
No se cambian umbrales ni política adaptativa al observar estos resultados.

Antes de congelar se corrigen dos asimetrías del harness:

1. AgentMemory añade el proyecto al texto cuando falta. Inicialmente la caché
   común solo preparaba el texto sin proyecto, favoreciendo en tiempo a otros
   brazos. Ahora TODAS las fuentes incluyen desde su creación el mismo proyecto.
2. recall entrega sus 20 elegidos en orden cronológico. Recortar en ese orden
   castigaba artificialmente la memoria actual. El adaptador común de presupuesto
   vuelve a ordenar esos mismos 20 por similitud ANTES de seleccionar lo que cabe;
   después los presenta cronológicamente. No amplía el conjunto recuperado.
   Por ello CURRENT significa API actual + adaptador de presupuesto explícito,
   no reproducción literal de una respuesta MCP sin límite de tokens.
3. Se añaden consultas a cinco hechos ajenos al borrado tras cada fase para
   detectar daño colateral; se incluyen en G2, no en el endpoint primario.
4. Ahorro de CPU no puede declararse si el tiempo de referencia es cero por
   resolución del reloj de Windows. Se exige denominador positivo en G4.

Se conserva DEV inicial y se vuelve a ejecutar el mismo DEV en `dev-final`;
no es una muestra nueva. Los resultados iniciales NO son confirmatorios.
No se ha generado ni puntuado ningún mundo HOLDOUT. Las pruebas comprueban solo
identificadores de semillas HOLDOUT para disjunción, nunca contenido o respuestas.

Los tiempos de consulta usan caché de vectores preparada por igual; se carga a
cada brazo el mismo coste medido de preparación. Preparación incluye preguntas
sin etiquetas; no interviene en la política de consolidación. Carga/descarga del
modelo se excluye del coste operativo. SQLite se compacta para medir bytes;
archivos y vista se miden por serialización JSON UTF-8 canónica, sin compresión.
No se cuentan cachés RAM como disco; RAG vuelve a calcular vectores al abrir.
El coste de mantenimiento es subconjunto de ingestión/consulta/olvido, no se suma
dos veces. La reconstrucción tras reinicio se incluye también en CPU total de
servicio; se exporta el índice antes/después para auditar la procedencia.

Limitación: extracción por gramática explícita y consultas estructuradas en
español. No entrenar un modelo, no afirmar comprensión de textos libres. Relaciones
son vínculos explícitos responsable→herramienta, no inferencias causales generales.

Instrumentación final (`dev-frozen`): persistencia real de archivo JSONL y vista
JSON, reinicio leyendo ese archivo, auditoría de nodos/vínculos y filas tras
borrado. No cambian política, corpus, umbrales ni orden de recuperación.
Resultados discretos: CURRENT 138/160, RAG 135/160, ADAPTIVE 160/160,
PERIODIC 158/160. El gate G4 basado en CPU cambió de FAIL a PASS entre
ejecuciones DEV: los intervalos muy breves acumulan ceros y saltos de reloj.
Esto es una limitación de medición, NO evidencia estable de ahorro. Conservar
el resultado formal del gate; contrastarlo con tiempo de pared, reconstrucciones,
filas procesadas y repetición. No usar un PASS ruidoso para justificar integración.
Antes de HOLDOUT se añade solo una comprobación de hashes del modelo congelado.
El p95 de servicio final es el percentil agregado de todas las consultas puntuadas
(ambos presupuestos y fases), no la media de percentiles por mundo.

Modelo descargado: minishlab/potion-multilingual-128M, revisión observada en
la caché `73908c3438cf03b6a01bcb9611d62b23d0726f08`. Archivo convertido por el
exportador del proyecto; pesos y ficheros auxiliares identificados por SHA256.
