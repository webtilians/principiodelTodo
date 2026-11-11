# E13-A — Consolidación autónoma adaptativa

Prerregistro prospectivo, 2026-11-05. Se confirma en git antes de implementar
o ejecutar el experimento. Base: master `61a3329e46b797812ca2b01b7509c1615ac0fc83`.
Referencia: propuesta E13-A ya registrada en INFINITO con id
`22b5c958-5578-446f-ab9b-20b728a2bded`. No se copia ningún recuerdo personal al corpus.

## Pregunta y alcance

¿Una vista derivada, reconstruible y adaptada al uso mejora la recuperación de
evidencia frente al AgentMemory actual y a almacenar todo con recuperación
semántica, conservando fuentes, temporalidad, modalidades y olvido?
Es un estudio offline de recuperación y lectura determinista de un lenguaje
sintético limitado, NO comprensión general, entrenamiento de un LLM, ni prueba
del despliegue remoto. No se cambia `src/`, no se llama al MCP personal ni a APIs
de pago, no se usa una base de datos existente. Las consultas de memoria de la
conversación sirven exclusivamente para identificar esta propuesta.

## Sistemas y presupuesto

- CURRENT: AgentMemory auténtico, remember/recall/forget, SQLite temporal,
  MeaningRanker local potion-multilingual-128M del proyecto.
- RAG: todas las fuentes originales, mismo ranker, orden semántico y contexto
  cronológico. Sin deduplicar ni resumir. Borrado explícito de fuentes.
- ADAPTIVE: CURRENT más archivo de fuentes y vista derivada con deduplicación
  por contenido factual/fecha/modalidad, etiquetas de entidad/atributo, vínculos
  explícitos, cronologías y referencias a todas las fuentes. Nunca sustituye el
  texto original. Reconstruye al acumular 24 escrituras; antes si hay al menos
  8 y >=20% de duplicados, o 6 y tres consultas al mismo proyecto. Tras forget
  invalida inmediatamente todas las derivaciones y reconstruye desde vivos.
- PERIODIC: mismo mecanismo, solo disparador fijo cada 24 escrituras. Ablación
  del beneficio específico de adaptarse; no basta superar CURRENT.

Mismo presupuesto de contexto estimado (ceil(caracteres/4), incluyendo fechas,
modalidad y procedencia): 512 primario, 256 sensibilidad. Mismo lector limitado
para todos; no recibe respuestas, semillas ni metadatos privados del generador.
Ranking y lector solo ven texto/metadata que los sistemas podrían observar.
Los vínculos no autorizan inventar hechos. Una propuesta no responde a una
consulta factual; un conflicto simultáneo exige ambos valores y sus fuentes.

## Datos y particiones

Generador versionado: 16 entidades por mundo, historiales con estados en fechas
distintas, duplicados literales y paráfrasis, responsables y herramientas
vinculados, propuestas, conflictos en la misma fecha y distractores. Al menos
150 registros por mundo para ejercitar la selección real. Orden de ingestión
aleatorio, distinto del tiempo del hecho; todas las fechas son sintéticas.
Consultas de hechos actuales/antiguos, cronología, relación de dos saltos,
contradicción, hipótesis y desconocidos, antes y después de borrados. Un grupo
de consultas de uso sin etiquetas se intercala entre escrituras. El borrado
incluye fuente de un vínculo y duplicados, seguido de reconstrucción y reinicio.

Semillas: SHA256 de `INFINITO/E13-A/v1/{partition}/{replica}/{world}` convertido
a entero completo, sin truncar. DEV: replica 0, world 0..7. HOLDOUT: replicas
0..3, world 0..7. Nombres de partición distintos, entidades y valores aleatorios
nuevos. No reutiliza semillas numéricas históricas. Familias compartidas:
confirmación de mundos nuevos, NO generalización a familias desconocidas.
DEV se puede inspeccionar; cualquier ajuste y fallo se declara. Antes de abrir
HOLDOUT se confirma código, tests, configuración elegida y hashes en git.
Una sola ejecución confirmatoria; repetir para reproducibilidad no crea otra
muestra independiente. No retocar tras ver HOLDOUT. Error de harness invalida
la confirmación y debe declararse, no reciclar sus semillas como nuevas.

## Medidas y gates fijados

Por consulta guardar contexto, predicción, referencia y acierto exacto; por
mundo y sistema, proporción por familia y conjunta. Informar cobertura de
evidencia y modalidad; los controles negativos no inflan el endpoint primario
(factual actual/antigua, cronología, relación y contradicción).
Comparación pareada por mundo, IC bootstrap 95% (5000 remuestreos, semilla
estadística 13), sin tratar preguntas del mismo mundo como independientes.
Registrar bytes SQLite compactados, fuentes, índice y total; filas/nodos,
reconstrucciones, CPU y tiempo de ingesta/mantenimiento/consulta, p50/p95,
tokens estimados de contexto. Coste de API = cero; informar cómputo medido,
no inventar una tarifa monetaria ni descontar el coste de conservar fuentes.
Pesos del modelo compartidos se miden aparte; descarga/setup fuera de latencia.

G0 integridad: particiones disjuntas, trazabilidad completa, presupuestos
respetados, reproducibilidad de resultados discretos y fuentes no reescritas.
G1 recuperación: ADAPTIVE mejora >=5 puntos porcentuales frente a CURRENT Y
RAG en HOLDOUT a 512; límite inferior de ambos IC >0; ninguna familia cae >2pp.
G2 seguridad: cero resurrecciones tras forget/reconstrucción/reinicio, cero
propuestas elevadas a hechos, cero procedencias inválidas o cambios de fuentes.
G3 eficiencia: total de almacenamiento <=1.5x CURRENT, CPU total <=2x CURRENT,
p95 consulta <=2x CURRENT (medición local exploratoria, hardware declarado).
G4 adaptación: frente a PERIODIC, mejora >=2pp de recuperación O reduce >=20%
CPU de mantenimiento, sin empeorar recuperación >1pp ni violar G2.
Todos necesarios para E13A_PASS; resultado válido que falle alguno = E13A_FAIL;
integridad/entorno sin evidencia suficiente = E13A_INCONCLUSIVE.

## Decisión y publicación

Un FAIL documenta límites y detiene cualquier integración. PASS solo justifica
otra validación con lenguaje natural y agentes reales: tampoco integra producto
automáticamente. Publicar código, pruebas, hashes, datos sintéticos regenerables,
resultados completos incluidos negativos, informe y PR draft contra master.
El mantenedor fusiona. No entrenar modelos con recuerdos personales.
