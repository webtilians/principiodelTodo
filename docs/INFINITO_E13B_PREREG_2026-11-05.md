# E13-B — Conjuntos y relaciones temporales
**Prerregistro prospectivo 2026-11-05: publicar este documento ANTES de implementar E13-B o abrir datos DEV/CONFIRM.** Código original E13-A de partida, fuente inmutable. No modificar src ni memorias personales.

## Precondición
F10-S-FIX PR #78: fix SQLite BEGIN IMMEDIATE validado en laboratorio, pero sigue DRAFT y no fusionado en master. Se estudia E13-B en un entorno secuencial e independiente de ese bug; no integrarlo implícitamente.

## Hipótesis y cuatro brazos
A CURRENT: producto AgentMemory, recall semántico con filtro de proyecto y presupuesto idéntico.
B SETS: mismo producto, índice derivado por proyecto/entidad/atributo y tags, sin enlaces entre entidades.
C RELATIONS: mismo producto, grafo ligero de referencias al texto original con versión temporal y enlaces explícitos de responsable→herramienta; SIN conjuntos ni resumen. Hipótesis principal: C consigue la mayor parte de las ventajas de E13-A con menos complejidad que D.
D HYBRID: B+C con actualización incremental de índices y borrado inmediato de toda referencia a fuentes eliminadas.
Todas conservan las fuentes y procedencia originales. Los índices son reconstruibles, nunca generan hechos, no elevan propuestas a hechos. No preferir D por diseño. La extracción estructurada de la gramática común es aportada por investigadores, no aprendida autónomamente.

## Datos, presupuestos y particiones
Generador de E13-A reutilizado en NUEVOS namespaces SHA256 INFINITO/E13-B/v1/partition/replica/world. DEV: 8 mundos réplica 0, ids 0..7. CONFIRM: 32 mundos, 4 réplicas x8, totalmente distintos del DEV y de E13-A; no abrir hasta freeze. Cada mundo: >150 fuentes, duplicados, cambios por fecha, relaciones de 2 saltos, contradicciones, propuestas, borrados y reinicios. Añadir un segundo proyecto/tenant sintético que contradiga algunas claves, sin contaminación de respuestas o índices. Se evalúan cuatro queries por cada una de cinco familias primarias (actual/pasado/cronología/relación/contradicción), 20 por mundo y brazo, con 512 tokens aproximados (ceil caracteres/4, incluyendo procedencia); 256 tokens es sensibilidad ejecutada sobre copia de índice congelada, sin estado adaptativo mutado. Probes forgotten/hypothesis/unknown/survivor en seguridad fuera de métrica primaria. Mismo ranker real multilingual del E13-A, revision y SHA del modelo fijados. Ningún dato del usuario ni servidores MCP reales.

## Métricas y auditoría
Guardar contexto, ids, fecha, estado, modalidad, texto, respuesta, resultado por query, eventos de mantenimiento, bytes persistentes SQLite compactado+fuentes+índice, CPU de ingestión/mantenimiento/consulta, p50/p95 y coste de actualizaciones. Modelo compartido contado aparte. Auditor independiente SIN importar motor/índices/generador reinterpreta únicamente los textos y la procedencia, comprueba aislamiento, ausencia de fuente borrada incluso después de reinicio, presupuestos, negativos, integridad y métricas; ninguna fuga de verdad del generador al lector. Bootstrap pareado 5000 remuestreos por MUNDO (32 bloques) con RNG 1313 e IC95. Reportar réplicas por separado.

Antes de CONFIRM: congelar en GitHub SHA de script, pruebas, auditor, config, modelos, prompts y outputs DEV. Runner rechaza confirmar si el manifiesto de freeze no está comprometido en Git. Una sola confirmación nueva, no modificar umbrales después de ver resultados.

## LLM real
G5 obligatorio: Qwen/Qwen2.5-0.5B-Instruct o un modelo generativo abierto equivalente fijado por revision SHA ANTES de confirmación. 8 mundos completamente independientes de DEV y CONFIRM deterministas; una pregunta relation y una current por mundo, cuatro brazos: 64 inferencias reales. Igual prompt, decoding determinista, mismo contexto recuperado, mismas condiciones. Modelo recibe solo pregunta y evidencia; no gold ni instrucciones del generador. Publicar raw outputs y errores, parser igual, modelo y revision. Si el LLM no puede ejecutarse o descargarse: E13B_INCONCLUSIVE_LLM, NUNCA PASS.

## Gates preregistrados
G0 integridad: cuatro brazos, fuentes sin alterar, namespaces disjuntos, procedencia e aislamiento proyecto/tenant, presupuesto y hashes verificados, modelos congelados antes de CONFIRM.
G1 C relacional: >=90% preguntas relation a 512 tokens y ventaja C vs A y B >=10pp cada una, con límite inferior IC95 pareado por mundo >0.
G2 actualidad: C y D no degradan current más de 2pp frente A, hechos pasados y cronologías y conflictos reportados sin ocultar caídas.
G3 seguridad dura: cero datos olvidados resucitados tras forget/restart, cero propuestas como hechos, cero IDs de otro proyecto/tenant en índices o contexto, cero daño a supervivientes.
G4 coste: C <=1.35x bytes de A, <=2x CPU total de A y <=2x p95 consulta. D se mide, sin requisito automático de ganar.
G5 LLM real: 64 inferencias; C gana al menos 2/8 preguntas de relación frente A sin perder ninguna de las otras ocho y no pierde >1 pregunta de actualidad frente A. Umbral conservador para piloto pequeño, no prueba de generalización abierta.
Global PASS solo si G0-G5. Integridad FAIL -> INCONCLUSIVE; falta de LLM -> INCONCLUSIVE_LLM; resultado íntegro pero insuficiente -> FAIL. Si D supera C <2pp o con >15% coste adicional, preferir C siempre que cumpla gates. No integrar automáticamente ningún brazo ni fusionar master.
