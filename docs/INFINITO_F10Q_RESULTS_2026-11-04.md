# INFINITO F10-Q — Informe de MCP y SQLite en modo shadow

**4 de noviembre de 2026. F10Q_MCP_SHADOW_PASS científico, auditoría técnica completada. No se ha modificado el producto.**

## Resultado

Con un servidor MCP por stdio ejecutado realmente y una herramienta genérica sql_select de consultas SELECT a SQLite de solo lectura, las memorias de efectos de F10-P reconstruyeron SRX, una cadena no ejecutada en los 1.024 programas TRAIN por réplica. Cuatro réplicas, datos sintéticos exclusivamente, experimentos independientes por semilla.

El modelo E100 entrenado con salidas SQL tuvo **1536/1536 respuestas correctas (100%)** sobre casos nuevos con evidencia válida, y el adaptador bloqueó la respuesta en **512/512 casos con SELECT sin resultados o payload NULL**. IMPORTANTÍSIMO: **la memoria E100 ORIGINAL congelada de F10-P y sin entrenamiento MCP adicional también obtuvo 1536/1536**. Por tanto, este es un resultado de portabilidad de arquitectura y ejecución en MCP real, NO una demostración de aprendizaje de nuevas operaciones, superioridad incremental o preparación para despliegue.

## Protocolo e integridad

Prerregistro [INFINITO_F10Q_PREREG_2026-11-04.md](INFINITO_F10Q_PREREG_2026-11-04.md) publicado primero en commit **311455d03d81bf01d79273b235766a6ec12a98d3**, antes del código. Código del producto, F10-P y F10-P-C inalterados. Primer [run 41899660986](https://github.com/webtilians/principiodelTodo/actions/runs/41899660986) FAIL TÉCNICO en instalación del SDK antes de cualquier entrenamiento o datos OOD (rama de laboratorio sin pyproject.toml). Única corrección de instalación en commit **332907a8b578278d5b9392d107c5fad88d11afbc**. Run científico original único [41899822515](https://github.com/webtilians/principiodelTodo/actions/runs/41899822515), los cuatro trabajos de réplica y auditor independiente SUCCESS.

Cuatro réplicas, **1024 programas realmente ejecutados TRAIN/r, 4096 en total**; SRX excluido TRAIN. Cinco memorias E100/E50/E0/E_noX/E_shuffleX recibieron las mismas llamadas, pero diferentes máscaras de los eventos SQL públicos. Memorias congeladas SHA antes de generar OOD. Frozen_F10P cargó modelo E100 original SHA-autenticado del run 41888234405 sin entrenamiento. Zero responde cero sin observar herramientas.

Mundos OOD nuevos: 128 mundos/r, cuatro variantes por mundo. **96 mundos/r válidos** con valores binarios, última evidencia acreditada versionada, contradictorio más reciente sin acreditar y cambio de acreditación S_after. **32 mundos/r inválidos** por ausencia de fila acreditada o payload NULL. Solo SOURCE=MEMORY y FLIP=1, ausente en entrenamiento. No se abrieron las semillas confirmatorias 29300000+.

La prueba efectuó **9.106, 9.110, 9.140 y 9.084 tool calls SELECT MCP**, respectivamente (36.440 total, TRAIN+OOD e intervenciones). El endpoint rechazó INSERT en cada réplica. Los únicos datos persistidos fueron fixtures transitorios sintéticos y los registros de auditoría; ningún dato real del usuario ni memoria MCP personal.

## Tabla original de resultados (solo 1536 casos con respuesta válida)

| Réplica | E100 MCP entrenado | E50 | E0 solo terminal | E_noX | X permutado | Frozen_F10P cero-shot | Zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **100%** | 100% | 50,00% | 50,00% | 50,00% | **100%** | 50,00% |
| 1 | **100%** | 100% | 50,00% | 50,00% | 50,00% | **100%** | 50,00% |
| 2 | **100%** | 100% | 51,56% | 51,56% | 51,56% | **100%** | 51,56% |
| 3 | **100%** | 100% | 48,96% | 48,96% | 48,96% | **100%** | 48,96% |
| **Total** | **1536/1536** | **1536/1536** | **770/1536 (50,13%)** | 50,13% | 50,13% | **1536/1536** | 50,13% |

El control sin observaciones intermedias y el control sin observar XOR colapsan al nivel de un programa vacío; sus valores son comparables porque usan el mismo learner y las mismas experiencias. E100 propuso SRX en los 1536 casos válidos. El mismo programa con XOR SQL desactivado perdió 100 puntos, y al quitar la cláusula de acreditación de SQL perdió 100 puntos: **es un entorno deliberadamente construido para que el distractor no acreditado más reciente contradiga la respuesta correcta**. No debe interpretarse ese extremo como evidencia de robustez frente a evidencias reales menos estructuradas.

La abstención en los **512 casos inválidos** procede del **adaptador externo con validación explícita de filas ausentes o NULL**, NO de una decisión aprendida autónomamente. E100 y la memoria congelada de F10-P aprendieron la misma estrategia observable bajo una semántica de operación predefinida.

## Puertas científicas fijadas antes de los datos

| Puerta | Réplicas que cumplen |
|---|---:|
| G0 MCP real, 1024 TRAIN, cero SRX, SHA y endpoint SELECT-only | **4/4** |
| G1 85% OOD, CI inferior >75%, estratos >=75%, SRX >=90% | **4/4** |
| G2 ventaja de salidas reales completas sobre E0 y E_noX | **4/4** |
| G3 efectos causales sobre XOR SQL y acreditación | **4/4** |
| G4 abstención sobre ausencias/NULL, cero escrituras | **4/4** |
| G5 diagnóstico E50 y Frozen_F10P | **4/4** |

Bootstrap 2000 remuestreos por mundo base, agrupando cuatro variantes; los 1536 casos no son 1536 mundos independientes. Auditor independiente solo de JSON/NumPy, sin importar MCP/SQLite/learner, artefacto f10q-independent-mcp-sqlite-science del run original.

## Significado y decisión

Sí: el mecanismo ya aprendido funciona al recibir eventos traducidos de **resultados de consultas nativas mediante MCP verdadero**, y un piloto shadow de lectura puede aislarse y abstenerse por falta de información. **No**: un servidor externo al laboratorio, respuestas no previstas, nuevas semánticas, ontología aprendida, políticas libres, seguridad de agentes que escriben memoria, ni beneficio frente al mecanismo original; Frozen_F10P empató perfectamente. El servidor MCP, las consultas SELECT, el parser de filas y el planificador fueron escritos por nosotros, aunque SQLite ejecuta realmente las operaciones. Por tanto el 100% no acredita inteligencia autónoma general ni ventaja de producto.

**NO-GO para integrar controlador en el núcleo del producto.** Siguiente estudio recomendable: conectarse al servidor **MCP real de INFINITO** (remember/recall/forget) sobre un SQLite exclusivamente sintético e independiente, sin acceso a las memorias personales. Medir beneficio contra el comportamiento actual y una política simple documentada, con casos de actualización, selección, olvido y permiso. Si no mejora ese control, detener la integración en vez de optimizar el porcentaje del simulador. PR #73 DRAFT, sin merge ni despliegue.