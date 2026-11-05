# INFINITO F11-A v3 — verificación independiente dentro de TRAIN

**5 de noviembre de 2026. Resultado científico restringido: NO se confirma una ventaja incremental de A3.** Estado experimental: `TRAIN_HELDOUT_DIAGNOSTIC`; NO es PASS/FAIL de los gates confirmatorios F11-A, porque DEV y CONFIRM no se han abierto.

## Orden cronológico y reproducibilidad

1. Prerregistro original de F11-A: PR #86, integrado en master; umbrales G0–G5 inalterados.
2. TRAIN v1 PR #87: saturación en los mismos mundos de entrenamiento. TRAIN v2 PR #88: ventaja aparente de A3, pero A2 infracapacitado, A1 con sesgo de actualidad y colector controlado parcialmente por A3.
3. Enmienda prospectiva F11-A v3: `docs/INFINITO_F11A_V3_VERIFY_PREREG_2026-11-05.md` confirmada en **c3544d2ecf6ff32efbde13bb6cafb373a49e76c2**, ANTES del código v3 y de abrir resultados v3.
4. [Ejecución GitHub Actions 42037750825](https://github.com/webtilians/principiodelTodo/actions/runs/42037750825), launch commit **08b54b26e92a9bbdea7312c35a593146e167da35**. Contratos de v2 y v3 SUCCESS; las cuatro réplicas de TRAIN / TRAIN-heldout SUCCESS. Resultados crudos por mundo, trazas de entrenamiento, sumarios y checkpoints guardados como cuatro artefactos de Actions, `f11a-v3-train-heldout-r0..r3`.
5. Reproducción de la misma verificación: `python -m unittest tests.test_f11a_verify_v3 -v` y `python -m observatory.f11a_verify_v3 --replica N --fit-worlds 384 --holdout-worlds 128 --episodes 30 --out DIR`.

## Diseño realizado

- Cada réplica: **384 mundos TRAIN-FIT** indices 0..383, 30 episodios/mundo; **128 mundos TRAIN-heldout** indices 384..511, completamente excluidos del aprendizaje. Total 1.536 mundos para ajustar y 512 mundos disjuntos para probar, distribuidos en cuatro réplicas.
- Exploración **100% aleatoria y común** para los brazos A1/A2/A3. Ninguna experiencia es elegida por A3 y el mismo log de acción/observación/recompensa se entrega a todos.
- A1 almacena episodios literales y recupera candidatos históricos de distintas épocas (K=12), sin política Q persistente. A2 aprende atribución FIFO de efectos retardados con señales públicas y planifica hasta 12 pasos en un **modelo simplificado guiado por el esquema del problema**; no lee reglas ocultas, pero tampoco es un agente de planificación completamente general. A3 aprende un máximo de cuatro opciones por estado inicial, cae en ese A2 cuando no usa una opción; A3_ablated usa A2 solo. A4 es una heurística informada y no forma parte del aprendizaje.
- Igualdad de evidencia y pares de semillas, fases de reglas originales y cambiadas por separado. Probes heldout no actualizan modelos ni opciones.
- Reglas físicas y cuatro modos de tarea permanecen en la misma familia que v2. **Heldout por semillas no es transferencia de estructura, ni prueba G2 de composición nueva**. No se hizo entrenamiento online para medir G3.

## Datos de ajuste

Cuatro réplicas × 384 mundos × 30 episodios = **46.080 episodios**, todos recogidos mediante exploración aleatoria y misma cantidad para todos los brazos. Éxitos de exploración por réplica: r0=1502, r1=1544, r2=1393, r3=1527; total **5.966**. Cada réplica almacenó **64 opciones distintas** al aplicar el límite de cuatro por estado inicial; total 256 opciones almacenadas (no necesariamente diferentes entre réplicas).

## Resultados en mundos nuevos SOLO dentro de TRAIN

Cada fase: cuatro réplicas × 128 mundos = 512 episodios por brazo. Cada mundo base aparece en ambas fases; no tratar ambas como observaciones independientes.

| Réplica | A1 antigua | A2 antigua | A3 antigua | A1 nueva | A2 nueva | A3 nueva |
|---|---:|---:|---:|---:|---:|---:|
| r0 | 29/128 | 0/128 | 35/128 | 34/128 | **128/128** | 126/128 |
| r1 | 50/128 | 0/128 | 29/128 | 58/128 | **128/128** | 124/128 |
| r2 | 31/128 | 0/128 | 40/128 | 56/128 | **128/128** | 127/128 |
| r3 | 46/128 | 0/128 | 18/128 | 48/128 | **128/128** | 113/128 |
| **Total** | **156/512 (30,47%)** | **0/512 (0%)** | **122/512 (23,83%)** | **196/512 (38,28%)** | **512/512 (100%)** | **490/512 (95,70%)** |

Otros controles: A0 reactivo 67/512 (13,09%) fase antigua y 63/512 (12,30%) fase nueva; A3 sin opciones = A2 (0/512 antigua y 512/512 nueva); A4 privilegiado 512/512 en ambas fases.

**Comparación crucial:** en fase nueva, A3–A2 = **−22/512 = −4,30 puntos porcentuales**. En fase antigua A3–A1 = **−34/512 = −6,64 pp**. Ni la ventaja +10pp G1 ni su causalidad G4 pueden darse por confirmadas. No hay intervalos bootstrap preregistrados; sería impropio declarar PASS.

### El límite de costes también es un problema

La razón aproximada de bytes de JSON de **A3 incluyendo su A2** frente a **A2 solo** fue: r0=6,08×; r1=6,68×; r2=6,62×; r3=6,63×, muy por encima del techo preregistrado de 1,25× para el coste total. Esta medida solo cubre la representación serializada del modelo y las opciones; **no** incluye CPU/latencia real, episodio A1 ni compresión y no acredita un FAIL final de G5, pero desaconseja abrir DEV sin rediseño.

## Conclusiones y limitaciones

1. **A3 sí almacena y utiliza políticas**, pero no obtiene ventaja incremental frente a un planificador que atribuye efectos temporales en las reglas nuevas. Sus opciones incluso le hacen fallar en 22 casos donde A2 resuelve todos.
2. A2 conoce la estructura pública de los recursos y utiliza una **simulación aproximada de hasta 12 pasos** con modelo de acciones aprendido. Es un baseline guiado por el esquema; su 100% no demuestra planificación abierta, pero sí invalida interpretar la ventaja anterior de A3 sobre A2-v2 como evidencia suficiente.
3. A2 obtiene 0/512 al reponer las reglas antiguas porque su estimación pondera evidencia reciente del régimen nuevo; A3 todavía reproduce algunas opciones viejas. Ninguno ha demostrado detectar autónomamente un cambio posterior de forma robusta.
4. TRAIN-heldout comparte familia, metas y cuatro modos con TRAIN; los registros exponen `world_seed` y fase SOLO en artefactos/auditoría, nunca al actor. Los cuatro sujetos no tienen herramientas reales, no usan LLM ni pesos aprendidos, y no han sido examinados por un auditor científico independiente.
5. Los resultados muestran que mejorar los controles y neutralizar la exploración eran condiciones necesarias para una interpretación válida. No revelan inteligencia general, ni políticas nuevas en dominios abiertos.

## Decisión

**NO-GO para DEV y CONFIRM de esta implementación.** Mantener las semillas 61_200_000+, 61_300_000+ y 61_400_000+ completamente reservadas. No modificar los gates G0–G5 ni forzar nuevas ejecuciones sobre CONFIRM. La siguiente hipótesis diferenciada, si se sigue investigando, debería ser un meta-controlador que **aprenda cuándo utilizar o abandonar las opciones** según el rendimiento esperado frente a A2, con límites de almacenamiento y gasto de cómputo. Proponer una enmienda o experimento separado antes de nuevos datos, sin reinterpretar la prueba actual como victoria de A3.

No se ha tocado el producto `src/infinito3`, el servidor MCP, memoria personal, secretos ni servicios externos. No fusionar automáticamente con master.
