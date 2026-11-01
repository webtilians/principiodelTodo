# INFINITO F9-V — Resultado real de transferencia EcoDispatch → valor causal de lectura F4

**2026-11-01 — confirmación prospectiva completada. Resultado del gate global: NEGATIVO. Hay evidencia limitada de una capacidad causal útil, pero no robusta en 4 réplicas.**

## Auditoría y procedencia

- [Protocolo antes del código y resultados](./INFINITO_F9V_LEXICOGRAPHIC_CAUSAL_F4_PROTOCOL.md), commit preregistrado **e1347f4603f781d7730222b13ae5e9bc09591291**.
- [Manifest de pesos y reportes congelados ANTES de CONFIRM](./INFINITO_F9V_CONFIRMATION_LOCK.json), commit **9466e30e8a7a4e2311a6b8d76e68980f989f2451**. Cuatro checkpoints F9-P, doce cabezas entrenadas, cuatro reportes y código/protocolo verificados mediante SHA-256.
- [TRAIN/DEV GitHub Actions #41345497955](https://github.com/webtilians/principiodelTodo/actions/runs/41345497955) **SUCCESS** en cuatro réplicas más agregado, TRAIN 8_100_000+, DEV 8_200_000+.
- [**Única confirmación independiente GitHub Actions #41346198163**](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163): **SUCCESS** técnico: gate global de 4 fuentes congeladas, cuatro trabajos confirmatorios, agregado. **SUCCESS del workflow no significa SUCCESS de la hipótesis**.
- [Artifact agregado confirmatorio](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163/artifacts/13098949078). Artifacts originales por semilla: [0](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163/artifacts/13098549636), [1](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163/artifacts/13099546327), [2](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163/artifacts/13099825207), [3](https://github.com/webtilians/principiodelTodo/actions/runs/41346198163/artifacts/13099785761). Todos incluyen cada estado, sus dos intervenciones, la decisión de cada cabeza y el valor causal D.
- **CONFIRM 8_300_000+ CONSUMIDO irreversiblemente**. F9-T 6_300_000+, F9-U 7_300_000+ e histórico 900_000+ **sin abrir**. No reajustar y re-confirmar sobre estos casos.

## Pregunta investigada y significado

EcoDispatch prioriza severidad cubierta → número de casos cubiertos → mínimo coste, y compara coste inmediato contra exposición futura. **No se transplantan los algoritmos exactos de k-server/matching ni sus garantías teóricas**. F9-V usa el PRINCIPIO lexicográfico en un dominio distinto: **primero escoger la primera salida correcta, luego evitar una lectura F4 innecesaria**.

El comparador experimental reconstruye dos continuaciones del mismo estado de la política recurrente **H/C 64+64**, misma observación pública 148, mismo entorno y mismo F4 guardado: READ en el primer cruce vs SKIP. En TODAS las observaciones posteriores, F4 está anulado. El actor continúa libremente hasta su primera elección o fin. `D=correcto_READ−correcto_SKIP` toma valores −1,0,+1; escoger READ solo si D=+1 es el **oráculo**, que ve ambos futuros y **NO** es desplegable.

La cabeza `lex_hc`, MLP 276→32→1, **solo ve los 148 públicos y H/C previos**, sin F4 actual, etiquetas docentes, objetivo, futuro, régimen o `D` en inferencia. Entrena TRAIN con etiquetas causales del oráculo; en confirmación pesos congelados. Control `lex_public` de la misma arquitectura, pero H/C puestos a cero en la cabeza. Se entrena mediante BCE SUPERVISADA, no descubrimiento autónomo de políticas ni RL. Actor F9-P y escritor F4 no se modifican.

**Límite muy importante:** el prefijo hasta el cruce está guiado por profesor para igualar la geometría. Esto es autonomía de la elección desde el fork, **NO navegación autónoma end-to-end**.

## Resultados de DEV, preregistrados antes de confirmación

4 réplicas × 16 mundos DEV por réplica × 2 cues invertidos × 2 regímenes = **256 estados**.

| Lector | Correcto primer brazo /256 | READ /256 |
|---|---:|---:|
| always | 182 | 256 |
| never | 168 | 0 |
| lex_hc | **183** | **15** |
| lex_public | 168 | 0 |
| scalar_hc (coste excesivo, control negativo) | 168 | 0 |
| oracle (no desplegable) | 193 | 25 |

Efectos: `D=-1:11`, `D=0:220`, `D=+1:25`. Gate DEV **PASS**, por lo que se congelaron todos los pesos y abrió UNA única confirmación.

## Resultados de confirmación INDEPENDIENTE (PRIMARIO)

4 réplicas × 40 mundos CONFIRM por réplica × 2 cues invertidos × 2 regímenes = **640 estados**.

| Lector | Primer brazo correcto /640 | Lecturas F4 /640 | Pares dos pistas correctas, F4 normal /160 | Pares dos pistas correctas, apagón /160 |
|---|---:|---:|---:|---:|
| `always` | **480 (75,00%)** | 640 | 156 | 17 |
| `never` | 429 (67,03%) | 0 | 109 | 0 |
| **`lex_hc`** | **467 (72,97%)** | **38 (5,94%)** | **147** | **0** |
| `lex_public` | 429 (67,03%) | 0 | 109 | 0 |
| `scalar_hc` | 429 (67,03%) | 0 | 109 | 0 |
| `oracle` (no desplegable) | 497 (77,66%) | 68 | 160 | 17 |

Efectos reales sobre las 640 situaciones: **68** favorecen READ (D=+1), **17** lo perjudican (D=−1), y **555** no cambian la primera elección (D=0). Para el oráculo diagnóstico, solo 68/640 lecturas son necesarias y nunca elige READ cuando no mejora. En particular, leer indiscriminadamente puede empeorar algunas primeras elecciones.

La política `lex_hc` hace **38 lecturas, todas 38 con D=+1; cero falsos positivos** en esta partición, y evita las 17 lecturas perjudiciales. Sin embargo **omite 30 de las 68 lecturas que habrían rescatado la elección** (recall de D positivo 38/68 = 55,88%). Frente a `never`, suma 38 aciertos; frente a `always`, pierde 13 aciertos (2,03 pp), aunque usa 94,06% menos lecturas. Frente a `lex_public`, suma 38 aciertos (5,94 pp); ese contraste entre cabezas NO equivale por sí solo a ablar H/C de la misma cabeza con pesos idénticos.

### Heterogeneidad CRÍTICA por réplica

| Checkpoint F9-P congelado | `lex_hc` correctos /160 | `always` correctos /160 | `lex_hc` READ /160 | Oráculo READ necesarias |
|---|---:|---:|---:|---:|
| 0 | 108 | 116 | 18 | 30 |
| 1 | 120 | 131 | **0** | 11 |
| 2 | 119 | 113 | **0** | 7 |
| 3 | 120 | 120 | 20 | 20 |

Diferencias `lex_hc - always`: **−8, −11, +6, 0**. Solo **2/4** réplicas cumplen la condición preregistrada de no perder más de dos decisiones vs always (se requerían ≥3/4).

### Dependencia del régimen e identidad de pista

- Historia F4 **normal** hasta el fork: `lex_hc` 307/320 decisiones y 147/160 pares de dos pistas correctas, con 38 READ; `always` 316/320, 156/160 pares; `never` 269/320 y 109/160 pares. Oráculo 320/320 con 51 READ.
- Historia con **apagón F4 prolongado postcue**: `lex_hc` **160/320**, **0/160 pares** y **cero READ**; `always` 164/320 y 17/160 pares; el oráculo 177/320 y 17/160 pares con 17 READ. El lector NO generalizó las oportunidades de rescate del régimen blackout.
- En la confirmación, las **38 lecturas de `lex_hc`** provienen únicamente de *cue key* en el régimen normal, en réplicas 0 (18 lecturas) y 3 (20). La política ignora casos con efecto positivo para cue ball y en régimen blackout. Esto puede reflejar aprendizaje de contingencias particulares de pista/actor, **NO una representación general de incertidumbre o necesidad de memoria**.
- El contraste de cue invertida y las dos intervenciones desde el mismo estado permiten aislar el **efecto local de una lectura**; no prueban que el agente sepa inventar estrategias ni que el valor causal permanezca estable a largo plazo.

## Gate original de confirmación: FAIL

| Criterio fijado antes de confirmar | Resultado |
|---|---|
| Existen efectos positivos identificables | PASS, D=+1 en 68 casos |
| Conserva ≥90 % de los aciertos de always | PASS, 467/480=97,29 % |
| Ahorra ≥20 % de las lecturas | PASS, 94,06 % ahorro |
| **≥3/4 réplicas a no más de 2 aciertos por debajo de always** | **FAIL: 2/4** |
| Supera en ≥3 puntos porcentuales al control público | PASS, +5,94 pp |
| **Resultado combinado** | **FAIL (preregistrado)** |

No aplicar a posteriori una regla más favorable (p.ej. seleccionar solo las réplicas 0 y 3, o escoger solo historia F4 normal) y llamarla confirmación. Los buenos promedios globales y la precisión de las 38 consultas son hallazgos **acotados y secundarios**, no superan el fallo de robustez.

## Implicaciones y siguiente hipótesis separada

Existe una señal replicable de que **la cabeza neuronal conectada a H/C puede aprender a predecir algunas decisiones de lectura que aportan valor causal**, al menos en dos réplicas y en la primera bifurcación después de un historial F4 normal. Esto va más allá de REINFORCE global de F9-T/U porque un contrafactual evalúa READ y SKIP desde el mismo estado.

**Pero la hipótesis fuerte de capacidad robusta y transferible NO se confirmó.** Si se sigue, diseñar un experimento NUEVO sobre ventanas temporales de 1–8 lecturas, pares de historial natural/blackout, entrenamiento que cubra decisiones útiles en cada régimen y selección lexicográfica bajo coste, con nuevos checkpoints y nuevas semillas preregistradas. Medir si la generalización depende realmente de H/C al ablar su entrada en la misma cabeza congelada, y comparar contra una regla simple cue→READ. Nunca reutilizar 8300000+ para decidir parámetros y después reconfirmar en él.

Este proyecto conserva evidencia negativa y control de costes tal como EcoDispatch aprendió de su validación histórica: mayor complejidad no garantiza capacidad útil.
