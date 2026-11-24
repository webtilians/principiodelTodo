# INFINITO M1: estructura de memoria descubierta y revisada, resultados de CONFIRM

**Fecha del informe:** 23 de noviembre de 2026
**Ejecución:** 20 de noviembre de 2026; fase 1 de 17:19 a 17:36 y fase 2 de 17:36 a 23:24, en una sola pasada y sin cortes
**Protocolo:** [INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md](INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md) (PR #177)
**Congelación:** [INFINITO_M1_FREEZE.json](INFINITO_M1_FREEZE.json), commit `87292ea`
**Harness:** `observatory/m1_fractal_memory.py`
**Rama:** `experiment/m1-fractal-memory-run-20261120`

## Veredicto

**No pasa nada.** H1 no pasa, H2 estático no pasa, H2 adaptativo no pasa, y el criterio fuerte de la adenda
tampoco se cumple.

| | Veredicto |
|---|---|
| H1: jerarquía temporal fija (A1) | no pasa |
| H2 estático: estructura descubierta y congelada (A2) | no pasa |
| H2 adaptativo: estructura revisada tras cada sesión (A3) | no pasa |
| Criterio fuerte | no se cumple |

Además, dos puertas duras no se superan:

- **Olvido:** falla tal como está escrita. A1 revela 8 respuestas olvidadas y A0 revela 7.
- **Juez:** la fase 2 queda **INCONCLUSIVE**. El mantenedor no etiquetó la auditoría, y el contraste
  suplementario de Claude coincide con el juez en 25 de 30 juicios, por debajo de 27.

El veredicto **no depende** de esas dos puertas. Cada hipótesis falla también en los criterios de cobertura de la
fase 1, que no usa ningún LLM:

- **H1:** A1 no supera a A0 en cobertura relacional (−1,47 puntos; hacía falta +5 con IC > 0).
- **H2 estático:** A2 queda 12,23 puntos por debajo de A0.
- **H2 adaptativo:** A3 mejora mucho a A2 tras el corte, pero pierde entre 3,2 y 4,8 puntos de cobertura en tres
  tipos de pregunta anteriores al corte. El límite era 2.

Según la sección «What each result changes» del protocolo, *nothing passes*:

- la recuperación plana se queda;
- esta línea se cierra para recuperación de este tipo;
- AtlasKV no se abre a partir de este resultado;
- el siguiente trabajo de memoria es el punto 2 de la hoja de ruta: la comparación pública con otros sistemas,
  que parte de las cifras de A0 en LoCoMo medidas aquí.

Como falla una puerta dura, no se integra nada, sea cual sea la precisión.

## Qué se ejecutó

| | |
|---|---|
| Datos | LoCoMo `locomo10.json`, sha256 `79fa87e9…` (igual que en la congelación) |
| CONFIRM | conv-43, conv-42, conv-50, conv-30, conv-49, conv-41, conv-48, conv-44 |
| Presupuesto decisivo | 1.024 tokens |
| Lector | `qwen3.8-motor`, digest `7991c412…` (igual que en la congelación) |
| Juez | `llama3.1:8b`, digest `46e0c10c…` (igual que en la congelación) |
| Fase 1 | 477 preguntas relacionales, 687 single-hop y 68 open-domain, con 3 réplicas para A2, A3 y controles |
| Fase 2 | 310 preguntas × 4 brazos (1.240 filas) y 40 preguntas de olvido × 4 brazos (160 filas) |

Integridad de la fase 2:

- ninguna fila duplicada;
- ningún veredicto ilegible;
- ninguna llamada sin respuesta;
- el fichero de progreso tiene 2.544 líneas: 1.400 respuestas del lector y 1.144 juicios.

Pérdidas al escribir en memoria, iguales para todos los brazos:

- 90 de 4.774 turnos rechazados por el filtro que no guarda preguntas ni acuses de recibo. 58 de ellos son de
  conv-48.
- 5 preguntas quedaron sin su evidencia en el almacén: conv-42 una temporal y una open-domain; conv-50, conv-49 y
  conv-44 una cada una.
- 4 preguntas se excluyeron porque citan un identificador que no es un turno.
- Ningún turno llega al límite de 2.000 caracteres.

## Puertas duras

**Olvido: falla tal como está escrita.**

- Las cuatro comprobaciones mecánicas pasan en las 8 conversaciones.
- La comprobación de historial, añadida tras DEV y fuera de la puerta, también pasa: ninguna versión guardada de A3
  ni su registro de revisiones conserva turnos olvidados.
- Ningún turno olvidado apareció en un paquete.

Respuestas olvidadas reveladas, según el juez, de 40 por brazo:

| | conv-43 | conv-42 | conv-50 | conv-30 | conv-49 | conv-41 | conv-48 | conv-44 | Total |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 1 | 2 | 1 | 1 | 2 | 0 | 0 | 0 | **7** |
| A1 | 1 | 2 | 1 | 1 | 2 | 1 | 0 | 0 | **8** |
| A2 | 1 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | **6** |
| A3 | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | **6** |

La regla dice que ningún brazo revela más que A0, y A1 revela una más. Decide la pregunta conv-41/83: «What areas
is John particularly interested in for policymaking?», con referencia «education and infrastructure».

- Se olvidó su evidencia, D9:8, pero la misma respuesta sigue en otros turnos, entre ellos D1:8, D4:16, D12:5 y
  D22:1.
- A0 contestó «Not in memory»; A1, A2 y A3 la encontraron en esos turnos.
- No es una fuga del mecanismo de olvido. Es información superviviente que A1 recupera mejor que A0, que es
  justo lo que el protocolo anticipaba («It may survive in other turns»). Aun así, la regla no tenía margen y no
  se cambia después de ver el dato.

**Aislamiento por proyecto:** pasa. En los cuatro pares (43–42, 50–30, 49–41, 48–44) no hay fugas.

**Integridad:** pasa.

- CONFIRM empezó a las 17:19, después del commit de congelación (`87292ea`, 16:52).
- Los modelos coinciden con sus digests.
- La traza tiene un registro por pregunta y brazo al presupuesto decisivo: 22.274 líneas, sha256
  `d9e78fea…`, el mismo que guarda la fase 1.
- El rollback pasa en todas las réplicas.

**Auditoría del juez: incompleta. La fase 2 queda INCONCLUSIVE.** El protocolo pide que el mantenedor etiquete a
ciegas 30 juicios, y no lo hizo. A petición suya, Claude (Opus 5.5) etiquetó la hoja ciega, sin ver el brazo ni el
veredicto del juez. Las etiquetas se guardaron antes de calcular la concordancia
(`confirm_audit_labelled_by_claude.json`, sha256 `57e25338…`). Es un tercer modelo, distinto del lector y del
juez, pero no es la auditoría humana que exige la puerta.

Concordancia: **25 de 30**, por debajo de 27 en cualquier caso.

| Ítem | Pregunta | Tipo | Brazo | Claude | Juez |
|---|---|---|---|---|---|
| 4 | conv-48/37 | temporal | A3 | CORRECT | WRONG |
| 10 | conv-42/171 | single-hop | A0 | WRONG | CORRECT |
| 14 | conv-49/13 | temporal | A1 | CORRECT | WRONG |
| 18 | conv-48/66 | temporal | A3 | CORRECT | WRONG |
| 22 | conv-42/44 | temporal | A0 | CORRECT | WRONG |

Cuatro de las cinco discrepancias son respuestas temporales con fecha relativa, como «Yesterday, relative to 30
August, 2023» frente a «29 August, 2023». El juez las rechazó, a pesar de que su instrucción admite otro formato de
fecha. Los ítems 14 y 22 se marcaron como dudosos antes de calcular la concordancia: la respuesta repite la
evidencia literal («last week», dicho en una fecha) y la referencia de LoCoMo la convierte en un mes o una semana.
Leídos como WRONG, la concordancia sería 27 de 30. Se informa así y no se reetiqueta.

En las preguntas temporales, el juez da por correctas más o menos la mitad de las respuestas con fecha relativa en
todos los brazos: A0 8 de 16, A1 6 de 13, A2 5 de 12 y A3 9 de 17. No parece un sesgo a favor de un brazo, pero no
se ha contrastado.

## Cobertura (fase 1, sin LLM)

Porcentaje de preguntas cuyo paquete de 1.024 tokens contiene toda su evidencia. Entre paréntesis va la diferencia
con A0 en puntos, con intervalo bootstrap del 95 % sobre las 8 conversaciones. Los brazos con réplicas se promedian
sobre las 3.

| Brazo | Relacional (n=477) | Single-hop (n=687) | Open-domain (n=68) |
|---|---|---|---|
| A0 RAG plano | 38,6 | 59,8 | 27,9 |
| A1 árbol temporal | 37,1 (−1,47 [−5,44; 2,15]) | 61,0 (+1,16 [0,00; 2,90]) | 23,5 (−4,41 [−11,11; 0,00]) |
| A2 descubierta, congelada | 26,3 (−12,23 [−16,04; −8,49]) | 42,0 (−17,81 [−20,68; −14,05]) | 21,6 (−6,37 [−13,01; 0,97]) |
| A3 revisada | 33,8 (−4,82 [−8,08; −2,01]) | 57,1 (−2,72 [−4,71; −0,27]) | 26,5 (−1,47 [−7,81; 3,21]) |
| A2 barajada | 22,5 (−16,07 [−21,29; −11,26]) | 33,5 (−26,35 [−29,23; −21,90]) | 14,2 (−13,73 [−23,53; −3,24]) |
| A2 reconstruida | 32,5 (−6,08 [−9,52; −2,92]) | 54,9 (−4,95 [−7,08; −1,96]) | 22,5 (−5,39 [−12,50; 0,52]) |

Otras comparaciones de cobertura relacional:

- **A2 frente a A1:** −10,76 [−15,13; −6,94].
- **A2 frente a A2 barajada:** +3,84 [0,88; 6,79]. Lo descubierto agrupa mejor que el azar, pero no llega al
  umbral de +5.
- **A2 frente a A0, por réplica:** −11,74, −12,58 y −12,37. Las tres son negativas.
- **A3 frente a A2, tras el corte:** A2 cubre el 11,7 % y A3 18,50 puntos más [11,94; 25,12]. Pasa el criterio 1
  de H2 adaptativo.
- **A3 frente a A2, antes del corte:**
  - multi-hop −1,77 [−4,70; 0,29];
  - temporal −3,21 [−6,43; −0,23];
  - single-hop −4,81 [−7,64; −1,75];
  - open-domain −3,70 [−8,57; 0,00].

  Tres tipos superan la pérdida permitida de 2 puntos, así que el criterio 2 falla.
- **A2 reconstruida frente a A2 congelada:** +6,15 [3,48; 9,19].
- **A3 frente a A1, fuera de las puertas:** −3,35 [−4,60; −2,36].

Cobertura relacional por conversación:

| Brazo | conv-43 | conv-42 | conv-50 | conv-30 | conv-49 | conv-41 | conv-48 | conv-44 |
|---|---|---|---|---|---|---|---|---|
| A0 | 39,3 | 38,2 | 31,7 | 75,7 | 25,7 | 43,1 | 52,4 | 16,7 |
| A1 | 35,7 | 38,2 | 30,2 | 78,4 | 31,4 | 36,2 | 41,3 | 20,4 |
| A2 | 29,2 | 21,5 | 22,8 | 51,4 | 18,6 | 26,4 | 38,1 | 13,6 |
| A3 | 32,7 | 36,4 | 27,0 | 71,2 | 26,2 | 33,9 | 39,7 | 16,0 |

## Precisión (fase 2, INCONCLUSIVE)

Estas cifras son descriptivas, porque la auditoría del juez no pasa. La diferencia con A0 va en puntos, con
intervalo del 95 %.

| Brazo | Relacional (n=128) | Local (n=118) | Adversarial (n=64) |
|---|---|---|---|
| A0 | 38,3 | 38,1 | 90,6 |
| A1 | 37,5 (−0,78 [−7,81; 5,47]) | 37,3 (−0,85 [−5,56; 3,97]) | 93,8 (+3,12 [0,00; 6,25]) |
| A2 | 31,2 (−7,03 [−17,19; 2,34]) | 28,8 (−9,32 [−14,29; −4,03]) | 92,2 (+1,56 [−3,12; 6,25]) |
| A3 | 36,7 (−1,56 [−9,38; 3,91]) | 39,8 (+1,69 [−5,56; 9,91]) | 96,9 (+6,25 [1,56; 12,50]) |

A3 frente a A2:

- relacional tras el corte: +11,94 [2,94; 22,22], n=67;
- local: +11,02 [3,39; 20,54].

## Criterios, uno a uno

**H1 (A1 frente a A0):**

- falla la cobertura relacional (−1,47, hacía falta ≥ +5 con IC > 0);
- falla la cobertura open-domain (−4,41, el límite era −2);
- falla la precisión relacional (−0,78);
- cumplen la cobertura single-hop, la precisión local y la adversarial.

**H2 estático (A2):** falla todo salvo la precisión adversarial.

- cobertura relacional frente a A0, frente a A1, frente a la barajada y por réplica;
- coberturas locales;
- precisión relacional frente a A0 y frente a A1;
- precisión local.

**H2 adaptativo (A3 frente a A2):**

- cumplen la cobertura relacional tras el corte, la precisión relacional tras el corte, la precisión local y la
  adversarial frente a A0. Las tres de precisión son INCONCLUSIVE.
- falla la cobertura antes del corte en temporal, single-hop y open-domain.

## Coste de búsqueda (fuera de las puertas)

Medianas por paquete al presupuesto decisivo, medidas en el portátil de la ejecución:

| Brazo | ms | Grupos visitados | Similitudes leídas |
|---|---|---|---|
| A0 | 0,377 | 0 | 0 |
| A1 | 0,587 | 7 | 891 |
| A2 | 0,680 | 11 | 1.333 |
| A3 | 0,722 | 8 | 1.556 |
| A2 barajada | 0,701 | 13 | 1.458 |
| A2 reconstruida | 0,805 | 12 | 2.047 |

Construir A2 y A3 en flujo cuesta entre 0,9 y 1,9 s por conversación, y reconstruir A2 entre 0,8 y 1,6 s.

Este coste solo cubre el recorrido. No incluye embeddings, el lector ni el juez. A esta escala, el coste de
recorrer los grupos no decide nada; lo que decide es la cobertura.

## Qué deja aprendido

1. **El recorrido por grupos es lo que pierde cobertura.** A3 y A2 reconstruida recuperan buena parte de lo que
   pierde A2 congelada: +18,50 tras el corte y +6,15 en total. Aun así, ninguna estructura descubierta alcanza a
   A0. A1 queda a la par de A0 en relacional (−1,47, con un intervalo que incluye el 0), sin ganarle.
2. **La estructura descubierta lleva algo de señal** (+3,84 frente a la barajada), pero no la suficiente para
   compensar lo que cuesta ir primero a los grupos.

Para el próximo protocolo quedan cuatro lecciones. No cambian este veredicto y no abren ningún experimento:

- **La regla de revelación debería descontar la información superviviente.** Por ejemplo, contando una revelación
  solo cuando ningún turno vivo contenga la respuesta.
- **La concordancia humana con el juez se mide antes de abrir CONFIRM**, con una regla explícita para fechas
  relativas, y la etiqueta el mantenedor.
- **LoCoMo es público,** así que el lector pudo verlo durante su entrenamiento. Eso puede afectar a la precisión,
  no a la cobertura. Cualquier afirmación futura de mejora necesita un banco nuevo y sellado.
- **M1 solo prueba agrupación con Louvain y recuperación que empieza por los grupos.** No refuta la memoria
  emergente en general ni prueba generación de conceptos, aprendizaje de políticas o transferencia.

## Límites

- 8 conversaciones, un lector, un juez y un presupuesto decisivo.
- Las pérdidas al escribir en memoria limitan la cobertura de todos los brazos por igual.
- El juez es generoso con algunas respuestas parciales y estricto con fechas relativas.

## Ficheros

Los resultados están en `artifacts/m1/`, fuera de git como en ejecuciones anteriores. Sus hashes:

| Fichero | sha256 |
|---|---|
| `confirm_stage1.json` | `943307477bca48a32b9fb78c99ea5115fadbb1dc706e7bb9e5917e74b4f10ebc` |
| `confirm_stage1.traces.jsonl` | `d9e78feaec0869ea48c9e651fbcb254d206d441f720cb4876ebdcfd70a5889f8` |
| `confirm_stage2.json` | `3b210a245d4fa7466bae39c3c4f18fd87ca5cc7d36322f96cd040ca8503eebac` |
| `confirm_audit_sheet.json` | `1629187d9da5f2adaba36c3330b08a8e01ff46558f3737a11932f33ef25e65e0` |
| `confirm_audit_labelled_by_claude.json` | `57e253388abd8aaab918ff1a841543c81f09acc57d3b96a9cee67025b6bae379` |
| `confirm_analysis.json` (sin auditoría) | `ff0156e85cb0ca392edd151872d50cb8e7bb1e881232c8355d5d18087ec463c8` |
| `confirm_analysis_claude_audit.json` | `d1bf1c6e75c5c9c2f6c74ecbb8cec7b51db1d626c04571f1a93166084d403982` |

Para reproducir el análisis:

```
python -m observatory.m1_fractal_memory analyze --data <locomo10.json> --model <potion-multilingual-int8> \
  --split confirm --stage1-file artifacts/m1/confirm_stage1.json \
  --stage2-file artifacts/m1/confirm_stage2.json --out artifacts/m1/confirm_analysis.json
```
