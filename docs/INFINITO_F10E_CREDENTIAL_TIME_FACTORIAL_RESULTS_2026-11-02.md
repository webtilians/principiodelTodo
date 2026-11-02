# INFINITO F10-E — Diagnóstico causal factorial de firma y precedencia

**2026-11-02. Resultado DEV final, no confirmación: `F10E_FACTORIAL_DIAGNOSTIC_FAIL`.** Protocolo [prerregistrado antes de implementar generador/ejecutor y abrir mundos](./INFINITO_F10E_CREDENTIAL_TIME_FACTORIAL_PREREG.md), commit **`e7667782119c5034c565339cc9c839917bbe5cac`**; versión original preliminar `40c0e1bbf38f74294b6e7ac85ec84f821c42a097` conservada en historia, solo corregía cómputo dimensional antes de código. [GitHub Actions #41456894131](https://github.com/webtilians/principiodelTodo/actions/runs/41456894131) **SUCCESS técnico 4/4 réplicas + agregado científico independiente**; artefactos completos `f10e-actor-{0,1,2,3}` con cada historia contrafactual y probabilidades, y `f10e-independent-preregistered-factorial-gate`.

## Diseño que controla qué cambia realmente

**432 mundos base completamente nuevos:** `16_200_000+10_000*index+j`, index0..3, j0..107, con 108 por réplica; **cuatro contrafactuales cada uno = 1728 consultas dependientes**, no 1728 mundos independientes. Todos los checkpoints originales F10-C (relacional, unario, GRU-only, Actions #41447171697) y F10-B (READ y GRU-only, Actions #41443064754) **congelados y verificados por SHA de artifacts originales**. No entrenamiento, no actualización de pesos, ni ajuste de hiperparámetros ni cabeza READ. Ningún mundo anterior reutilizado; `16_300_000+` no abierto.

Cada mundo genera 24 claves, 60 informes públicos, una consulta final única sin pista temprana, y tres cantidades de informes de la clave `k={3,5,8}` × tres pares de posiciones `early=(34,35)`, `middle=(39,49)`, `late=(48,59)` con **12 mundos por cada una de las 9 celdas de rejilla por réplica**. Fuente y bits iniciales aleatorizados; fuentes asociadas a los informes y preservadas bajo intercambio; credencial de firma es token público sintético. Valor correcto = último informe firmado reconstruido **exclusivamente de eventos públicos**, sin etiquetas ocultas.

Variantes `U_before, U_after, S_before, S_after`: mismo informe contradictorio X de valor `1−b` cambia exclusivamente el **bit de firma** U/S y su orden respecto a confirmación firmada `C=S(b)`. `U_before` (X no firmada, antes de C), `U_after` (X no firmada, después de C) y `S_before` (X firmada, antes de C) requieren respuesta b; **solo `S_after` (X firmada y posterior) exige cambiar a 1−b**. Tests verificaron una sola diferencia de bit para cambiar firma con orden fijo y cambios restringidos a las dos posiciones del par al intercambiar el orden. Historial público, y por tanto estado GRU, **sí cambian entre variantes**; READ/SKIP en una misma variante comparten historial GRU exacto y modifican solo F4.

## Resultado principal — el punto débil es decidir la precedencia entre evidencias firmadas contradictorias

| Contrafactual | Respuesta relacional READ (432 casos) | Porcentaje | Control unario READ |
|---|---:|---:|---:|
| `U_before` — no acreditado, anterior | **431/432** | 99,8% | 376/432 |
| `U_after` — no acreditado, posterior | **423/432** | 97,9% | 349/432 |
| `S_before` — acreditado, anterior | **306/432** | **70,8%** | 227/432 |
| **`S_after` — acreditado y posterior** | **176/432** | **40,7%** | **207/432** |
| **Total** | **1336/1728** | **77,3%** | **1159/1728 = 67,1%** |

**Interpretación limitada a conducta:** el actor desconoce/ignora pocas veces los informes SIN firma (los controles U_before/U_after son casi perfectos). Pero falla al elegir qué informe prevalece cuando ambos están firmados y contradicen su valor. En particular, cuando una nueva actualización autorizada aparece después, a menudo conserva el valor anterior. Esto apoya un problema **de precedencia/integación de procedencia y temporalidad**, más que una incapacidad general de observar el bit de firma. **No demuestra un mecanismo neuronal ni qué neuronas hacen el cálculo**.

El relacional supera globalmente al unario congelado en **177/1728 = +10,2 pp**, pero en la condición crucial `S_after` **el unario logra 207/432 frente a 176/432 del relacional**. No afirmar una superioridad relacional uniforme. Además, ambos pueden aprovechar plantillas o sesgos de diseño (matching de clave estructural y topk5); F10-E no prueba comprensión general de verdad/autenticidad.

## Contrastes preregistrados del factorial emparejado

Los cuatro contrastes exigen **predicciones binarias conjuntas correctas dentro del mismo mundo**, no solo dos aciertos medios de grupos distintos:

| Contraste conductual | Mundos correctos /432 | Tasa |
|---|---:|---:|
| **Credencial decisiva**: `U_after=b` y `S_after=1−b` | **167/432** | **38,7%** |
| **Orden decisivo**: `S_before=b` y `S_after=1−b` | **79/432** | **18,3%** |
| **Credencial no decisiva**: `U_before=S_before=b` | 306/432 | 70,8% |
| **Orden no decisivo**: `U_before=U_after=b` | **423/432** | **97,9%** |
| **Acertar las cuatro variantes del mismo mundo** | **78/432** | **18,1%** |

Por réplica (cada una 108 mundos), contraste de **orden decisivo** `1, 78, 0, 0`; contraste de **credencial decisiva** `28, 78, 31, 30`; mundos con **las cuatro respuestas** `0, 78, 0, 0`. La réplica 1 tiene competencia parcial clara, pero no se puede escoger solo esa réplica a posteriori. Las réplicas 0/2/3 fallan casi totalmente el contraste de precedencia efectiva.

La estrategia trivial «responde siempre al primer informe firmado» obtiene 3/4 condiciones correctas: **1296/1728 = 75%**. Igualmente «responde al último informe» acierta 75% en este factorial. El actor relacional logra solo **+2,3 puntos porcentuales agregados frente a estos controles simplistas**, aunque supera a ambos en ciertos pares y falla en otros: la exactitud general del 77,3% **no sería evidencia suficiente de resolver la regla**. Por definición de tarea, «último firmado público» obtiene **1728/1728 (100%)**, como techo algorítmico exacto derivado de información disponible.

## Separar número de registros y distancia temporal

| Registros de clave | Relacional /576 |
|---|---:|
| 3 | **457/576 = 79,3%** |
| 5 | **463/576 = 80,4%** |
| 8 | **416/576 = 72,2%** |

| Ubicación del par de evidencias | Relacional /576 |
|---|---:|
| Inicio del tramo final `early` | **449/576 = 78,0%** |
| Zona media `middle` | **454/576 = 78,8%** |
| Final `late` | **433/576 = 75,2%** |

Aumentar registros a 8 perjudica en promedio, especialmente a réplica1 (102/144), lo cual es compatible con límites del `topk5`; **sin una intervención separada del selector topk no se puede afirmar que éste sea la causa exclusiva**. La distancia temporal apenas altera el resultado medio frente a la gran caída cuando ambas evidencias son firmadas.

## Causalidad de F4 en cada variante

| Modelo en los mismos 1728 casos | Correctos |
|---|---:|
| Relacional F10-C READ auténtico | **1336/1728 = 77,3%** |
| El mismo actor relacional SKIP (F4=0) | **884/1728 = 51,2%** |
| Control unario congelado F10-C READ | **1159/1728 = 67,1%** |
| GRU-only congelado F10-C | **908/1728 = 52,5%** |
| READ F10-B congelado | **1223/1728 = 70,8%** |
| Primer informe firmado (heurística trivial) | **1296/1728 = 75,0%** |
| Último informe sin validar | **1296/1728 = 75,0%** |
| Último firmado público (regla exacta) | **1728/1728 = 100%** |

La **ventaja causal READ−SKIP es +452/1728 = +26,2 puntos porcentuales**, usando **el mismo actor y la misma historia GRU dentro de cada contrafactual**. Distribución: **D+=659, D0=862, D−=207**: F4 perjudica en una fracción relevante y no debe omitirse. Los cuatro IC95% bootstrap por **108 mundos base** para READ−SKIP tienen límite inferior positivo:
- réplica 0: **+22,9 pp**, IC [18,3;27,8]
- réplica 1: **+38,4 pp**, IC [32,9;44,4]
- réplica 2: **+21,8 pp**, IC [17,1;26,4]
- réplica 3: **+21,5 pp**, IC [16,9;26,2]

El efecto global de F4 sigue siendo causalmente utilizable aunque READ **no** resuelva correctamente los estados relevantes. Aquí **no** se ha aprendido decidir cuándo recuperar.

## Resultado formal del prerregistro

Agregado independiente ejecutado **sin Torch/NumPy** devuelve `F10E_FACTORIAL_DIAGNOSTIC_FAIL`:

| Gate | Umbral prospectivo | Resultado |
|---|---|---|
| **G1** cuatro respuestas exactas, precisión por variante/cardinalidad/posición | >=3/4 réplicas | **FAIL 0/4** |
| **G2** cuatro contrastes causal-conductuales de firma/orden | >=3/4 réplicas | **FAIL 0/4** |
| **G3** READ−SKIP >=+0,10, IC inferior >0, READ>GRU-only +0,05 | >=3/4 réplicas | **PASS 4/4** |

Registros individuales originales y auditores guardados en el run #41456894131, PR #47 borrador. **No se retocan pesos, generador, umbrales ni se abren las semillas reservadas `16_300_000+`**.

## Qué se ha aprendido y próximo experimento posible (no ejecutado)

- Se descarta «la red no ve nunca la credencial»: casi siempre trata correctamente informes sin acreditar, aun si aparecen más tarde.
- Se descarta «la red aplica sistemáticamente la regla exacta de última versión acreditada»: cuando dos versiones firmadas se contradicen falla en gran parte de las réplicas.
- El problema no se identifica únicamente como «recencia»: falla al **combinar prioridad de firma con orden entre versiones acreditadas**. Hay heterogeneidad fuerte entre réplicas y capacidad limitada ante k8.
- La comparación de una red unaria y relacional congeladas NO aísla aisladamente el componente pairwise (difieren pesos, entrenamiento y tamaño); que una sea mejor globalmente no justifica un cambio de arquitectura inmediato.

**Siguiente decisión propuesta**: congelar F10-E como resultado negativo informativo. Antes de otro aprendizaje, examinar *sin usar DEV para ajustes* diagnósticos mecanicistas de qué slots entran en el topk5 y su contribución (atención por informe), y formular un NUEVO experimento prospectivo con informes de diferentes prioridades y test de **conmutación de respuesta** condicionado a última actualización firmada, comparando arquitecturas con la misma capacidad. No abrir F10-E confirm, no entrenar READ aún. Limitaciones: firmas públicas sintéticas, task construida, una sola consulta binaria al final, sin verificación criptográfica real, sin continuidad longitud variable del entorno ni F4 MiniGrid/LLM/RAG integrados.
