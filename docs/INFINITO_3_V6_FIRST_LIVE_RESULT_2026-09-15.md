# V6: primera ejecución interrumpida antes del primer probe

La primera ejecución autorizada se realizó el 15 de septiembre de 2026 y quedó incompleta. **No existe una puntuación V6 ni evidencia nueva de generalización.** No se ha reintentado.

- [Run 34984319827](https://github.com/webtilians/principiodelTodo/actions/runs/34984319827), intento 1; job 104432393789.
- Revisión evaluada: `a2388ce0ddfd4659356b436ec937c9935f4e6e2c`.
- Lanzador: `00992b953dbc64bd462db818bf0fcb5b9210bece`, rama `eval/v6-first-live-20260915`.
- Las 166 pruebas deterministas y el preflight pasaron en GitHub antes de las llamadas al proveedor.
- Candidato, banco V6, scorer y runner congelados sin modificaciones. Sin merges a master/infinito-3.0.

## Causa y responsabilidad del protocolo

La llamada 32, del brazo cognitivo en el sexto turno de la primera trayectoria, respondió a «What is a lenticular cloud?». El proveedor devolvió `status=incomplete`, `incomplete_details.reason=max_output_tokens`, con 96 tokens de salida y 180 de entrada. El texto explicaba las nubes, pero terminó cortado. No fue un error de autenticación ni una caída del proveedor.

El banco fija 96 tokens de salida por turno. El runner nuevo aborta ante cualquier respuesta incompleta. Esa combinación resultó insuficiente para esta pregunta explicativa de relleno. Es un fallo operativo del protocolo de evaluación que preparamos; no demuestra un fallo de memoria o razonamiento del candidato. Las pruebas sintéticas verificaron la parada, pero no establecieron que el presupuesto alcanzara para todas las respuestas reales.

El runner guardó las 32 peticiones y respuestas del proveedor, incluida la truncada, antes de detenerse. No hay report.json ni métricas completas porque el harness no acabó la primera trayectoria.

## Exposición y observaciones parciales

Cinco turnos iniciales completados en ambos brazos. El sexto completó baseline y dejó una respuesta cognitiva truncada. Se expusieron seis de los 122 textos de usuario; las otras tres trayectorias no comenzaron. Se alcanzaron **0 de 28 probes**.

En los cinco turnos completos, los registros muestran cinco eventos ASSERT_FACT correctos: nombre Idris, residencia Graz, bicicleta Marin Rift Zone, idioma Estonian y profesión ceramic conservator. El contexto enviado reflejaba el hecho del turno. Son observaciones de alta inicial, no pruebas de recuerdo a largo plazo ni de actualización/retractación.

La solicitud al modelo en el sexto turno no contiene un bloque adicional INFINITO CONTEXT, aunque sí conserva el historial conversacional breve previsto en ambos brazos. Como la llamada abortó antes de turn_result, no se guardó el paquete de contexto de ese turno; la ausencia del bloque enviado no permite reconstruir todos sus diagnósticos internos.

## Consumo registrado

| Componente | Llamadas | Entrada | Salida | Total tokens |
| --- | ---: | ---: | ---: | ---: |
| baseline_answers | 6 | 968 | 162 | 1130 |
| cognitive_answers | 6 | 1289 | 162 | 1451 |
| embeddings | 15 | 129 | 0 | 129 |
| events | 5 | 1907 | 304 | 2211 |
| reranker | 0 | 0 | 0 | 0 |
| **Total** | **32** | **4293** | **628** | **4921** |

Incluye la respuesta truncada y los embeddings. No se estima importe monetario: no hay tabla de precios congelada.

## Auditoría probe por probe

Todos quedaron sin alcanzar. No asignamos ceros, empates ni aprobados a casos no ejecutados.

| Trayectoria | Probe | Estado |
| --- | --- | --- |
| v6_profile_register | v6 profile register | No alcanzado |
| v6_profile_register | v6 revised profile | No alcanzado |
| v6_profile_register | v6 city predecessor | No alcanzado |
| v6_profile_register | v6 bicycle predecessor | No alcanzado |
| v6_profile_register | v6 latest residence | No alcanzado |
| v6_profile_register | v6 second predecessor | No alcanzado |
| v6_profile_register | v6 profile isolation | No alcanzado |
| v6_schedule_register | v6 calendar interval | No alcanzado |
| v6_schedule_register | v6 remaining appointments | No alcanzado |
| v6_schedule_register | v6 friday only | No alcanzado |
| v6_schedule_register | v6 elapsed morning | No alcanzado |
| v6_schedule_register | v6 closed violin | No alcanzado |
| v6_schedule_register | v6 future map | No alcanzado |
| v6_schedule_register | v6 calendar isolation | No alcanzado |
| v6_preference_register | v6 crafts | No alcanzado |
| v6_preference_register | v6 instrument | No alcanzado |
| v6_preference_register | v6 retracted bird | No alcanzado |
| v6_preference_register | v6 retracted skating | No alcanzado |
| v6_preference_register | v6 new craft | No alcanzado |
| v6_preference_register | v6 current nature | No alcanzado |
| v6_preference_register | v6 preference isolation | No alcanzado |
| v6_mixed_register | v6 mixed profile | No alcanzado |
| v6_mixed_register | v6 literal data | No alcanzado |
| v6_mixed_register | v6 pet scope | No alcanzado |
| v6_mixed_register | v6 mixed predecessor | No alcanzado |
| v6_mixed_register | v6 mixed retraction | No alcanzado |
| v6_mixed_register | v6 mixed future | No alcanzado |
| v6_mixed_register | v6 injection isolation | No alcanzado |

## Decisión y siguiente paso

Integración: **NOT_APPROVED**. No se puede evaluar ningún umbral de calidad, ningún control estricto de contexto vacío ni los objetivos finales. No se cambia arquitectura basándose en esta ejecución.

El próximo trabajo acotado es revisar, con datos de desarrollo ajenos a V6, la política global de longitud y de truncamiento del protocolo. Cualquier cambio de presupuesto o tratamiento de respuestas incompletas debe congelarse como una nueva configuración antes de otra llamada en vivo. No se modifica el banco congelado para permitir que termine.

La autorización de una sola ejecución ya se consumió. No habrá repetición automática. V6 queda registrado como parcialmente expuesto; una ejecución posterior debe identificar esta exposición y no presentarse como el primer V6 limpio. Para una afirmación nueva de generalización tras ajustes informados por esta exposición, se necesita otro banco independiente.

## Evidencia preservada

El ZIP original se conserva en `reports/infinito3/v6-first-live-20260915/artifact.zip`, sin recomprimir. `audit.json` contiene el resumen estructurado y los 28 estados no alcanzados.

- Artifact ID: `10401759912`.
- ZIP SHA256: `56e0774951a36ba0c7ce7511db2a580ad2f750a6d167a08450d8a74389f3eb9d`.
- JSONL interno SHA256: `a9b8ae24b58a0296e7084c9941f1ba8fb324de973a242d9542d9d7b0bd497e8c`.
- El ZIP contiene únicamente `provider-and-turn-audit.jsonl` (587.513 bytes).
- Las credenciales no forman parte del registro del runner.
