# INFINITO F10-B — Resultado DEV: utilidad causal de memoria con procedencia y versiones

**2026-11-02. Resultado FINAL DEL DESARROLLO, NO CONFIRMACIÓN.** [Prerregistro prospectivo](./INFINITO_F10B_PROVENANCE_TEMPORAL_PREREG.md) congelado antes de implementar generador/modelos/ejecutores o abrir nuevos mundos: commit `daa7b621c933d7bf8970dfa13b628d3baf34594e`, con borrador inicial conservado en `45296e1c5b5b20d1918412c51c656f5c73d70d64`. [Ejecución GitHub Actions #41443064754](https://github.com/webtilians/principiodelTodo/actions/runs/41443064754): **SUCCESS técnico** en cuatro réplicas, 8 entrenamientos y el agregado Python stdlib independiente. Artefactos guardados por réplica: dos checkpoints, manifest TRAIN y informe detallado DEV; además gate agregado.

## Alcance y qué significa F4 aquí

Tarea **nueva y sintética**, no MiniGrid: 24 claves por mundo y exactamente 60 informes públicos de fuente A/B, bit, credencial de firma observable y timestamp. Para cada mundo las **mismas 60 observaciones** se usan para cuatro preguntas finales distintas (último informe firmado/no firmado × estable/revisado); el identificador solicitado solo aparece al FINAL. Valor correcto = bit del **último informe firmado** para esa clave, calculado a partir de la evidencia pública, nunca entregado al actor como etiqueta de autenticidad oculta. Informe sin firma puede contradecir al firmado; informe firmado antiguo puede quedar sustituido por actualización firmada posterior.

**F4 aquí** es un *log FIFO estructurado* de hasta 64 informes públicos: **NO** es la F4 ContextDiversity original de F9-P, ni otro test sobre los mismos checkpoints. El actor es un **GRU64 NUEVO** con matching de clave fijo como sesgo arquitectónico; aprende puntuaciones para firma, antigüedad y fuente. Compite contra **otro actor GRU64 igual, entrenado con el log a cero**, con idénticos datos, inicialización, batches y 12 épocas. No se entrenó cabeza READ ni se seleccionaron casos por D. Es clasificación secuencial supervisada, autónoma desde el primer evento (sin prefijo docente).

TRAIN exclusivamente `13_100_000+10_000*index+j, j=0..255`: 256 mundos/actor y 4 consultas/mundo. DEV exclusivo `13_200_000+10_000*index+j, j=0..63`: 64 mundos/actor y 4 consultas/mundo. **4 réplicas × 64 mundos × 4 consultas = 1024 estados DEV, pero solo 256 mundos independientes**. CONFIRM `13_300_000+` sellado; todos los antiguos holdouts y F10-A `12_300_000+` no reutilizados.

READ, SKIP, memoria obsoleta y firma falsificada son **intervenciones sobre el log al final**. El historial público ya consumido, los pesos, la clave final y la GRU son idénticos en esos cuatro contrafactuales. La falsificación de firma **es un ataque fuera de distribución**, no un evento natural que el actor pueda verificar criptográficamente.

## Resultado 1 — ventaja de memoria externa reproducible entre réplicas

| Réplica | READ limpio /256 | SKIP mismo actor /256 | GRU-only entrenada /256 | READ−SKIP | IC95% bootstrap por mundo |
|---|---:|---:|---:|---:|---|
| 0 | 188 | 148 | 132 | **+0,1563** | [0,0820; 0,2305] |
| 1 | 195 | 131 | 118 | **+0,2500** | [0,1914; 0,3086] |
| 2 | 184 | 115 | 132 | **+0,2695** | [0,1992; 0,3398] |
| 3 | 197 | 116 | 124 | **+0,3164** | [0,2343; 0,4023] |
| **Total** | **764 /1024 = 74,6%** | **510 /1024 = 49,8%** | **506 /1024 = 49,4%** | **+0,2480** | **No calcular un IC como si fueran 1024 mundos independientes** |

Las cuatro réplicas superan el subcriterio global de >=10 pp de utilidad READ sobre SKIP y límite inferior bootstrap por mundo >0; **4/4 superan además a la GRU-only entrenada** por >5 pp. Esto sí constituye evidencia de **valor causal de la memoria externa dentro de esta tarea**, más fuerte que las asociaciones H/C previas de F9-Y. No equivale aún a que el agente haya aprendido la **decisión de consultar**: se fuerza READ al evaluar.

## Resultado 2 — localizar la falla de procedencia temporal

| Condición final | READ limpio aciertos /256 | Precisión | D_clean+ por réplicas 0/1/2/3 |
|---|---:|---:|---|
| Firmado, estable | **256** | **100,0%** | 28 / 34 / 36 / 36 |
| Sin firma, estable | **213** | **83,2%** | 23 / 29 / 25 / 35 |
| Firmado, revisado | **247** | **96,5%** | 31 / 29 / 29 / 35 |
| **Sin firma, revisado** | **48** | **18,8%** | **8 / 4 / 3 / 14** |

El agente aprende perfectamente o casi perfectamente las categorías firmado-estable y firmado-revisado, y suele ignorar informes sin firma cuando el valor oficial **no ha cambiado**. Sin embargo, cuando una fuente autorizada actualiza un valor y **un informe posterior no firmado contradice esa actualización**, el agente suele fallar. Confunde recencia aparente con validez en un caso que requiere integrar **ambos** criterios simultáneamente, no uno aislado.

Por el protocolo prospectivo, cada actor debía acumular >=8 D+ y >=8 no positivos en cada una de sus cuatro celdas. Solo las réplicas 0 y 3 cubrieron las cuatro condiciones: **2/4**, por debajo de las 4/4 requeridas. Las réplicas 1 y 2 solo obtuvieron 4 y 3 D+ respectivamente en sin-firma+revisado. El gate de procedencia exigía >=80% en las cuatro condiciones para >=3/4 actores; **0/4** cumplen esto, debido al estrato sin-firma+revisado. NO debe promediarse con otras condiciones para decir que pasó.

## Resultado 3 — intervenciones de validez y falsificación

Resultados agregados sobre los mismos 1024 estados:
- F4 auténtica: **764/1024 = 74,6%**.
- SKIP mismo actor: **510/1024 = 49,8%**.
- F4 obsoleta forzando versiones antiguas como últimas firmadas: **469/1024 = 45,8%**. En la condición firmado+revisado, el rendimiento cae de **247/256 a 0/256** si se retiran de F4 las actualizaciones firmadas. Esto refuerza que la memoria auténtica contiene información causalmente utilizada.
- F4 con credencial del último informe sin firma falsificada a firmada: **611/1024 = 59,7%**. En sin-firma+estable, baja de **213/256 a 88/256**. Este es un ataque **fuera de distribución**; no es prueba de resistencia criptográfica ni de desconfianza robusta ante toda manipulación.

Interpretación: el agente depende de las actualizaciones externas y del token de procedencia para muchas decisiones, pero todavía **no implementa de forma fiable la regla conjunta «la firma autorizada más reciente para cada clave»**. La arquitectura de atención con un score aditivo sencillo puede sufrir al tener tres informes para una clave (original firmado, actualización firmada, posterior no firmado). Es una **hipótesis mecanicista nueva**, no una demostración ni justificación para ajustar este mismo DEV.

## Gate científico exacto

El agregador precomprometido comunica **`ACTOR_MEMORY_VALIDITY_DEV_FAIL`**:
- Integridad de tarea, splits, consultas compartidas y contrafactuales: **PASS técnico**.
- Ventaja global READ−SKIP y frente a GRU-only: **4/4 PASS en ambos subcriterios**.
- Positividad >=8 D+ en todas las cuatro celdas de cada réplica: **2/4 — FAIL del requisito 4/4**.
- Acierto >=80% por condición en >=3 réplicas: **0/4 — FAIL**.
- **STOP**: no entrenar cabeza READ con estos DEV, no reoptimizar modelo aquí, no abrir confirmación `13_300_000+`.

## Siguiente pregunta separada (NO ejecutada)

Investigar con nueva familia de mundos y reglas preregistradas si una operación explícita de resolución *versionada y acreditada* (e.g. atención relacional/pairwise con pruebas de precedencia o consulta recurrente sobre registros) puede aprender a elegir la **última evidencia acreditada**, no la última que se ve. No proporcionar al agente verdad del evaluador: la prioridad entre informes se determina a partir de credenciales/tiempos públicos. Comparar contra este actor F10-B congelado y frente a GRU-only en nuevos splits, y medir si la memoria sigue teniendo efecto causal con múltiples contradicciones. No usar estos DEV para ajustar hiperparámetros retrospectivamente ni llamarlos independientes.

**Limitaciones:** evidencia sintética y credenciales públicas modeladas como bits, fuente A/B aleatorizada, una sola pregunta binaria al final, matching de clave parcialmente arquitectónico, sin LLM/RAG/continuidad longitudinal ni transferencia a MiniGrid. El siguiente objetivo todavía es demostrar utilidad de la memoria bajo información contradictoria sin atajos y luego —en estudio separado— aprender cuándo hacer READ.
