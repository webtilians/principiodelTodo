# F12-I — Inferir planes a partir de efectos públicos, sin confundirlo con su ejecución

**Prerregistro prospectivo, 8/11/2026. Este commit precede a cualquier código, prueba o ejecución F12-I.**

## Hipótesis

¿Puede el mismo Qwen2.5-7B congelado inferir una secuencia **no ejecutada durante adquisición de evidencia** que invierta un bit oculto, a partir de efectos observados separadamente? F12-H quedó inconcluso: solo 3/9 TRAIN tuvieron éxito; el actor no ejecutó sus planes. Aquí separamos la **inducción del plan** de la ejecución: un ejecutor determinista evalúa cada plan propuesto en cuatro contrafactuales reservados, sin participación del LLM y **sin que su resultado vuelva al planificador**.

No es aprendizaje autónomo de exploración: un *probe scheduler* del investigador ejecuta únicamente sondeos parciales, declarados abajo; el LLM aprende, si puede, de sus resultados públicos. No introducir demostraciones del plan completo, ni usar datos privados como textos de memoria. Conservar F12-H intacto y no reutilizar Z/Q, S/H/V/W/U/Y de F12 anteriores.

## Modelo y backend

Mismo `FrozenGGUF` F12-F: Qwen2.5-7B-Instruct Q4_K_M, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 de pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, `llama-cpp-python==0.3.36`, CPU, temperatura 0, contexto 4096. Misma instrucción y esquema para todos los brazos. Sin gradientes. `SQLiteCognitiveMemoryStore(path=":memory:")` con `HashEmbeddingProvider` auténtico; datos sintéticos exclusivamente.

## Ocho mundos (pareados; congelados)

Hay cinco herramientas opacas `t0..t4` más `verify`. Un mundo asigna de forma privada las funciones LOAD (carga entrada pública), ENABLE (habilita READ), READ (lee bit oculto), FLIP (invierte el registro) y HOLD (mantiene el valor). El catálogo público solo enumera herramientas y tipos de respuesta `{"value":0/1}`, `{"ack":true}`, `{"error":"..."}`, `{"verified":bool}`; no revela qué herramienta implementa qué función.

La tabla aquí es verdad **exclusiva del evaluador**, no del LLM, contextos o memoria. Secuencia de roles por `t0,t1,t2,t3,t4`, congelada:

- I0: LOAD,ENABLE,READ,FLIP,HOLD
- I1: ENABLE,READ,FLIP,HOLD,LOAD
- I2: READ,FLIP,HOLD,LOAD,ENABLE
- I3: FLIP,HOLD,LOAD,ENABLE,READ
- I4: HOLD,LOAD,ENABLE,READ,FLIP
- I5: ENABLE,FLIP,LOAD,HOLD,READ
- I6: FLIP,READ,ENABLE,LOAD,HOLD
- I7: READ,HOLD,FLIP,ENABLE,LOAD

Mundos I0..I7 constituyen **un piloto de ocho unidades**, no ocho réplicas independientes de aprendizaje neuronal. Para cada mundo, el mismo plan se ejecutará offline con (bit público, bit oculto) en `(0,0),(0,1),(1,0),(1,1)`. Solo 4/4 éxitos significa plan universal para ese mundo. Aislar el estado para cada ejecución. El LLM no verá ni recibirá resultados de esta evaluación.

## Experiencias observadas: adquisición independiente de la capacidad del LLM

Se ejecutan diez sondeos públicos por mundo, con world reset tras cada uno:

1. LOAD con público 0; 2. LOAD con público 1.
3. LOAD→FLIP con público 0; 4. LOAD→FLIP con público 1.
5. ENABLE→READ con oculto 0; 6. ENABLE→READ con oculto 1.
7. READ sin ENABLE (error).
8. LOAD→HOLD con público 0; 9. LOAD→HOLD con público 1.
10. ENABLE solo (ACK).

En todos los casos, guardar **solo acciones ejecutadas y respuestas públicas**, entradas públicas y source ID. Ninguna secuencia de sondeo incluirá ENABLE→READ→FLIP→verify ni observará la combinación objetivo. Las identidades de herramienta dentro del sondeo las establece el scheduler externo (no el LLM); esta es una limitación declarada, no experiencia elegida autónomamente.

Los textos de episodios se guardan en SQLite y se recuperan con `store.search` (top_k suficiente para los diez) sin incluir mapping de roles del evaluador. Asegurar que 10/10 estén en el contexto y que su longitud sea ≤4500 caracteres; si falta evidencia, INVALID, nunca completar con respuestas expertas. Capturar el texto exacto y SHA.

## Brazos y métricas

Cuatro brazos por mundo, **un único plan por brazo, sin ejecución online por LLM**:

- A0: LLM sin episodios, herramientas y objetivo idénticos.
- A1: LLM recibe los diez episodios públicos completos de ese mundo.
- A2: LLM recibe episodios con todas las observaciones de FLIP **eliminadas**, pero mismos IDs, orden y demás observaciones. Control de ablación causal, sin inventar observaciones.
- A3: planificador programado usando exclusivamente respuestas y acciones de los sondeos; control superior de inferencia de efectos, claramente etiquetado como diseñado por el investigador (no un LLM).

Los tres brazos LLM deben devolver **solo** `{"plan":[...],"sources":[...]}` (herramientas existentes, longitud 1..5, `verify` final). A0 permite `sources=[]`. A1/A2 deben citar IDs existentes, de lo contrario INVALID para ese brazo. No reparar esquemas, no traducir frases, no incorporar feedback de evaluación. A3 deriva ENABLE, READ y FLIP de patrones de resultados públicos con comprobaciones de ambos valores; si no puede hacerlo, ABSTAIN.

La medición primaria es número de mundos con planes **4/4**, no éxitos individuales tratados como independientes. Secundarias: éxitos 0-4, plan válido, citas observadas, longitud, efectos irrelevantes, tokens, abstenciones, tiempo.

## Gates preregistrados

- G0 Integridad: modelo/hash y protocolo congelados, 8 mundos, 10 sondeos por mundo, 24 generaciones reales de planes, 8 planes programados, 128 evaluaciones offline (=8×4 brazos×4 combinaciones), misma identidad en cada brazo, cero fugas; cualquier incumplimiento => INVALID.
- G1 Evidencia: todos los mundos tienen los 10 sondeos intactos y cobertura de entrada 0/1 de LOAD→FLIP, además de READ tras ENABLE; cero secuencias completas de prueba; si no => INVALID.
- G2 Aprendizaje por experiencias: A1 ≥6/8 planes 4/4, ≥3 mundos más que A0 y ≥3 más que A2; no rescatar error de parsing.
- G3 Diagnóstico: A3 ≥7/8 mundos con plan 4/4, solo si evidencia disponible; fracaso de A3 => problema de diseño/preflight, no prueba negativa del LLM.
- G4 Procedencia: cada plan A1 exitoso se basa en source IDs reales y las proposiciones de plan son compatibles con acciones/feedback de los sondeos; examinar salidas literales. Citas por sí solas no prueban causalidad.
- G5 Costes: informe tokens/latencia por brazo, costes de adquisición de sondeos, bytes de memoria; no reclamar eficiencia sin comparación.

GO solo para diseñar confirmación **nueva** si G0..G5 y revisión independiente pasan. 8 mundos y permutaciones relacionadas son piloto diagnóstico; ningún GO directo a Producto, MCP o F11 CONFIRM. No modificar umbrales, tablas ni prompts después de ver resultados de I0..I7. Si el sistema llega a construir planes válidos, **no** atribuir capacidad de ejecutarlos al actor, que se estudia separadamente en F12-J.
