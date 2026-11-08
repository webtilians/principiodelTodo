# INFINITO F12-H — Composición de efectos observados con un LLM real

**Prerregistro prospectivo — 8 de noviembre de 2026. Este documento debe existir en un commit anterior a cualquier implementación, ejecución o resultado F12-H.**

## Pregunta y antecedentes
¿Puede un LLM Qwen2.5-7B congelado convertir su propia experiencia con herramientas en un plan reutilizable que componga dos capacidades observadas **por separado**, sin haber ejecutado esa composición durante TRAIN? ¿Supera esa ayuda a la recuperación episódica competente y a obtener el mismo plan sin evidencia? La hipótesis surge de F10-N/P (síntesis de una secuencia SRX nunca ejecutada mediante efectos públicos) y del NO-GO de F12-E/F/G para estrategias narrativas y contextos con combinaciones perfil/revisión ya vistas. **No** demostrar aprendizaje de pesos, inteligencia general ni beneficio frente a un planner diseñado por el investigador.

## Congelación experimental y prohibiciones
- Modelo real **Qwen2.5-7B-Instruct GGUF Q4_K_M**, repo `bartowski/Qwen2.5-7B-Instruct-GGUF`, revisión `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`; `llama-cpp-python==0.3.36`, temperature=0, n_ctx=4096, CPU. Reusar `FrozenGGUF` de F12-F, no editar su configuración tras resultados. LLM único como actor y como generador de políticas.
- Memoria auténtica `SQLiteCognitiveMemoryStore(path=":memory:", embedding_provider=HashEmbeddingProvider())`; solo datos sintéticos. Ni servidor MCP de usuario, ni secretos, ni producto.
- Protocolo de **piloto diagnóstico** con TRAIN y TRAIN-heldout nuevos. No reutilizar S/H/V/W/U/Y de F12-E/F/G, ni usar F11 DEV/CONFIRM. Las tareas de confirmación independiente permanecen sin generarse/abrir. No reajustar con resultados del heldout.

## Nueva familia de operaciones y aprendizaje
Simulador de un **registro binario** y herramientas `op_p`, `op_s`, `op_r`, `op_x`, `op_j`, `verify`. Las operaciones tienen acciones y mensajes públicos (valor 0/1, ACK, error, verificación positiva/negativa); solo el simulador conoce el bit oculto, el estado de control y el criterio de éxito. La entrada pública (`public_bit`) y objetivos fuente/transformación son visibles.

- Efectos **privados al simulador**, no se entregan como tabla al actor: `op_p` carga bit público; `op_s` habilita `op_r`; `op_r` copia el bit oculto si fue habilitado, o devuelve error; `op_x` invierte el registro si existe; `op_j` deja el registro sin cambiar (distractor); `verify` compara registro frente al objetivo calculado por evaluador, sin revelar valor oculto antes de leerlo.
- Catálogo público indica nombres, tipos de respuesta, que las acciones pueden cambiar un registro y que se requiere `verify` para finalizar; **no explica qué transforma cada operación**, ni revela bit oculto/objetivo resuelto. Las respuestas de las herramientas, no un oráculo, son la fuente de aprendizaje.
- **TRAIN**: nueve incidentes prefijados Z0–Z8: tres tareas copiar bit público sin invertir (Z0–Z2), tres invertir bit público (Z3–Z5) y tres copiar bit oculto sin invertir (Z6–Z8). Bits 0/1 alternos, objetivos independientes del servicio. El mismo LLM decide hasta **7 llamadas** por incidente sin memoria, recibe feedback y su experiencia completa (incluidos errores) se almacena. No se introduce una demostración experta ni se rellena artificialmente un éxito. Deberá emerger evidencia de `op_x` sobre ambos valores y `op_s`→`op_r`.
- **TRAIN-heldout**: seis incidentes nuevos Q0–Q5 con fuente **oculta e inversión** a la vez; tres bits ocultos 0 y tres 1. No existe una secuencia TRAIN que resuelva esa misma combinación. Se mantienen las herramientas y los efectos, por tanto es transferencia composicional dentro de la familia, **no** generalización a semánticas de API nuevas.
- Horizontes **tight=4** y **slack=6**; cada caso/brazo utiliza estado nuevo. En todos los brazos: modelo/pesos, sistema, catálogo, condiciones, feedback y acción válida idénticos. Orden de los brazos rotado determinísticamente. No reinterpretar pruebas tight donde todos puedan quedar en cero.

## Tratamientos y controles preregistrados
Se fija el mismo límite de memoria de **1600 caracteres**, registrando SHA y texto exacto. Misma recuperación `store.search` de episodios realmente observados y selección por familias de capacidad public/hidden e invert/copy, sin acceso a resultados del heldout.

- **A0**: sin memoria y sin política.
- **A1**: fuerte recuperación episódica INFINITO; conservar secuencias públicas completas y mezclas de tareas de entrenamiento con ambas capacidades, ordenar con relevancia y no solo recencia.
- **A2**: mismos episodios y un **plan provisional creado por Qwen** antes de ver heldout, basado en acciones y efectos observados, con citas de Z*. No presentarlo como hecho; no validar su éxito con el evaluador ni corregir su formato. Contexto total <=1600.
- **A3**: mismo formato de plan provisional creado por Qwen **sin acceso a experiencias**, como control de capacidad de razonar desde el prompt/modelo preentrenado. El presupuesto de síntesis y formato se registran.
- **A4**: **control investigador** de modelo de efectos públicos con conteos de transiciones y reglas de precedencia observadas, inspirado en F10-P; solo podrá producir plan si hay observaciones de las piezas necesarias. Etiquetarlo `engineered-effect-planner`, no aprendizaje autónomo del LLM; jamás leer bits privados ni hacer consultas contrafactuales al mundo. Registrar qué evidencias sustentan cada paso.

No afirmar causalidad de verificación o aprendizaje si la política propuesta es idéntica a la propuesta sin recuerdos. Reportar contextos, propuesta literal, cambios en la primera acción y éxito.

## Métricas y gates anteriores a los datos
**P0 integridad:** pesos congelados/cotejados, 9 episodios TRAIN, **60 evaluaciones** (=6 casos x 2 horizontes x 5 brazos), memorias congeladas antes de heldout, sin fugas privadas ni modificaciones del almacén durante evaluación; registrar todas las llamadas, prompts, salida, tokens y segundos. Fallo => INVALID/BLOCKED.
**P1 evidencia suficiente:** >=5/9 TRAIN con verify positivo, >=1 secuencia pública que muestre `op_x` invirtiendo 0 y >=1 que muestre `op_x` invirtiendo 1; >=2 secuencias observadas donde `op_s` precede a `op_r` exitoso. Falta => INCONCLUSIVE, no rescatar con ejemplos sintéticos.
**P2 composición inédita:** auditoría automática confirma que ninguna trayectoria TRAIN contiene `op_s`, `op_r`, `op_x` y `verify` en ese orden. Si se da, INVALID por contaminación del holdout.
**P3 utilidad de política:** A2 > A1 por >=2/6 tareas en tight, sin degradación en slack, y no peor que A3 en ambos horizontes; si no, NO-GO. Los seis casos son unidades emparejadas; no tratar 12 medidas como independientes.
**P4 atribución a experiencias:** plan de A2 cita episodios existentes con acciones observadas relevantes y muestra una diferencia de contenido material frente al plan A3, y A2 debe superar A3 en al menos 2/6 tight. Citas solas no equivalen a reglas correctas; inspección posterior no cambia el criterio.
**P5 diagnóstico F10:** A4 exhibe al menos una planificación fundada solo en efectos públicos; registrar tanto aciertos como errores; no utilizar A4 como referencia de GO si el presupuesto/contexto es desigual.
**P6 coste:** tokens de entrada de evaluación A2 <=1,25 * A1; reportar por separado TRAIN, síntesis, evaluación, coste de cálculo y caracteres reales.

**GO únicamente para diseñar una confirmación nueva si P0–P6 pasan y no hay hallazgos adversariales sustantivos**. Un run técnicamente SUCCESS con P3/P4 FAIL es NO-GO científico. Nueve experiencias y seis heldout hacen esto un piloto, **no** evidencia generalizable.

## Auditoría causal y límites
- La salida de `verify` devuelve solo success/fail; el modelo de efectos ve solo acciones ejecutadas y salidas de herramienta. La compatibilidad/objetivo privado nunca debe llegar al sintetizador, al planner ni a los contextos.
- Contrato de pruebas deterministas con `FakeOnly`: prueba tipos, split, truncación, ausencia de fugas, congelación de memoria y gates; ningún fake cuenta como inferencia.
- Usar `HashEmbeddingProvider` (no embeddings aprendidos); la recuperación no equivale a RAG semántico competente del mundo real. A1 debe compactar todos los eventos con precisión. No cambiar selección después de ver Q0–Q5.
- No modificar pesos, no entrenar gradientes, no integrar producto, no hacer merge automático. Publicar resultados negativos con el mismo detalle que positivos.
