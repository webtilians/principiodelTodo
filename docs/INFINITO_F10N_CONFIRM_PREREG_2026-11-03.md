# INFINITO F10-N-C — confirmación prospectiva de memoria de efectos y síntesis SRX, sin aprender nada más

**PRERREGISTRO PUBLICADO ANTES de escribir generador, evaluador o auditor F10-N-C y ANTES de generar cualquier mundo 26300000+. Fecha: 2026-11-03, zona Europa/Madrid.**

## Pregunta y resultados ya observados (no repetidos como hipótesis nueva)

F10-N obtuvo **F10N_EFFECT_MEMORY_COMPOSITION_PASS** en OOD 26400000+, 97,41% (1995/2048) para memoria de efectos C, y propuso SRX en 2048/2048 casos, aunque SRX tuvo cero ejecuciones durante TRAIN. La exploración dirigida D no obtuvo ventaja sobre C (G4 FAIL 0/4). En la réplica 2, una tarea conocida DEV quedó en 84,38%, con G0 individual FAIL; G0 global PASS 3/4. Es una familia de simulación sintética con catálogo de 85 programas, estado y semántica de objetivo provistos por el investigador; no implica invención de operaciones ni razonamiento autónomo general.

**Hipótesis de confirmación**: con esas **mismas memorias de efectos y tablas originales congeladas**, el agente puede volver a proponer una cadena nunca ejecutada en entrenamiento y utilizarla correctamente en 128 nuevos mundos base por réplica, sin repetir el aprendizaje ni cambiar el planificador.

## Congelación verificable y cronología obligatoria

- **Repositorio/código original de F10-N**: commit exacto **6d3e7bd8753f3ad5d1d4a6e44208866bac036298** de la rama \`experiment/f10n-explicit-experience-effect-memory\`. Se copian sin cambios: \`observatory/f10n_worlds.py\`, \`observatory/f10n_memory.py\`, \`observatory/f10n_experiment.py\`, \`observatory/f10n_audit.py\`, \`observatory/f10m_experiment.py\`, \`observatory/f10l_train.py\` y \`src/infinito3\`. No se edita \`master\` ni el producto.
- **Experiencias y 12 memorias originales F10-N**: Descargar **solo** los artefactos \`f10n-original-experience-0\` a \`f10n-original-experience-3\` del **run original 41714113859**. Verificar los SHA-256 del \`training.json\` de cada réplica para \`reward_table.json\`, \`learned_effects_uniform.json\`, \`learned_effects_directed.json\` y logs de experiencias originales; comprobar \`train_heldout_srx_count=0\`, \`confirm_26300000_sealed=true\`, \`episodes_per_condition=98304\` y \`unseen_memory_flip1_never_trained=true\`. **No reentrenar, no reconstruir ni reoptimizar las memorias**.
- **Operadores originales F10-L**: Descargar los cinco checkpoints originales por réplica de \`f10l-stage1-0..3\` y el manifiesto \`f10l-independent-g0-original\` del **run original 41685499939**. Verificar los 20 SHA originales contra G0 y contra la identidad registrada en los cuatro \`training.json\` de F10-N; \`sel_uniform_pair\` y \`xor_two_actions\` solo en inferencia, \`requires_grad=False\`. Ni entrenamiento, ni actualización, ni selección de un checkpoint nuevo.
- **Auditoría original ya completada**: [Actions 41714672031](https://github.com/webtilians/principiodelTodo/actions/runs/41714672031), solo auditor independiente sobre artefactos antiguos y gates originales. El primer auditor del run 41714113859 falló por sombrear una variable, corregido posteriormente **sin cambiar datos, umbrales ni inferencia**. La confirmación debe usar el estado ORIGINAL congelado, no volver a ejecutar el experimento original.
- **Antes de cualquier nuevo caso CONFIRM**, prueba con semillas **OOD antiguas ya abiertas** (26400000+) de que el nuevo generador reproduce **bit a bit** los 2048 casos OOD originales (incluyendo float32, firmas, valores, mascaras y hashes) y de que el evaluador original \`observatory.f10n_experiment.evaluate\` reproduce exactamente todas las predicciones DEV/OOD originales almacenadas en \`rows.json\`, con los mismos SHA. El nuevo evaluador debe llamar el **mismo \`evaluate\` original sin modificaciones** inyectando exclusivamente el nuevo origen de datos en \`dataset("ood",r)\`. Devolver separadamente las predicciones confirmatorias (el campo \`split\` se marca \`confirm\` en el archivo final de resultados; esto es metadato, no alteración de inferencia).
- Ante cualquier fallo de hashes/regresión no abrir confirmación y reportar error técnico; no cambiar pesos, umbrales, mundo o algoritmo para rescatar puntuaciones.

## Partición de confirmación 26300000+ — reservada hasta verificar todo lo anterior

- **Namespace de semilla:** \`26_300_000 + 10_000*r + j\`, con \`r∈{0,1,2,3}\` y \`j∈{0,…,127}\`. Cuatro réplicas originales, 128 mundos base por réplica, cuatro variantes \`U_before\`, \`U_after\`, \`S_before\`, \`S_after\` por mundo → **512 casos por réplica, 2048 confirmatorios**.
- **Misma distribución OOD que F10-N original y cero entrenamiento en ella**: solo tarea \`MEMORY,FLIP=1\`, informes n=9 si j par o 11 si impar, 4 o 5 firmados, cuatro regímenes de fuentes \`iid/signed_source0/signed_source1/alternating\` siguiendo \`(j//2)%4\`; se varían versiones, orden de almacenamiento y bit público señuelo con el mismo algoritmo RNG. Solo \`S_after\` invierte la verdad acreditada. **No introducir deliberadamente nuevos cambios de distribución**; la novedad es el bloque de semillas.
- Igual catálogo de 85 cadenas que F10-N y mismas memorias. Ninguna cadena target entregada como ejemplo nuevo. Cada brazo A/B/C/D decide la secuencia al inicio usando exclusivamente (SOURCE=MEMORY, FLIP=1, public_bit) y la memoria congelada; ningún ganador o F4 verdadero antes de READ. Tras la decisión, aplicar \`normal\`, \`shuffle\`, \`x_disabled\`, \`public_randomized\` exactamente como el código original. Se registra la salida, programa, SHA de caso y respuesta oracle SRX como **diagnóstico de evaluador**, no entrada a la política.
- **NINGÚN gradiente, \`observe\`, \`train\`, actualización de \`Counter\` ni escritura de memoria durante confirmación**. Verificar SHA antes y después de inferencia. El único resultado nuevo son las predicciones. Los cuatro brazos se evalúan aunque C falle.

## Estadística y puertas preregistradas

Conservar las **mismas condiciones científicas originales** de F10-N; no seleccionar entre réplicas ni cambiar el mínimo tras ver los datos. El G0 original DEV PASS 3/4 es prerrequisito congelado, NO se recalcula ni se reentrena tras OOD. En cada réplica se calculan precisión, CIs, estratos N/fuente/variante, proporción de SRX, programas elegidos, intervenciones y coste.

**IC95:** exactamente **2000 remuestreos con reposición por los 128 MUNDOS BASE** por réplica, manteniendo juntas las cuatro variantes contrafactuales; \`rng=np.random.default_rng(268000+100*r+offset)\`. Offset=0 para precisión C, 1 para C−shuffle, 2 para C−x_disabled, 3 para C−B (mismas predicciones por caso), 4 para D−C. No usar intervalos binomiales de 2048 casos correlacionados.

- **C-G1 transferencia inédita**: precisión C OOD confirm >=80%, límite inferior de CI95 por mundo >70%, precisión >=70% para **cada** N=9/11, régimen de procedencia y variante, y fracción de C seleccionando exactamente SRX >=80% (se reportará elección observada). Cumplido en >=3/4 réplicas.
- **C-G2 causalidad de F4 y XOR**: caídas C(normal)−C(shuffle) y C(normal)−C(x_disabled) >=+20 puntos porcentuales **cada una**, límite inferior de IC95 por mundo >0 en ambas. Cumplido en >=3/4.
- **C-G3 memoria de efectos vs puntuación aditiva**: C−B >=+15pp, límite inferior IC95 por mundo >0; en >=3/4 réplicas. A y B se evalúan con exactamente las mismas memorias/recompensas originales utilizadas por C durante F10-N.
- **C-G4 dirigidos vs uniforme, diagnóstico**: D−C >=+5pp, CI inferior>0 en >=3/4. **No** requisito para el PASS global.
- **Confirmación global \`F10N_CONFIRM_PASS\` SOLO si el G0 original estaba PASS>=3/4 y C-G1, C-G2, C-G3 pasan cada uno en >=3/4 réplicas**. De otro modo \`F10N_CONFIRM_FAIL\`; un problema de hashes/ejecución produce \`F10N_CONFIRM_TECHNICAL_INCOMPLETE\` y no se comunica como confirmación científica. G4 permanecerá negativo si no aparece diferencia nueva.

Auditoría de salida **en job separado** sin PyTorch ni importación de generador/modelos: examina SHA de memorias/manifiestos, disjunción de identidad de 2048 casos, réplicas y 4 variantes por mundo, 2048 filas por brazo, competencia en estratos, CIs de bootstrap preregistrados y gates independientes. Comparar con el original F10-N solo como contexto, **no** ajustar código/hiperparámetros según el resultado.

## Interpretación de un PASS, si lo hubiese

Un PASS probaría **replicación interna en semillas nuevas de la misma familia sintética** de la síntesis y uso causal de SRX desde una memoria de efectos y búsqueda exhaustiva con catálogo, estados y semántica de objetivos diseñados por el investigador. **No** prueba invención autónoma de operadores, detección de estado latente desde observaciones crudas, adaptación online después de READ, superioridad de exploración dirigida ni aplicabilidad al producto/LLM real.

**STOP después de esta única confirmación**; no abrir más semillas para mejorar puntuaciones, no modificar memorias ni fusionar esta rama experimental con master.
