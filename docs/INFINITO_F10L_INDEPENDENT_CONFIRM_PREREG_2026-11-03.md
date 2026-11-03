# INFINITO F10-L-C — CONFIRMACIÓN INDEPENDIENTE CON PESOS ORIGINALES CONGELADOS

**Prerregistro prospectivo: 2026-11-03 (Europa/Madrid). Documento publicado ANTES de crear el generador o ejecutar cualquier mundo de CONFIRM 23300000+.** No se añaden épocas, no se cambia la arquitectura, no se ajusta el umbral de decisión, no se elige el mejor checkpoint y no se modifican recompensas ni objetivos. La confirmación aplica exactamente la política ya evaluada en F10-L: selector \`sel_uniform_pair\` + lector del slot elegido + procesador \`xor_two_actions\` con flip=1, deterministas mediante argmax y logit>=0.

## Contexto y distinción

F10-L superó prospectivamente OOD composición \`23400000+\` con 96,73% descriptivo agregado (1981/2048), G3/G4 4/4, pero NO ha demostrado que el estimador emparejado venza al RL ordinario (XOR G1 0/4; selección G2 1/4). El resultado corresponde a **transferencia entre dos módulos entrenados a base de recompensas locales** en un simulador con operadores y composición programados por el investigador, no al descubrimiento de nuevas operaciones cognitivas. Las recompensas de ambos brazos en XOR binario son informacionalmente equivalentes a la etiqueta correcta.

El objetivo aquí es probar **reproducibilidad fuera de la partición OOD explorada, sin modificación posterior del modelo o evaluación**, NO probar un nuevo algoritmo ni generalidad hacia un LLM real.

## Congelación de código y pesos ANTES de CONFIRM

**Base de investigación original inmutable**: commit \`b5c32cbfb50a24414607e3a24e30e139ca92fede\` de \`experiment/f10l-experience-credit-primitives\`. **Run original** [GitHub Actions #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939), cuya ejecución generó **20 checkpoints originales SHA** en \`f10l-stage1-0..3\` y un manifiesto independiente de sus hashes \`f10l-independent-g0-original/f10l-g0.json\`. La nueva rama \`experiment/f10l-independent-confirmation-23300000\` parte exactamente de dicho commit. Los SHA SHA-256 de todos los modelos de los cuatro réplicas se validarán contra \`train.json\` Y contra \`f10l-g0.json\` de **ese run original**. No se recrearán pesos ni se entrenará nada. No se admiten reejecuciones que seleccionen checkpoints distintos.

Se comparará con \`git diff --exit-code\` el código antiguo \`observatory/f10l_worlds.py\`, \`observatory/f10l_train.py\`, \`observatory/f10l_stage2.py\`, \`observatory/f10l_audit_stage1.py\` y \`observatory/f10l_audit_stage2.py\` con el commit congelado. NO editar estos archivos, ni \`master\`, ni el producto \`src/infinito3\`.

**Antes de abrir CONFIRM**, preflight del nuevo generador por inyección de semillas **OOD ya utilizadas** debe reproducir **bit a bit** las características, valores, máscaras, winner y metadatos (salvo ID/hash) de \`select_world("ood", r, j)\`. Esto prueba que la lógica del mundo es idéntica salvo el namespace de semilla. Los tests preflight jamás generan 23300000+. También se verifica mediante una regresión con el **conjunto OOD original** que la rutina de inferencia nueva emite exactamente las mismas predicciones que \`f10l_stage2.run\` para los 8 modos (normal, permutar F4, anular flip, oracle_slot, controles con otros checkpoints y control simbólico). Esta regresión usa SOLO el OOD antiguo \`23400000+\`, antes de abrir CONFIRM; no modifica código/pesos según su desempeño.

## Partición de CONFIRM

**Semilla por réplica r=0..3 y mundo base j=0..127:** \`23300000 + 10000*r + j\` (nuevas semillas), totalmente disjuntas de TRAIN 231..., DEV 232... y OOD 234.... 128 mundos base por réplica × **cuatro variantes contrafactuales** (\`U_before,U_after,S_before,S_after\`) = 512 casos por réplica = **2048 predicciones de confirmación**. Los cuatro casos de cada mundo constituyen **una sola unidad estadística**.

El generador utiliza **exactamente la distribución OOD preregistrada de F10-L**, sin cambios: N=9 o 11 informes según \`j%2\`; cuatro/cinco firmas, timestamps públicos sin duplicados, ubicación permutada, informe X contradictorio con firma/fecha intervenidas, un unsigned posterior al último firmado incluso tras X, fuentes \`iid\`, \`signed_source0\`, \`signed_source1\` y \`alternating\` según \`(j//2)%4\`. La respuesta cambia solo en \`S_after\`. Los valores F4 NO están disponibles antes de seleccionar slot. **No utilizar la propia confirmación para depurar o reespecificar mundo, algoritmo, evaluación o criterios**. Ante un fallo técnico, reportarlo y distinguirlo explícitamente de un segundo intento; ninguna reejecución con semillas nuevas sin prerregistro nuevo.

## Evaluación y puertas de confirmación congeladas

Una vez autenticados todos los SHA de pesos y confirmada la regresión OOD original, abrir CONFIRM **una sola vez**. Misma evaluación que el run OOD:
- \`composed\`: \`sel_uniform_pair\` argmax, leer su valor F4, \`xor_two_actions\` con flip=1 y salida \`logit>=0\`.
- Intervenciones sin cambio de pesos: \`shuffle\` rota valores F4 entre slots válidos manteniendo metadatos y selección; \`no_flip\` impone flip=0; \`oracle_slot\` inyecta winner solo como diagnóstico post hoc, NO como acción del modelo; controles \`xor_onpolicy_selector_paired\`, \`xor_paired_selector_onpolicy\`, \`symbolic_xor_learned_selector\`, \`symbolic_oracle\`.
- Bootstrap: 2.000 remuestreos por **mundo base** de sus cuatro variantes, seeds \`241000+100*r+offset\`, con \`offset=0\` para accuracy, 1 para \`composed-shuffle\` y 2 para \`composed-no_flip\`. No usar un IC simple de 2.048 respuestas dependientes.
- Estratos por 9/11 informes, las cuatro fuentes y cuatro variantes; informe de precisión de selección del último firmado, precisión por réplica, intervalos de confianza, deltas causales, hashes de cada caso y trazas de integridad de los checkpoints.
- Controles: selección simbólica de última firma y XOR simbólico 100% (sanidad del simulador), precisión base de no usar F4 y respuesta constante, y ventaja descriptiva de los dos módulos alternativos. No cambian la definición de PASS.

**Criterios G3 y G4 IDÉNTICOS al OOD F10-L**:
1. **C-G3 COMPOSICIÓN**: en al menos **3/4 réplicas**, accuracy de \`composed\` >=80%, **límite inferior IC95 por mundo >70%**, y accuracy >=70% en todos los estratos de N, fuente y variante.
2. **C-G4 DEPENDENCIA CAUSAL**: en al menos **3/4 réplicas**, \`composed − shuffle\` >=+20pp y \`composed − no_flip\` >=+20pp, **límites inferiores IC95 >0** en ambos.
3. Confirmación global **\`F10L_CONFIRM_PASS\` solo si C-G3 y C-G4 pasan**. De otro modo **\`F10L_CONFIRM_FAIL\`**. Cualquier error técnico queda marcado como **\`F10L_CONFIRM_TECHNICAL_INCOMPLETE\`** y NO se presenta como un PASS científico.

El prerequisito G0 fue verificado en el run original (3/4). No se vuelve a usar DEV para escoger réplicas o umbrales. **Todas las cuatro réplicas se incluyen siempre**, también la réplica 2 más débil. Los controles G1/G2 originales siguen NEGATIVOS; ni un PASS de confirmación los convierte en positivos.

## Informes e independencia

La validación se realizará en un workflow GitHub Actions separado: descarga desde **run_id fijo \`41685499939\`** de los artefactos de entrenamiento/manifest originales, verifica los SHA y la identidad exacta de código, ejecuta regresión OOD vieja antes de CONFIRM, carga exclusivamente pesos congelados, evalúa las cuatro réplicas y, en un job de auditoría sin PyTorch, aplica los gates y registra el resultado. Capturar run_id nuevo, commit SHA, checksums y salida completa antes de reclamar CONFIRM PASS.

**Límites científicos:** partición CONFIRM independiente en semillas, pero mismo generador, mismos tipos de operación y misma arquitectura programada. Un PASS confirmaría **replicación interna de composición funcional en mundos sintéticos**, NO generalización fuera de esa familia, ni invención autónoma de operaciones, ni superioridad del estimador de recompensa emparejado, ni utilidad comercial. No fusionar esta rama con \`master\` ni modificar producto.
