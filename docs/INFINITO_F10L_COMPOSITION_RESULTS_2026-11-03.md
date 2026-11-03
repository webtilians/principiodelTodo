# INFINITO F10-L — composición cero-shot de primitivas recompensadas: PASS en OOD preregistrado

**Fecha de ejecución: 3 de noviembre de 2026.**
**ESTADO CIENTÍFICO OOD: \`F10L_COMPOSITION_PASS\`.**
**ESTADO TÉCNICO: fases 1 y 2 + auditorías independientes SUCCESS.**

> **Importante:** esto es un **PASS en OOD prospectivo**, NO una confirmación externa. **CONFIRM \`23300000+\` permanece sellada.** No llamar «confirmado» a la hipótesis hasta preregistrar y superar una confirmación adicional en esa partición sin retuning. No hay evidencia de invención espontánea de operaciones: la arquitectura, las interfaces entre habilidades y las acciones disponibles fueron especificadas por el investigador.

## Procedencia reproducible

- **[Prerregistro F10-L](INFINITO_F10L_PRIMITIVE_CREDIT_PREREG.md)** commit **\`7fc030e2d5a119e9d9895b3c96c0e5dbd16561ad\`**, creado **ANTES de cualquier generador, entrenamiento o evaluación F10-L**. Preguntas, semillas, splits, presupuestos y gates no se modificaron tras observar resultados.
- **[Actions #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939)** HEAD del run \`7714a8fa96884b15724e0bd569e7f9e9bcc015f4\`, 4 entrenamientos y 4 evaluaciones DEV exitosos, un auditor de prerrequisito G0 exitoso, **y solo entonces** 4 ejecuciones OOD de composición usando SHA congelados y el auditor final. Todos los jobs técnicos marcaron SUCCESS. El flujo de trabajo tenía un bloqueo real \`needs.g0_gate.outputs.eligible == 'true'\`: con G0 negativo no habría generado OOD.
- Artefactos de la ejecución: \`f10l-stage1-0..3\` contienen **20 checkpoints SHA originales**, curvas, \`train.json\` y \`dev.json\`; \`f10l-independent-g0-original\` contiene \`f10l-g0.json\`; \`f10l-original-ood-0..3\` contiene predicciones congeladas \`f10l-ood-R.json\`; \`f10l-independent-final-gates\` contiene \`f10l-final.json\`. Dos auditorías separadas **sin carga de PyTorch ni reentrenamiento**.
- Cada actor entrenó en **8 épocas × 4096 estados × 2 acciones ejecutadas = 65.536 episodios**; 512 actualizaciones. Cinco brazos × cuatro réplicas, mismas observaciones y misma inicialización dentro de cada tipo de actor. Entrenamiento XOR en \`23100000+\` y selección en \`23105000+\`, DEV en \`23200000+\` y \`23205000+\`; OOD composición \`23400000+\` solo tras G0. **\`23300000+\` CONFIRM nunca abierta**.
- DEV 512 predicciones por brazo/réplica, 2.560 por réplica; OOD 512 casos por réplica, 2.048 decisiones compuestas, cada grupo de 4 variantes contrafactuales del mismo mundo se mantiene junto en 2.000 bootstrap. Auditoría etapa 1 verificó **36.864 SHA de identidad de caso** TRAIN+DEV (no son 36.864 observaciones semánticamente distintas); etapa 2 verificó 2.048 hashes OOD nuevos y SHA originales.
- Producto en \`master\` y \`src/infinito3\` intocados; benchmarks de Bemula-Lite, E10 y datos anteriores F10-I/J/K no utilizados en nuevos datasets F10-L. PR en borrador, sin merge.

## Preguntas y cinco condiciones

- **X-A \`xor_onpolicy\`**: 2 acciones Bernoulli realmente ejecutadas, REINFORCE estándar con baseline EMA de retornos previos.
- **X-B \`xor_two_actions\`**: ejecuta tanto acción 0 como 1 para el MISMO contexto; aprende con \`−(R_1−R_0)×p(acción 1)\`. **Esta doble intervención en un XOR binario determinista es informacionalmente equivalente a mostrar la etiqueta verdadera**. No llamarla aprendizaje autónomo sin señal correctiva.
- **S-A \`sel_onpolicy\`**: 2 slots muestreados on-policy con REINFORCE.
- **S-B \`sel_uniform_credit\`**: 2 slots uniformes distintos realmente ejecutados, gradiente de logπ ponderado por retorno/baseline; es un estimador off-policy sesgado, control de exploración.
- **S-C \`sel_uniform_pair\`**: MISMAS parejas uniformes realmente ejecutadas que S-B, pero objetivo de preferencia por diferencia de retornos. No se pasa índice de ganador como target. El entorno sí devuelve éxito/fracaso **localmente por elegir una versión acreditada**, lo que simplifica mucho el crédito comparado con F10-K.

XOR usa una red tanh 2→16→1 con (bit,flip), SELECT un scorer lineal sobre metadatos **ya estructurados** de firma/tiempo/firma×tiempo/fuente. Las políticas se congelan independientemente. El investigador cablea SELECT→lectura del slot→XOR; **ningún ejemplo combinado entra al TRAIN de ninguno de los dos actores**. OOD aumenta de 5/7 a 9/11 versiones e introduce nuevos patrones de fuente.

## Fase 1 — habilidades en DEV, auditoría de prerrequisito

| Réplica | XOR on-policy | XOR ambas acciones | Selector on-policy | Selector uniforme/EMA | Selector uniforme/pares | ¿G0? |
|---|---:|---:|---:|---:|---:|---|
| 0 | 100% | **100%** | 100% | 100% | **100%** | PASS |
| 1 | 100% | **100%** | 100% | 100% | **100%** | PASS |
| 2 | 100% | **100%** | 75,20% | 66,60% | **75,20%** | FAIL |
| 3 | 100% | **100%** | 100% | 100% | **100%** | PASS |

**G0 PASS 3/4**, y DEV bootstrap lower bounds XOR>90% y SELECT>75% en estas tres réplicas. Por decisión preregistrada, la auditoría **autorizó abrir composición OOD**. XOR X-B aprende las **4 de 4** combinaciones con 100% en las cuatro réplicas: el bloqueo previo de F10-K en 3/4 de XOR se ha superado, pero hay **múltiples cambios simultáneos** respecto a F10-K (descomposición, reward local, mayor presupuesto y distinta tarea de entrenamiento); no aislar una única causa.

**G1 ventaja de ejecutar dos acciones en XOR: FAIL 0/4.** X-A también alcanza 100%, de modo que X-B−X-A =0 e IC95 [0,0] en las cuatro. **No hay evidencia de superioridad del algoritmo emparejado en esta tarea/budget.**

**G2 ventaja de crédito por pares en selección frente a control de exploración uniforme: FAIL 1/4.** En tres réplicas S-C−S-B=0, IC95 [0,0]. En réplica 2 S-C supera S-B por **+8,59pp**, IC95 agrupado [**+5,66**, **+11,91**]pp. Una réplica sola no cumple el umbral preregistrado ≥3/4, por lo que no se ha demostrado una mejora estable del estimador emparejado.

## Fase 2 — composición zero-shot OOD sin actualización de pesos

Tarea NUEVA: elegir última versión firmada, leer exactamente ese slot e interpretar con XOR flip=1. Ni X-A/X-B ni S-A/B/C entrenaron composición memoria+flip=1.

| Réplica | Último firmado seleccionado | Composición correcta | IC95 composición por mundo |
|---|---:|---:|---|
| 0 | **100%** | **100%** | [100;100]% |
| 1 | **100%** | **100%** | [100;100]% |
| 2 | **85,16%** | **86,91%** | [83,20;90,43]% |
| 3 | **100%** | **100%** | [100;100]% |
| **Total descriptivo** | **96,29%** | **96,73% (1981/2048)** | Ver IC por réplica (no combinar como datos independientes) |

El peor estrato preregistrado en réplica 2 fue \`S_after\` **71,09%**, mayor que el mínimo 70%; los estratos de procedencia y 9/11 registros también pasaron los mínimos. **G3 COMPOSICIÓN ZERO-SHOT PASS 4/4.** El control simbólico último firmado+XOR alcanza 100% por definición; el resultado aprendido no debe confundirse con su código.

### Ablaciones causales de F4 y del operador

| Réplica | Normal | Valores F4 permutados | Flip desactivado | Caída F4 | Caída flip |
|---|---:|---:|---:|---:|---:|
| 0 | 100% | 47,07% | 0% | **52,93 pp** | **100 pp** |
| 1 | 100% | 47,66% | 0% | **52,34 pp** | **100 pp** |
| 2 | 86,91% | 44,34% | 13,09% | **42,58 pp** | **73,83 pp** |
| 3 | 100% | 45,90% | 0% | **54,10 pp** | **100 pp** |

En las cuatro réplicas, las caídas superan los umbrales preregistrados de 20pp y sus **IC95 agrupados tienen límite inferior >0**; en la réplica más débil F4 drop IC95 [35,55;49,80] pp y flip drop IC95 [66,41;81,25] pp. La inyección de slot ganador solo como **diagnóstico evaluador** rescata la composición de la réplica 2 a 100%; esto localiza sus fallos en selección más que en XOR. **G4 REUTILIZACIÓN CAUSAL PASS 4/4**.

### Estado científico

- G0 **adquisición previa de XOR+selección**: **PASS 3/4** (permiso de abrir OOD).
- G1 **XOR paired > on-policy**: **FAIL 0/4**.
- G2 **selector paired > uniforme/EMA**: **FAIL 1/4**.
- G3 **composición zero-shot en OOD nuevas**: **PASS 4/4**.
- G4 **dependencia causal simultánea de F4 y flip**: **PASS 4/4**.
- Global original **\`F10L_COMPOSITION_PASS\`** según contrato G0∧G3∧G4.

## Lo que SÍ y NO concluye F10-L

**Sí:** una arquitectura modular puede adquirir por recompensas dos habilidades útiles por separado y **transferirlas sin actualizar pesos** a un problema nuevo compuesto que exige utilizarlas conjuntamente, en un simulador explícito de firma/tiempo, con transferencia 5/7→9/11 y variaciones de fuente. Las ablaciones refutan que se resuelva sistemáticamente ignorando la memoria o el operador. Es evidencia de **reutilización composicional de habilidades aprendidas**, bajo estructura facilitadora y recompensa local informativa.

**No:** que el agente haya inventado el operador XOR, las interfaces, la secuencia SELECT→READ→XOR, una política general de memoria, un nuevo algoritmo de optimización mejor que REINFORCE, ni aprendido a decidir en situaciones abiertas cuándo recuperar o conservar recuerdos. Ni demuestra transferibilidad al producto/usuario real. La tarea XOR tiene cuatro entradas discretas y feedback muy instructivo; SELECT observa metadatos perfectos, con un scorer lineal y una recompensa directa de validez, distinta de F10-K. Los tres 100% no equivalen a una prueba amplia de generalización. La réplica 2 muestra fragilidad del selector incluso con el mismo entrenamiento.

## Próxima decisión

**Congelar F10-L.** No tunear los 20 modelos, sus hiperparámetros ni los umbrales después de OOD. **Confirmación \`23300000+\` pendiente y sellada**: antes de abrirla, preregistrar un procedimiento independiente con pesos SHA congelados, mundo más complejo o combinación inversa, criterios y falsaciones. Como siguiente objetivo científico, preguntar si el agente **aprende a descubrir o elegir qué operaciones encadenar**, no solo ejecuta una composición que el investigador ha cableado. Producto independiente: proponer integración solamente después de validaciones pertinentes y revisión de seguridad.
