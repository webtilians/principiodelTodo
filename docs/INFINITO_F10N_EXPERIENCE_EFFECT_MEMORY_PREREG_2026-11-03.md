# INFINITO F10-N — memoria de experiencias y efectos para sintetizar una secuencia jamás ejecutada

**PRERREGISTRO PROSPECTIVO, 2026-11-03.** Este archivo se publica antes de cualquier código, mundo o entrenamiento F10-N. F10-M quedó congelado como resultado científico negativo. No modificar habilidades F10-L, sus SHA ni el producto.

## Pregunta falsable

F10-M recibió recompensa al terminar sus programas, pero solo la absorbió en pesos. ¿Una **memoria explícita de intentos, resultados y cambios de estado observados** permite reconstruir efectos de operaciones y formular secuencias que nunca se ejecutaron? Separar recordar puntuaciones de aprender efectos.

Operaciones y catálogo IGUALES a F10-M: P carga dato público; S selecciona registro con modelo F10-L original congelado; R lee valor de slot seleccionado; X aplica XOR congelado; A termina. Catálogo: 85 cadenas enumeradas de longitud 0–3 sobre {P,S,R,X}, incluidas secuencias inválidas. **SRX (S→R→X) se reserva desde el principio y tiene CERO ejecuciones en TRAIN**, en cualquier contexto; quedan 84 candidatas de exploración. Puede ser propuesta por el planificador durante evaluación congelada. No se presenta como etiqueta del programa correcto ni se premia sin ejecutarla.

El agente recibe objetivo PUBLIC/MEMORY y FLIP0/FLIP1, bit público (señuelo independiente en MEMORY), firmas/tiempos/fuentes, y consecuencias observadas de sus acciones. **Nunca ve valor F4 antes de ejecutar R ni recibe índice winner/programa ganador como target.** Tras cada acción REAL se puede observar slot seleccionado sí/no, registro de trabajo unset/0/1, incremento de error de precondición y, después de READ válido, valor realmente leído. La recompensa final es +1/-1 por respuesta correcta, menos 0,01 por operación y 0,10 por precondición fallida. La memoria guarda episodios y transiciones con conteos y confianza; no almacena oracle truth para tomar decisiones.

## Modelo de efectos y planeamiento

La memoria de efectos aprende transiciones por frecuencias observadas, sobre estado abstracto seleccionado sí/no + registro unset/0/1. En la firma de transición puede entrar el bit público para P y el valor observado tras READ para R, pero el origen público/memoria del registro de trabajo **no entra en la firma**; ese sesgo de generalización es aportado por el investigador, no descubierto. Las operaciones, sus entradas, la instrumentación y la meta semántica del problema también están dadas.

El planificador examina las 85 secuencias y predice resultados con las transiciones aprendidas. Como aún no puede conocer F4, simula ambas hipótesis de bit seleccionado 0 y 1, y exige resultados correctos bajo ambas respecto a la meta PUBLIC/MEMORY XOR FLIP; penaliza programa largo, errores y transiciones desconocidas. No dispone de retorno de ningún programa no ejecutado. El programa es elegido al inicio, sin replanteamiento después de leer.

## Cuatro condiciones, presupuesto emparejado

Por réplica, TRAIN contiene tres contextos vistos: PUBLIC0, PUBLIC1, MEMORY0. Cada contexto tiene 256 mundos × cuatro variantes, total **3072 casos**. Exactamente **16 épocas × 3072 × 2 programas realmente ejecutados = 98.304 episodios** por condición/réplica. Cuatro réplicas independientes. Los programas de TRAIN se eligen entre los 84 autorizados, nunca SRX.

- **A / reward_table**: memoria explícita por contexto-programa de número de intentos, tasa de éxito, recompensa media y coste. En contexto nuevo usa solo programas ya ejecutados y estimaciones agregadas, sin simular operaciones.
- **B / additive_reward_table**: MISMAS experiencias de A, extrapola linealmente utilidad por programa desde PUBLIC0, PUBLIC1, MEMORY0, sin efectos observados. Programas jamás ejecutados no reciben score privilegiado.
- **C / learned_effects_uniform**: MISMO HISTORIAL de 98.304 programas ejecutados que A/B; memoriza transición observada acción-estado y planifica con un modelo de efectos construido exclusivamente de trazas. Puede proponer SRX nunca ejecutado.
- **D / learned_effects_directed**: MISMO total de 98.304 experiencias sobre los mismos mundos, pero elige en 50% de los intentos el programa de mayor cobertura/incertidumbre esperada entre 16 candidatos aleatorios según transiciones previamente experimentadas, y el otro 50% uniformemente. No consulta reward de programas no ejecutados. Usa exactamente el mismo planificador de C; D−C evalúa el efecto de dirigir la experimentación. Registrar acciones y costes reales.

A/B/C reciben idénticos intentos y rewards; D recibe los mismos mundos y presupuesto pero intentos distintos. Solo se entrena la memoria de efectos/tablas; los cinco checkpoints originales F10-L de cada réplica se descargan de GitHub Actions **run 41685499939** y se validan contra el manifiesto original G0 antes y después. No ajustar modelos F10-L ni F10-M.

## Semillas y ceguera experimental

TRAIN: 26100000 + 10000*r + 1000*t + j, t=0,1,2 para PUBLIC0/PUBLIC1/MEMORY0, j=0..255. DEV: 26200000 + 10000*r + 1000*t + j, j=0..63. OOD: 26400000 + 10000*r + j, j=0..127, SOLO MEMORY1 (nunca entrenada). **CONFIRM 26300000+ SELLADA**. Cada mundo contiene cuatro variantes contrafactuales U_before, U_after, S_before, S_after; solo la última altera verdad acreditada en MEMORY. TRAIN/DEV 5 o 7 informes; OOD 9 u 11 y cuatro modos de procedencia. La posición no indica la fecha. El bit público en MEMORY es señuelo independiente.

Guardar y SHA-verificar las cuatro memorias antes de GENERAR DEV/OOD. Auditor técnico independiente, sin PyTorch ni código de entrenamiento, verificará budgets, SHA originales, disjunción de hashes, programa SRX nunca probado, historial compartido A/B/C, versiones por mundo y 20.480 filas de evaluación (4 réplicas × 4 brazos × [768 DEV + 512 OOD]). Bootstrap IC95 con **2000 remuestreos por MUNDO BASE**, siempre agrupando sus cuatro variantes, seed 267000+100*r+offset. No cambiar ninguna regla después de ver DEV/OOD.

## Puertas preregistradas

- **G0:** C consigue >=90% DEV en CADA uno de PUBLIC0, PUBLIC1 y MEMORY0, al menos 3/4 réplicas.
- **G1:** C consigue >=80% OOD MEMORY1; límite inferior bootstrap IC95 por mundo >70%; mínimo >=70% para cada N, fuente y variante, al menos 3/4 réplicas. Además SRX nunca ejecutado en TRAIN; para reclamar composición el programa propuesto debe respetar el orden S antes de R antes de X.
- **G2:** Intervenir valores de F4 (shuffle) y desactivar XOR reduce precisión OOD de C >=20 puntos porcentuales cada intervención, IC95 del contraste límite inferior >0, al menos 3/4 réplicas.
- **G3:** C − B OOD >=15 puntos e IC95 inferior >0 en al menos 3/4, con exactamente mismos programas realmente ensayados.
- **G4 (diagnóstico, no obligatorio):** D − C OOD >=5 puntos e IC95 inferior >0 en al menos 3/4. Reportar cobertura de efectos y número de acciones ejecutadas, aunque FAIL.
- **SCIENCE PASS** solo si G0, G1, G2 y G3 pasan en al menos 3/4 réplicas. Si no, informar F10N_EFFECT_MEMORY_COMPOSITION_FAIL; si sí, F10N_EFFECT_MEMORY_COMPOSITION_PASS. SUCCESS técnico no implica éxito científico.

## Limitaciones declaradas antes de medir

La experimentación está estructurada artificialmente. El entorno proporciona observaciones del registro de trabajo y errores; ese acceso es más rico que la señal escalar de F10-M. El diseñador especifica catálogo, SOURCE/FLIP, meta XOR, forma de estados y búsqueda exhaustiva. Una mejora confirmaría síntesis composicional dentro de esa representación **construida por nosotros**, no invención libre de operadores, autorreflexión general ni transferencia al producto LLM. El programa SRX se reserva explícitamente como holdout: la tarea mide si se pueden recombinar efectos parciales en él, no si lo inventa fuera de un catálogo. No reutilizar OOD F10-M ni confirmaciones anteriores, ni cambiar reglas para rescatar un FAIL.

Mantener PR científica en borrador y producto master intacto; no abrir CONFIRM si F10-N fracasa.
