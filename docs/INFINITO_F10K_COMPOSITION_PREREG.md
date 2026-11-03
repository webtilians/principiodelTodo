# INFINITO F10-K — primitivas aprendidas por recompensa y composición cero-shot

**Prerregistro prospectivo, 2026-11-03, escrito antes de implementar F10-K o abrir TRAIN/DEV/OOD.** F10-J terminó FAIL: forzar experiencias READ no enseñó a seleccionar ni interpretar y ni siquiera se aprendió consistentemente el bit público. F10-K formula una pregunta más pequeña: **¿puede aprenderse por recompensa un conjunto de habilidades elementales y reutilizarlas juntas, sin ejemplos de la composición?**

## Tarea, estructura y límites

Tres habilidades entrenadas, con recompensa observada de la acción ejecutada (+1 correcta, -1 errónea), sin pérdidas por etiqueta correcta ni índice ganador en los brazos RL:

- **P / procesar**: con bit público p y bandera flip (0 o 1), responder p XOR flip. Fuente pública: ningún registro F4.
- **I / interpretar**: observar el valor de un único registro F4 y responder ese valor (flip siempre 0). Aísla interpretación de la búsqueda.
- **S / seleccionar**: entre informes firmados/no firmados con timestamps, seleccionar un registro y responder el valor recuperado (flip siempre 0). El último informe firmado por timestamp es válido; la política no recibe su índice.
- **M / composición desconocida**: seleccionar el último firmado, leerlo y aplicar flip=1. Durante TRAIN y DEV solo se presentan P/I/S, **nunca fuente memoria + flip=1**. En OOD M los tamaños y patrones de fuente son inéditos. No hay actualización alguna de parámetros en M antes de la evaluación.

La arquitectura recibe la estructura de tareas (fuente pública/memoria, flag flip) y la **canalización** selector→lectura→procesador la fija el investigador. Este experimento puede demostrar transferencia/composición dentro de un espacio de operaciones dado, **NO invención espontánea de nuevas operaciones ni políticas de cuándo hacer READ/SKIP**. El indicador de fuente no entra en el procesador compartido (solo bit observado y flip). En memoria, los valores están ocultos al selector, y solo el valor del registro elegido entra en el procesador. La recompensa binaria determinista informa indirectamente sobre la verdad; no afirmar ausencia completa de corrección.

## Cuatro brazos y experiencia igualada

Política compartida: selector por registro sobre firma, tiempo normalizado, interacción firma×tiempo y fuente; procesador MLP compartido sobre bit y flip. Bernoulli de acción, Categorical de slot (solo S). Muestreo on-policy, REINFORCE con EMA de recompensas pasadas por tipo de primitiva, entropía 0.002, Adam 0.003, clip 1.0, torch CPU determinista. **No usar winner/truth como targets de A/B/C; solo el entorno los utiliza después de actuar.**

- **A joint_shared**: los mismos ejemplos P/I/S, mezclados aleatoriamente en cuatro pasadas.
- **B staged_shared**: exactamente el mismo multiconjunto de episodios, pero con fase 1 P+I (2048 de cada tipo), fase 2 S (4096) y fase 3 repaso mezclado P/I/S (6144,6144,4096 respectivamente). 24.576 episodios por réplica en A/B/C/D, **8192 por tipo P/I/S**, 384 pasos de 64. Mismos casos, pesos iniciales emparejados y presupuesto. B es el brazo de hipótesis principal.
- **C staged_unshared**: mismo orden y experiencias de B, pero dos procesadores disjuntos para entradas públicas y de memoria. Es un control de arquitectura con mayor número de parámetros; comparar con cautela, no atribuir las diferencias únicamente al orden.
- **D supervised_ceiling**: actor compartido de A/B, mismas experiencias, pérdida supervisada de respuesta y winner; techo de capacidad **privilegiado**, no competidor RL equivalente.

El orden de tareas y el mecanismo de peso compartido son intervenciones; las tareas primitivas nunca incluyen la combinación M. No hay currículum escogido mirando DEV ni mejores checkpoints seleccionados.

## Mundos y controles de correlación

4 réplicas r=0..3; TRAIN 512 mundos base por cada P/I/S, cuatro variantes por mundo (2048 casos/tipo/réplica; 6144 estados TRAIN). DEV 64 mundos por cada P/I/S ×4 (768 casos/réplica). OOD composición M 128 mundos ×4 (512 casos/réplica). TRAIN seeds 22100000+10000r+2000*type+j; DEV 22200000+10000r+2000*type+j; OOD 22400000+10000r+j; **CONFIRM 22300000+ sellada y no accesible en generador**. Seeds diferentes por réplica y split. El evaluador solo genera DEV/OOD **DESPUÉS** de escribir y SHA-verificar todos los checkpoints de una réplica. Total 4×4=16 checkpoints; 16×(768+512)=**20.480 filas** de evaluación; 4×(6144+768+512)=**29.696 hashes**.

Para S: TRAIN 5 o 7 informes de la clave, 2/3 firmados y un distractor unsigned más reciente que el último firmado, y un unsigned más reciente todavía que el candidato X, incluso si X se firma. OOD M: 9/11 informes, 4/5 firmados, más actualizaciones contradictorias y procedencias OOD correlacionadas de cuatro modos. Almacén permutado respecto a timestamps; ganador variable por posición. Contraejemplos de firma X y swap de timestamp originan cuatro variantes U_before,U_after,S_before,S_after; solo S_after cambia la verdad. Para generar una señal de selección identificable, el ganador y los demás valores de slot se configuran en oposición binaria en las muestras ordinarias; el actor no ve los otros valores antes de escoger. Esta es una **simplificación fuerte**, no una prueba de evidencia semántica general.

P genera las cuatro combinaciones de bit público y flip por mundo, con valor balanceado. I genera dos valores distintos y dos fuentes/ubicaciones. En I se suministra un único registro candidato, aunque el almacenamiento físico pueda variar. En OOD M flip siempre 1, valores ganadores 0/1 balanceados por mundo, cuatro fuentes (IID, firma asociada fuente0, firma asociada fuente1, alternancia), longitudes 9/11 cruzadas con fuentes. Nunca se entrena en combinación memoria+flip1. Registrar por réplica/condición/mundo/variante, selección, acción, bit, flip, longitud, procedencia, SHA y ablaciones.

## Reglas preregistradas de éxito

2000 remuestreos bootstrap por **mundo base**, manteniendo las cuatro variantes, semilla 227000+r. Auditor independiente de checkpoints, splits, distribución de tareas, balance de episodios y resultados. Controles: mismo actor con processor flip fijado a 0 en M; memoria de valores permutados solo dentro de registros candidatos (F4 shuffle), respuesta al azar, elección de último timestamp sin verificar firma, oráculo simbólico y control D supervisado.

- **G0 adquisición de primitivas**: B consigue DEV P>=90%, I>=90%, S>=80% y selección de winner S>=80% en >=3/4 réplicas.
- **G1 composición cero-shot**: B consigue OOD M>=80%, y límite inferior IC95 por mundo >65%; además >=70% en cada tamaño, modo de procedencia y variante, en >=3/4 réplicas. En ningún caso se entrena composición.
- **G2 reutilización causal**: B pierde >=20 puntos de precisión M tanto cuando se permutan valores F4 como al anular flip; IC95 por mundo inferior >0 en ambas ablaciones, en >=3/4 réplicas.
- **G3 ventaja del currículo**: B−A en M>=+5 puntos y límite inferior IC95 >0 en >=3/4 réplicas. Se informa por separado; **no es requisito** para demostrar composición (A podría también aprender y transferir).
- **G4 especificidad del procesador compartido**: comparación descriptiva B−C y C OOD, con diferencias de parámetros explícitas; no es requisito global.
- **SCIENCE_PASS=G0∧G1∧G2**, independiente del control privilegiado. Si G0 falla, no interpretar G1 como una prueba concluyente sobre composición; si G2 falla, no llamar reutilización funcional al porcentaje alto. No ajustar gates/algoritmo después de ver datos. Nombres finales F10K_COMPOSITION_PASS o F10K_COMPOSITION_FAIL.

**Integridad:** no modificar master, producto ni Bemula-Lite; no reutilizar F10-I/J mundos; no abrir confirmación 22300000+. PR de investigación en borrador. Cualquier próximo rediseño debe ser F10-L con mundos nuevos, no reparar científicamente F10-K después de OOD.
