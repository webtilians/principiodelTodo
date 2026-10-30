# INFINITO F9-S — Protocolo preregistrado: parche causal de subespacios H/C asociados a la pista

**30-10-2026, congelado ANTES de entrenar proyecciones y ANTES de abrir confirmación.** Continúa F9-R (PR #25), con confirmación \`850000+\` ya consumida. No repetir F9-Q \`800000+\`, F9-R \`850000+\` ni abrir el holdout histórico \`900000+\`.

## Objetivo científico de INFINITO

**Separar tres factores** de la memoria funcional en LSTM:
1. **Representación** de identidad de pista en una dirección de H/C.
2. **Dinámica del controlador** / codificación de navegación y fase no específica de pista.
3. **Actualización temporal** necesaria en el último paso sin F4, antes de la bifurcación.

La **prueba de memoria útil no es la distancia entre estados ni el éxito individual**: es elegir ambos primeros brazos correctos de dos historias del mismo mundo con pistas originales/invertidas y transferir específicamente esa elección al **cue donante** mediante intervenciones, con F4 ausente.

F9-R observó que congelar H/C durante el blackout no rescataba ningún par, mientras restaurar H/C natural **antes del último paso** recuperaba algunos pares y restaurarlo justo en el cruce recuperaba la política natural por construcción. F9-S **no** elegirá el mejor checkpoint, componente ni instante usando una confirmación anterior.

## Datos, políticas y particiones cerradas

- Tres grupos de políticas congeladas × **cuatro índices de entrenamiento**: \`bc_anchor\` F9-O continuous6_after_bc y \`ppo_sham\` / \`ppo_blackout\` de F9-P. Los 12 checkpoints originales se verifican SHA-256 antes de actuar. Ninguna política, PPO, recompensa, LSTM o arquitectura se actualiza.
- **TRAIN solamente para direcciones de proyección:** semillas \`890000 + index*1000+j\`, j=0..79, índices 0..3, **320 mundos S13 pareados**. Se calculan direcciones **por checkpoint e índice** y un control ortogonal determinista **sin etiquetas** (solo se permite usar las pistas para ajustar los vectores de proyección). Nada de TRAIN usa \`895000+\`.
- **Confirmación prospectiva:** semillas \`895000 + index*1000+j\`, j=0..39, índices 0..3, **160 pares S13** por checkpoint/condición. Rango 895000..898039, completamente separado de TRAIN, F9-Q \`800000+\`, F9-R \`850000+\`, y holdout histórico **900000+ SIN ABRIR**. Las semillas \`895000+\` **no** se usan para seleccionar hiperparámetros, proyecciones o tratamientos.
- Se selecciona **una sola dirección cue H (64) y otra C (64)**, por grupo/semilla, sobre TRAIN usando el contraste promedio \`state(ball)-state(key)\` al inicio del **último frame antes del fork**, obtenido de prefijos con F4 normal. Se normaliza a norma 1. Sin PCA/rank tuning/regresión ni elección retrospectiva de dirección según retorno. Para control de navegación, un vector normal aleatorio con semilla fija se ortogonaliza respecto a cada dirección cue y se normaliza. Esta dirección ortogonal no se ajusta con ningún resultado de acción.
- No usar F9-Q probes entrenados 750000+ ni muestras confirmatorias viejas para definir el subespacio.

## Emparejamiento causal y momento de intervención

Para cada mundo de las dos particiones, las dos historias *key* vs *ball*:
- Mismas posiciones, orientación, paredes, observaciones públicas y secuencia física del profesor hasta la **primera acción divergente**. El LSTM procesa el historial mediante sus predicciones, que **NO** mueven la simulación durante ese prefijo.
- \`p=postcue\` es el primer frame posterior a la última observación pública de la pista. \`f=fork\` la primera acción docente divergente. \`l=f−1\` es el último frame anterior al fork, **fijado antes de inspeccionar la acción del modelo**.
- Todos los tratamientos con blackout ponen a cero exclusivamente las 512 entradas externas F4 desde \`p\` hasta el fork. La observación pública de 148 dimensiones queda intacta. **F4=0 también en fork y toda la continuación autónoma**, en todos los tratamientos. Encoder/almacén F4 originales nunca cambian.
- \`N_j(l)\` es el estado H/C de la trayectoria natural para la pista j justo antes de procesar el último frame, y \`B_j(l)\` el estado después del blackout con actualizaciones normales hasta entonces. Se **parchea una sola vez** en \`l\` y se deja que el modelo procese con F4=0 ese último frame. Se reconstruyen ambos mundos y se miden desde el fork.
- \`P_w(target,source)=target+<source−target,w>w\` sustituye solo la componente de un estado paralela a una dirección cue aprendida, por la del donante.
- \`P_u\` parchea una **dirección ortogonal**, con norma de la perturbación igual a la del parche cue para el mismo estado, y signos fijos derivados de la proyección cue. Los controles se interpretan con cautela: igual L2 no iguala distribución ni función.

## Condiciones congeladas: ninguna selección post-hoc

1. \`normal_reference\`: historial F4 íntegro, en el cruce F4=0, control de que la política sabe escoger.
2. \`blackout_update\`: F4 borrada desde postcue, H/C se actualizan con normalidad.
3. \`full_last_same\`: desde \`B_j(l)\`, restaurar H/C natural **completo** \`N_j(l)\` antes del último frame (referencia de rescate F9-R, NO nueva hipótesis principal).
4. \`blackout_cue_axis_same\`: parchear a \`B_j(l)\` solamente los **dos coeficientes 1D cue** H y C desde \`N_j(l)\`. ¿Es suficiente ese código para recuperar decisiones?
5. \`blackout_cue_axis_opposite\`: parchear solamente esos coeficientes desde \`N_{¬j}(l)\`. ¿Cambia la salida hacia el cue donante?
6. \`blackout_orthogonal_same\`: perturbación sobre los dos ejes ortogonales, con **igual magnitud L2** que el parche de cue de #4. Control contra perturbación genérica.
7. \`full_last_axis_opposite\`: desde \`N_j(l)\` restaurado completo, sustituir solamente los coeficientes cue H y C por \`N_{¬j}(l)\`. Control de **suficiencia causal** del eje en un controlador temporalmente funcional.
8. \`full_last_orthogonal_opposite\`: desde \`N_j(l)\`, alterar solo dos direcciones ortogonales (misma norma L2 que #7), control de especificidad.
9. \`full_last_axis_clamp_mean\`: desde \`N_j(l)\`, fijar los dos coeficientes cue a la **media del par**. Control de **necesidad local** sin invertir directamente.
10. \`full_last_axis_swap_H\`: como #7, solo dirección cue de H.
11. \`full_last_axis_swap_C\`: como #7, solo dirección cue de C.

Para cada uno de los 11 tratamientos, evaluar los dos cues en un mundo dado, tanto con estado \`intact\` como con \`swap_HC\` completo entre historias, que aporta auditoría de **transferencia al donante** y dependencia de memoria. Registrar primera acción, primer compromiso espacial (o ninguno), primer acierto del destinatario y donante, denominadores completos, pasos de blackout y controles de igualdad exacta de O y replay.

## Criterios preregistrados, separados

**Primario A — recuperación de acceso:** \`blackout_cue_axis_same\` debe superar a \`blackout_update\` en pares con **ambas primeras salidas correctas** y en pares con transferencia al donante bajo \`swap_HC\`, con **aumento estrictamente positivo en ≥3/4 semillas y agregado** para BC y/o PPO sham. Contrastar adicionalmente contra \`blackout_orthogonal_same\` (misma norma de perturbación); si este control rinde igual, no interpretar el parche cue como específico.

**Primario B — causalidad del código en controlador funcional:** \`full_last_axis_opposite\` debe provocar ambos **cambios de salida correctos hacia la pista donante** frente a \`full_last_same\`, en ≥3/4 semillas y mayor total que \`full_last_orthogonal_opposite\`. La condición base debe tener **ambas primeras salidas correctas** y el parche invertido debe **invertirlas hacia el donante en ambos cues**, con idénticos 148 públicos y F4=0. Si falla, el eje lineal no identifica por sí solo el lector causal, aunque pueda decodificar la pista.

**Secundarios predefinidos:** H vs C aisladamente, clamp a la media y respuesta al parche opuesto directamente sobre blackout. Informar todas las once condiciones sin ocultar modelos o semillas. **No** declarar mecanismo plenamente identificado salvo que A y B muestren consistencia; un B positivo con A negativo significa que el eje puede controlar una política con estado funcional, pero no basta para recuperar un controlador deteriorado durante el blackout.

## Integridad y límites

- Testear que los vectores están entrenados solo en \`890000+\`, normas =1, ortogonales (|dot|<1e−5), sin fuga de muestras. El control de eje positivo \`P_w(x,x)=x\` y los offsets del último frame se comprueban.
- **Gate separado**: comprobar todos los 12 SHA-256, los 24 vectores cue/orth por componente antes de confirmar. Si algún problema se detecta antes del nuevo conjunto, enmendar y reiniciar solo calibración; no convertir la confirmación en depuración.
- Bajo \`full_last_same\`, reproducir exactamente el estado de control F9-R de restauración en último paso en fixtures **TRAIN**; no volver a abrir F9-R confirmación. Los resultados nuevos con esa condición son propios del conjunto 895000+.
- Posibilidad de que el «eje cue» contenga correlaciones con fase del controlador, y de que haya interacciones H/C no lineales: el contraste en parejas con misma geometría y control ortogonal limita, pero no elimina, confusión.
- F9-S estudia un **fork alcanzado mediante prefijo del profesor** y acciones autónomas posteriores; no equivale a navegación autónoma completa ni prueba de conciencia, autoconciencia o memoria humana. Ante un resultado negativo, no ajustar sobre \`895000+\` ni tocar \`900000+\`: toda nueva hipótesis requeriría otro conjunto verdaderamente virgen.
