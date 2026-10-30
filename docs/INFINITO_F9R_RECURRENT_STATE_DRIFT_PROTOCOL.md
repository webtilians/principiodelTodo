# INFINITO F9-R — Protocolo preregistrado: ¿las actualizaciones H/C inutilizan el recuerdo durante un blackout de F4?

**Fijado antes de cualquier evaluación F9-R el 30-10-2026.** Continúa F9-Q (PR #24), cuya nueva confirmación \`800000+\` quedó **consumida**: con F4 disponible durante el historial y suprimido en la bifurcación, la identidad de la pista dirigió causalmente las decisiones; al suprimir F4 diez pasos antes, H y C aún permitían decodificar la pista pero **0/160 pares** eligieron ambas salidas correctamente. Este nuevo experimento no reajusta F9-Q ni vuelve a usar su partición, y **no abre** el holdout histórico \`900000+\`.

## Pregunta precisa

Con **dos historias originales/invertidas de la misma geometría, misma secuencia de acciones del profesor, mismo instante de bifurcación y F4=0 en decisión y continuación**, ¿la pérdida del acceso funcional a la identidad ocurre por las **actualizaciones recurrentes del controlador mientras F4 está ausente** (degradación/reescritura), o es suficiente la ausencia externa actual para que el lector de la política deje de usar lo almacenado?

No suponer ninguna de las dos. Un efecto de H/C en el brazo elegido es un efecto causal del estado sobre la política, pero también puede transportar información de navegación; por eso se exige **transferencia contrafactual específica de la pista donante** y pares con ambas primeras decisiones correctas, no retorno de la tarea.

## Solo modelos congelados — ninguna política, reward, PPO ni probe nuevos

Reutilizar **las cuatro semillas** de cada uno de estos tres checkpoints, idénticos y verificados por SHA-256 desde F9-O y F9-P:
- \`bc_anchor\`: F9-O \`continuous6_after_bc.zip\`, CI #40957255987.
- \`ppo_sham\`: F9-P \`ppo_sham_after_rl.zip\`, CI #40960267485.
- \`ppo_blackout\`: F9-P \`ppo_blackout_after_rl.zip\`, CI #40960267485.

**12 modelos congelados**; misma arquitectura (context8, entrada 660, política LSTM 64/64, 380552 parámetros). No se utiliza el probe lineal para seleccionar acciones ni se entrena otro. F9-Q ya aporta la evidencia de señal H/C decodificable, que sirve de **motivación histórica, nunca de evaluación en el nuevo conjunto**.

## Un cruce, dos historias, trayectorias idénticas

Reutilizar \`matched_pair\` de F9-Q para obtener acciones docentes y dos observaciones de pistas opuestas hasta la **primera acción divergente del profesor**. El profesor mueve solo el simulador durante el prefijo, mientras la LSTM ve únicamente las observaciones públicas y actualiza su estado con \`model.predict\`; las predicciones de política no controlan la posición del prefijo. En cada instante y en todas las condiciones, las acciones, observaciones públicas, geometría y orientación *para el mismo cue* son idénticas. **En el cruce la observación pública de 148 elementos también es idéntica entre cues; la lectura F4 (512 elementos) es cero en TODOS los grupos y durante toda la continuación autónoma**.

Sea \`p = post_cue\` (primer frame tras la última observación pública de la pista), \`f=fork\` (primera acción docente contrafactual distinta), \`m=p+floor((f-p)/2)\` (mitad) y \`l=f-1\` (último frame anterior a decisión). F9-Q observó \`f-p=10\` en confirmación anterior, pero **no se utilizará ese valor para filtrar ni modificar** mundos F9-R. Se verifica \`p<m<l<f\`. La máscara F4 se aplica desde \`p\` en todos los grupos blackout, nunca sobre 148 públicos. Actor siempre predice durante prefijo, sin reward ni etiquetas docentes.

## Intervenciones de estado (todas preespecificadas)

Sobre cada checkpoint, mundo y cue, evaluar **las diez** condiciones, con las mismas observaciones públicas y F4=0 en la decisión:

1. \`normal_reference\`: prefijo natural con F4 disponible, para verificar el circuito de acción sin F4 en decisión.
2. \`blackout_update\`: F4 anulado durante \`p..f\`, se actualizan H/C normalmente; control negativo de F9-Q.
3. \`blackout_freeze_HC\`: F4 anulado; se llama \`predict\` pero **descarta las nuevas H y C** durante \`p..f-1\`, preservando el estado justo antes del apagón.
4. \`blackout_freeze_H\`: durante bloqueo conserva H anterior pero acepta actualizaciones de C.
5. \`blackout_freeze_C\`: conserva C anterior, acepta actualizaciones de H.
6. \`blackout_patch_normal_mid_HC\`: al instante \`m\` reemplaza **una sola vez** H y C por su valor de trayectoria natural **del mismo cue y tiempo**, y continúa con F4=0 y actualizaciones habituales.
7. \`blackout_patch_normal_last_HC\`: igual, una sola intervención en \`l\`, antes de procesar ese frame, luego actualiza con input masked.
8. \`blackout_patch_normal_fork_HC\`: al cruce reemplaza H y C por los de la trayectoria natural al mismo instante; es **control positivo de instrumentación** (debe reproducir exactamente \`normal_reference\`), no evidencia independiente de rescate.
9. \`blackout_patch_normal_fork_H\`: conserva C de trayectoria blackout y sustituye solo H natural en fork.
10. \`blackout_patch_normal_fork_C\`: conserva H blackout y sustituye solo C natural en fork.

Cada condición produce dos estados H/C (cue key/ball). Se evalúa una continuación autónoma desde el **mismo cruce** tanto con \`intact\` como con \`swap_HC\` (intercambio de H/C con historia cue opuesta). Se registra primera acción, primera salida espacial o \`none\`, llegada, timeout, correspondencia con **destinatario y donante**. Los estados solo intervienen mediante \`model.predict\`, no por mover actor o cambiar geometría/reward.

## Métricas y criterios congelados

**Resultado central (cada condición):** en 160 pares por checkpoint (**40 pares por semilla ×4**), contar pares con **ambas primeras salidas correctas** bajo \`intact\`, y pares con **ambos brazos transferidos hacia la pista donante al hacer \`swap_HC\`** respecto al \`intact\`. Los pares sin elección permanecen en el denominador. No se usa éxito individual como sustituto.

**Hipótesis primaria de mecanismo:** en modelos BC y PPO sham, \`blackout_freeze_HC\` produce una **ganancia estricta frente a \`blackout_update\`** en pares con ambos primeros brazos correctos **y** pares con transferencia específica del donante, en al menos **3/4 semillas y en el agregado**. Esto indicaría que impedir actualizaciones durante la ausencia externa **preserva algún estado necesario para una decisión cue-dependiente**, sin identificar todavía dimensión o función de memoria exclusiva. No se exige que \`ppo_blackout\` rescate, pero se informa por igual.

**Secundarios:** comparar \`freeze_H\` vs \`freeze_C\`, reinstalación natural en mitad vs último paso, y restauración solo H vs solo C en fork para localizar qué parte del controlador es funcional. Si solo \`patch_normal_fork_HC\` funciona, el resultado es consistente con degradación durante el gap, pero esa rama reproduce el control \`normal_reference\`, no prueba suficiente del mecanismo por sí sola. Si \`freeze_HC\` no rescata, puede significar que el estado temporal es necesario para interpretar la escena o que el apagón desordena una representación. **No** afirmar imposible la memoria.

Verificaciones: \`normal_reference\` vs \`patch_normal_fork_HC\` deben producir primeras acciones y salidas **idénticas caso por caso**; los prefijos deben tener exactamente los mismos 148 públicos, mismo número de frames enmascarados, estado del encoder intacto, checkpoint SHA exacto, \`m\`/\`l\` fijados por tiempo y visibilidad de pista (no por reward, predicción ni efectividad), F4 ausente en decisión y continuación. Si no se cumplen, abortar **antes** de abrir confirmación.

## Partición confirmatoria nueva e independiente

**F9-R**: \`850000 + index*10000 + j\`, \`index=0..3\`, \`j=0..39\` → 160 mundos S13 × dos pistas contrafactuales. Este intervalo está separado de F9-Q confirmación \`800000 + index*10000+j\` (última 830039), de sus probes TRAIN 750000+, F9-P 700000+, F9-O 600000+ y F9-M/N. **No se prueba ni utiliza para corregir el código antes de congelar tests**; tests sobre fixtures TRAIN nuevos en \`840000 + index*10000+j\`, j=0..9, rango separado de la confirmación. \`850000+\` es **nueva confirmación, no train**; fuera de todas las particiones precedentes y **sin tocar \`900000+\`**.

Pipeline: tests de intervención temporal en fixtures \`840000+\` → gate de los 12 checkpoints originales por hashes → solo entonces primera apertura de \`850000+\` → agregado completo por 4 semillas y 3 modelos → conservar artefactos de cada par y reporte JSON. Si ocurre un error tras abrir confirmación, la partición queda **consumida**, y un rerun reparado se marca exploratorio, nunca nueva confirmación.

## Interpretación honesta

Estudiamos un mecanismo local, con llegada al fork físicamente controlada por prefijo docente; NO medimos autonomía de ruta completa ni conciencia ni memoria humana. El \`swap_HC\` puede transferir correlaciones de controlador, pero aquí los cues contrafactuales comparten geometría y observación; exigir transferencia hacia la **identidad del donante** reduce explicaciones triviales. La manipulación de H/C en estados no naturales puede salir de distribución, registrar esa limitación. Este experimento no permite afirmar que un «recuerdo» sea una unidad neuronal única: H/C son estados distribuidos, se restaura el estado natural sin reajustarlo.
