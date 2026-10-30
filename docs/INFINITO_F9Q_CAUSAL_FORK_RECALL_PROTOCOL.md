# INFINITO F9-Q — Protocolo preregistrado: recuerdo causal en el cruce con F4 ausente

**Fijado antes de ejecutar la nueva confirmación (2026-10-30).** F9-P (PR #23) comprobó que entrenar PPO con blackout externo muy largo puede aumentar la llegada a la bifurcación y simultáneamente reducir a **0/160** el número de pares que toman correctamente las dos primeras salidas. F9-Q cambia el desenlace: **aislar la existencia, persistencia y uso causal de la información de la pista en la memoria recurrente H/C**. No optimizar navegación, rewards ni entrenamiento de políticas.

## Hipótesis causal exacta

**Mismo mundo S13, mismas geometría y secuencia de acciones controlada hasta la primera acción en la que el profesor se bifurca**, pero dos historias con pista inicial *key* vs *ball*. **En el instante de bifurcación, la observación pública de 148 dimensiones es idéntica**, se entrega **F4=0 (512 canales)** en todas las ramas del experimento, y se manipula exclusivamente el estado recurrente LSTM en ese instante.

Para que sea un efecto causal **del estado H/C sobre la política**, no basta con predecir la pista desde H. Se exige comprobar si **intercambiar H/C entre las dos historias cambia la primera acción o el primer brazo elegido**, con entorno y observación del destinatario idénticos. El profesor y las etiquetas no se presentan al actor.

## Tres checkpoints congelados y ninguna nueva política

Por cada una de cuatro semillas:
1. \`bc_anchor\`: política F9-O \`continuous6_after_bc.zip\`, SHA verificado de [F9-O #40957255987](https://github.com/webtilians/principiodelTodo/actions/runs/40957255987).
2. \`ppo_sham\`: PPO F9-P sobre ese mismo BC con F4 íntegra.
3. \`ppo_blackout\`: PPO F9-P sobre ese mismo BC con 112/128 pasos sin F4.

**No se entrena ni ajusta ningún actor/critic/LSTM.** Controlar hashes, arquitectura (380552 parámetros, entrada 660, 64 neuronas H y C), semilla y presupuesto previo antes de cualquier confirmación.

## Dos historias y dos regímenes de exposición

Reutilizar el **profesor nativo de siete acciones de F9-J/K/L** únicamente para fijar un mismo prefijo en el simulador. Las acciones del profesor pueden inspeccionar el entorno al construir la evaluación, pero **no afectan el H/C del actor** (el modelo predice sus propias acciones durante el prefijo para actualizar H/C, estas predicciones no mueven el entorno). El instante de decisión es la **primera acción distinta entre las dos rutas docentes**, una elección temporal cuya geometría y observación se comprueban idénticas.

- \`history_f4_normal\`: durante el prefijo, el agente puede leer F4; en la **decisión y toda continuación** F4 queda 100 % anulado.
- \`history_f4_postcue_off\`: desde el primer frame después de la **última observación pública de la pista inicial** hasta el cruce, F4=0 para el actor (writer y store intactos); en la decisión y toda continuación F4 permanece anulado. Las dos historias sufren igual calendario.

El actor ve la pista inicial por su **observación pública** y el escritor F4 funciona normalmente. La pista intercambiada invierte además el objetivo del simulador, manteniendo salida y otra geometría; se comprueba que los preﬁjos de acciones coinciden y que los 148 inputs en el instante de decisión son **byte a byte iguales**.

## Intervenciones cruzadas, con F4=0 en la decisión

A partir de \`state_A=(H_A,C_A)\` y \`state_B=(H_B,C_B)\` de historias de key/ball, en cada **entorno destinatario** A y B, aplicar una vez:

- \`intact\`: H y C propios.
- \`swap_HC\`: H y C de la historia con pista **contraria**.
- \`swap_H\`: sustituir solo H por el de la pista contraria, conservando C propio.
- \`swap_C\`: sustituir solo C por el de la pista contraria, conservando H propio.
- \`zero_HC\`: H y C a cero.
- \`zero_H\`: solo H a cero.
- \`zero_C\`: solo C a cero.

Cada rama reconstituye exactamente el mismo entorno del destinatario al inicio del cruce, sustituye H/C **antes de la primera acción**, anula **todos los canales F4 para el resto de la trayectoria** y deja las acciones posteriores en control autónomo. Registrar:
- la **primera acción** en la decisión (no equipararla sin más a elegir una salida);
- el **primer brazo espacial** realmente comprometido, o \`none\` si nunca elige;
- elección correcta según identidad de pista del **destinatario** y elección correcta según pista del **donante** (en \`swap_HC\`);
- pares con **ambas** primeras acciones de profesor correctas, **ambos** primeros compromisos correctos, y pares que cambian de salida cuando se intercambia H/C;
- conteos de alcanzabilidad/timeout de la **continuación autónoma** desde el mismo cruce forzado (NO confundir con éxito autónomo end-to-end);
- distancia H y C de ambas historias. **Una distancia grande sola no implica información sobre la pista.**

### Tres preguntas separadas

1. **Existencia / decodabilidad:** un probe lineal de identidad real \`key\`/ \`ball\` en H por separado y C por separado. Ridge **alpha 10** fijo, sin selección. Se ajusta **solo** sobre 80 pares **TRAIN de probe nuevos** S13 por semilla, \`750000 + index*10000 + j\`, \`j=0..79\`; ningún checkpoint cambia.
2. **Utilización natural:** bajo F4=0 en el cruce, \`intact\` resuelve las dos primeras acciones docentes y/o dos primeros compromisos en cada par, con los 40 pares nuevos S13 de esa semilla sin filtrar.
3. **Causalidad:** con observación pública y entorno idénticos, \`swap_HC\` cambia la elección hacia la **pista donante** en ambas historias, frente a \`intact\`; \`zero_HC\`, \`swap_H\` y \`swap_C\` son localizaciones/control de borrado, no prueba suficiente si no hay transferencia de elección contrafactual. No confundir correlación, intervención y memoria robusta.

Se publican todos los resultados, incluidos H/C decodificables sin acción correcta; predicciones dependientes de historia sin primera salida correcta; y control sesgado de lado fijo.

## Nueva confirmación bloqueada

\`800000 + index*10000 + j\`, con índices 0..3 y j=0..39: **160 mundos S13 con dos variantes originales/invertidas** (40 pares por semilla). Este rango es independiente del entrenamiento de probe \`750000+\`, de todos los entrenamientos anteriores (\`1000+\`, \`100000+\`, \`500000+\`, \`550000+\`), y confirmaciones \`300000+\`, \`400000+\`, \`600000+\`, \`700000+\`. El holdout histórico \`900000+\` **no se abre**.

**Primario preregistrado, contextualizado:** para cada checkpoint y régimen, contar \`intact\` con ambas primeras acciones docentes correctas y \`intact\` con ambos primeros **brazos** correctos; en paralelo, pares con **transferencia causal completa** donde al aplicar \`swap_HC\` ambas elecciones cambian y coinciden con el objetivo del *donante* (que es el opuesto al del destinatario). **Evidencia fuerte de memoria utilizable** requiere ambos: buen intacto y transferencia causal en **≥3/4 semillas** con mejora respecto a la línea cero; no basta con exactitud unilateral. La comparación antes/después de PPO es descriptiva, NO aleatorizada entre historias ya entrenadas: se presenta por semilla. El probe se evalúa exclusivamente con \`800000+\`, no se ajusta a ellos.

Se reporta la diferencia entre acciones correctas según profesor (teacher-forced en prefijo) y primeros brazos correctos (continuación autónoma desde el cruce). **Ningún resultado equivale a prueba de navegación autónoma completa**.

## Integridad y lectura honesta

1. Tests de máscara pública, replay de cue inversa, igualdad del prefijo, igualdad O en fork, forma H/C, estados intervenidos y primera salida, **solo** sobre TRAIN de probe o fixtures.
2. Gate independiente sobre 3 checkpoints ×4 semillas y probes entrenados **solo con 750000+**, antes de abrir confirmación. El probe no interviene nunca en la acción.
3. GitHub CI debe versionar resultados completos y agregar por semilla. Si se detecta un bug **después de abrir confirmación**, ese conjunto queda consumido; no puede retocarse y volverse a declarar confirmatorio.
4. Un \`swap_HC\` puede alterar navegación o estado latente en general; se exige **transferencia específica de la elección a la pista donante** como evidencia más precisa, y se advierte que swaps entre estados con distribuciones distintas pueden producir efectos fuera de distribución.
5. F9-Q no intenta validar conciencia ni lenguaje general ni agenciaría; mide un mecanismo de memoria a través del tiempo en una tarea pequeña. La prioridad es descubrir qué componente representa realmente la pista y **si causa la decisión**, no un mayor retorno de MiniGrid.
