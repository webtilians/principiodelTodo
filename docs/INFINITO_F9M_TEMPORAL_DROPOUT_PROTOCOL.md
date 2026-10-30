# INFINITO F9-M — Protocolo preregistrado: aprender robustez a cortes temporales de F4

29 de octubre de 2026. **Diseño registrado antes de ejecutar entrenamientos o consultar resultados de confirmación.**

## Pregunta e hipótesis

F9-L comprobó que cortar el readout externo F4 en un modelo ya entrenado causa una caída de acierto S13 y que un probe H puede quedar descalibrado sin que el contenido haya desaparecido (S7). **F9-M entrena modelos nuevos** para averiguar si una política puede aprender a sobrevivir a esa falta de F4, sin destruir la navegación ordinaria. No inferir resultado F9-M a partir de F9-L.

**Predicción primaria:** frente al control de entrenamiento igualado, el entrenamiento con blackout de F4 elevará el acierto autónomo emparejado de S13 durante la supresión del readout tras observar la pista. Se registrará también navegación normal S7 y S13 para cuantificar el coste de robustez.

## Diseño preasignado: 2 brazos × 4 inicializaciones

- Dos políticas context8, **ambas nuevas** con arquitectura exacta de F9-I: entrada pública 148, F4 8 slots (8×8=64 activos, 448 padding hasta 512), actor LSTM 64, MlpLstmPolicy RecurrentPPO, siete acciones nativas, mismo selector contextual predefinido, pesos inicialmente iguales por índice, mismos mundos y actions de teacher, mismo número de parámetros.
- **Control \`sham\`:** se programa la misma ventana de blackout pero solo se ponen a cero **448 canales de padding ya cero**. Así ambos brazos tienen la misma infraestructura de máscara y reglas de activación, sin perturbación efectiva de datos en el control.
- **Tratamiento \`dropout\`:** en idéntica ventana se ponen a cero todos los 512 canales externos **solo en la observación suministrada al actor**, nunca en la memoria persistente, la selección o la observación pública.
- **Disparador entrenable sin oracle:** reloj interno de aumento de datos, armado tras ver *cualquier objeto key/ball detectado en el frame público* por el mismo encoder F4; dos observaciones después de **la primera detección pública de cualquier key/ball** comienza una ventana de longitud **2 o 4 pasos (probabilidad 50/50)**; en el 50 % de episodios la ventana está activada, en el resto no. RNG local preasignado por índice y contador de episodio, separado del RNG de PPO y del mundo. **No** inspeccionar \`cue_visible\`, posición del cruce, success_pos, ruta, etiqueta correcta, reward futuro o teacher durante la selección del blackout. Así la misma transformación se usa en imitaciones S7 y entrenamiento PPO S7. **Enmienda anterior a entrenamiento (CI de integridad falló):** el disparador inicial de «ningún objeto key/ball en pantalla» no llegó a activarse en las demostraciones porque pueden verse otros objetos key/ball; se sustituye por la ventana con retraso público fijo de dos observaciones. La nueva regla no examina etiquetas privilegiadas, no depende de resultados y se congela antes de abrir confirmación.
- **Mismos presupuestos** que F9-I: cuatro semillas (0–3), **600 mundos S7 train × 2 variantes de cue** por semilla, **40 épocas** de behavior cloning sobre exactamente las mismas acciones, **20.000 pasos PPO solicitados** por brazo (n_steps=128, batch_size=128, n_epochs=4, gamma=.99, gae_lambda=.95, ent_coef=.01, learning_rate=3e-4), sin early stopping por rendimiento. El rollout PPO deja de estar idénticamente emparejado tras divergir políticas; los pasos/hiperparámetros sí se mantienen iguales.
- Guardar pesos y SHA-256 de ambas políticas, comparativa exacta de parámetros, widths, BC y actual_rl_steps, distribuciones de máscaras y *effective masked observations*, tasas de éxito antes/después PPO. No modificar los checkpoints F9-I/J/K/L originales.

## Evaluación: una nueva partición de confirmación, NO heldout histórico

Los pares confirmatorios se fijan ahora en **20 nuevos mundos por semilla para S7 y S13** con semillas deterministas \`300_000 + index*10_000 + j\` (j=0..19, índice 0..3). Esta región no se solapa con train (1.000...) ni development (100.000...), ni con el histórico heldout (900.000...). Se reserva exclusivamente a este único contraste de F9-M; el **heldout histórico permanece cerrado**. No abrir mundos de confirmación antes de que estén completados y verificados los 8 modelos y sus presupuestos. No elegir ninguna semilla, cutoff o métrica en función del resultado confirmatorio.

**Métricas primarias (autonomía real, sin profesor):**
- S13 blackout tras la pista (mismo diagnóstico que F9-J: readout F4 cero una vez la pista ya fue vista y no está visible): comparar tasa de éxito **dropout – sham**, con pares de semillas/mundos iguales; informar por semilla y promedio.
- S13 *both_correct*: porcentaje de pares key/ball donde ambas salidas se eligen correctamente; evita confundir elección fija de una salida con memoria.
- Control de coste: éxitos de navegación normal S7 y S13 y ambos aciertos por par, y intervención aplicada/timeout/fork_reached.

**Métricas secundarias (historia guiada diagnóstica):**
- Bifurcación con prefijos docentes idénticos, y cortes de F4 **post_cue**, **halfway** y **last_step**, con readout restaurado en la decisión. Para cada brazo y tamaño, primer acto del agente correcto, \`both_teacher_actions_correct\`, acciones distintas entre pistas, y duración/cobertura; exactamente los mismos mundos de confirmación.
- H (hidden + cell LSTM) medido en el fork. Se entrena un probe ridge lineal (\`alpha=10\` fijo) solo en **80 pares S7 train/semilla por brazo** en condición natural o interrumpida; ambos probes (fijo natural y adaptado al corte) se evalúan en S7/S13 confirmación. El probe no modifica la política. Decodificar contenido **no equivale a usarlo para decidir**; los resultados conductuales tienen prioridad.

## Integridad y decisión

1. Tests sin abrir confirmación validan contrafactual key/ball, máscaras activas y sham neutra, igualdad de observaciones externas y acciones/labels de BC entre brazos, RNG de máscara independiente de etiquetas ocultas, splits disjuntos y entrada 660.
2. Tras acabar los 8 entrenamientos con presupuesto fijo, un job **separado** descarga los checkpoints y abre confirmación una sola vez. Primero verifica SHA, parámetros, entrenamiento y manifest; después corre todos los controles de forma preprogramada, sin seleccionar modelos.
3. Métricas primarias y secundarias con agregación de **cuatro semillas**; resultados por semilla, incluidos negativos. Los 160 episodios confirmatorios por brazo/tarea no son 160 replicas de entrenamiento. Diferencias entre brazos emparejadas por seed, no reportar significación espuria tratando episodios como independientes.
4. Criterio de apoyo descriptivo: mejora de éxito S13-blackout **en al menos 3 de 4 semillas**, junto a incremento de \`both_correct\` y sin colapso fuerte de la navegación S13-normal. La mejora no demuestra conciencia ni memoria semántica general; repetir fuera de esta tarea sería necesario para generalizar.
5. Si un job falla por bug **antes** de abrir confirmación, arreglar código y repetir entrenamiento completo sin usar resultados de confirmación. Si falla después de abrir, no seguir ajustando sobre esa misma partición; registrar el problema y diseñar otra prueba preregistrada.
6. La política aprende bajo blackout F4; no se presenta como entrenamiento de **LearnedRelevance** ni F5. Sin teletransportes, reward shaping, filtros de acción, PC ni VPS.

## Interpretación posible

- **Mejora blackout + rendimiento normal aceptable:** evidencia de robustez aprendida a una perturbación temporal en F4, no necesariamente transferencia a otras arquitecturas.
- **Probe decodifica H pero acción no sigue la pista:** información interna disponible que la política no usa establemente.
- **Solo mejora teacher-forced y no autónoma:** no vender como robustez en navegación real.
- **No mejora:** el schedule de train puede ser insuficiente o la arquitectura dependiente de F4; no retocar condiciones tras ver confirmación.
