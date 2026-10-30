# INFINITO F9-P — Protocolo preregistrado: ¿PPO vuelve a depender de F4?

**Congelado el 29/10/2026 antes de ejecutar PPO F9-P y antes de abrir nuevos mundos confirmatorios.** Sobre F9-O (PR #22, run #40957255987). Objetivo de INFINITO: estudiar **memoria interna funcional** y uso causal de información histórica, no ganar MiniGrid por navegación ni atribuir conciencia a una métrica.

## Hallazgo que motiva la pregunta (evidencia previa, no nueva confirmación)

F9-O entrenó desde cero modelos S13 con seis lecturas F4 enmascaradas continuamente vs fragmentadas en la fase de imitación (BC), misma dosis. La variante continua tras BC resolvió 42/160 pares de pistas contrafactuales con **ambas primeras salidas correctas** bajo ausencia prolongada de F4, frente a 18/160. Tras PPO con F4 siempre disponible solo 14/160 vs 11/160, con control de llegada al cruce muy diferente. Estos resultados **ya consumieron 600000+**: motivan una nueva hipótesis y no son una nueva confirmación.

## Manipulación aislada, checkpoint inicial idéntico por índice

- Cargar el checkpoint **F9-O continuous6_after_bc.zip** de cuatro semillas de **run #40957255987**, verificar SHA-256 contra el reporte original de entrenamiento completo F9-O. **Mismo byte exacto del checkpoint tras BC** se carga dos veces por semilla para iniciar los dos grupos; no entrenar BC de nuevo ni elegir un checkpoint retrospectivamente entre nuevas semillas.
- \`ppo_sham\` (control): entrenamiento PPO normal con F4 íntegra; aplica el mismo calendario de **112 posiciones por cada 128 pasos de actor** a los 448 canales F4 de relleno ya nulos (esto permite auditar el reloj sin borrar información útil).
- \`ppo_blackout\` (tratamiento): las mismas 112 posiciones de cada 128 pasos de actor, contiguas en el **reloj global exógeno** \`t % 128 >= 16\`, pero bloqueando **los 512 canales de lectura F4**, sin modificar observación parcial pública 148, escritor, selector, store ni ningún reward. Las primeras 16 posiciones de cada bloque de 128 reciben F4 normal. **No** se activa por una etiqueta privilegiada, la pista ni la ruta.
- En ambos grupos hay **20.000 pasos PPO solicitados (20.096 reales)**, mismas hiperparámetros de F9-O (n_steps=128, 4 épocas, batch 128, semilla por par, rewards originales, 7 acciones, LSTM 64, 380.552 parámetros), mismo checkpoint inicial SHA y **las mismas oportunidades de reloj de blackout**, contadas por el índice de pasos PPO. Tratamiento y control divergirán en trayectorias por el aprendizaje; la cantidad **efectiva** de información no nula suprimida *no* es necesariamente igual entre grupos (por definición el sham no suprime información). Se publica el conteo real de máscaras que anularon F4 y duración/episodios. No confundir mismo número de oportunidades con mismas observaciones.
- **Nuevos mundos PPO TRAIN S13** \`550000 + index*10000 + j\`, j=0..599, alternando cue original/invertido; ninguna semilla de los TRAIN de F9-O 500000+, development, 300000+, 400000+, confirmación F9-O 600000+ ni holdout 900000+. El envoltorio de PPO solo consume imágenes públicas, salvo que el *simulador generador de TRAIN* invierte el cue para equilibrar ejemplos.
- Al guardar, registrar hashes antes y después, forma de observación, parámetros, reloj y restricciones. **No** entrenar sobre confirmación ni ajustar el reloj según sus resultados.

## Validaciones antes de abrir la confirmación

1. Tests con fixtures y un mundo **TRAIN nuevo** verifican que \`t%128>=16\` es independiente de episodio/posición/cue, que exactamente **112 de cada 128 pasos** están programados; que el sham no altera canales F4 informativos, el tratamiento pone a cero solo readout F4 y los 148 públicos son idénticos, y el escritor F4 sigue acumulando eventos.
2. Gate de **8 checkpoints PPO y 4 checkpoints BC originales**, usando cada uno de los originales por SHA; el *par* PPO debe comenzar desde la misma firma de política y tener los mismos pasos/hyperparámetros. Exigir reloj programado idéntico en ambas políticas y no-nulo F4 efectivamente anulado para el tratamiento; para sham, cero bytes informativos cambiados. Si esto falla, **no** se abre confirmación.
3. Cualquier error corregido antes de la confirmación debe registrarse como enmienda. Nunca utilizar la nueva partición para seleccionar políticas.

## Confirmación nueva, congelada desde ahora

Semillas S13/S7 \`700000+index*10000+j\`: j=0..39 en S13 → **160 pares/320 episodios por grupo y condición**; j=0..19 en S7 → 80 pares/160 episodios. **Distinta** de 300000+, 400000+, 500000+, 550000+, 600000+ y del holdout 900000+, que permanece cerrado. Criterio sin exclusión de cue no visto o cruce no alcanzado:

**Primario (S13 \`cue_lost_continuous\` después de PPO):** número de pares con **ambos primeros compromisos de salida correctos** para pista inicial original/invertida bajo navegación autónoma con F4 ausente después de que la pista haya salido de la vista. Diferencia \`ppo_blackout−ppo_sham\` tanto global como por cuatro semillas. Criterio descriptivo para evidencia consistente: **ganancia estricta en ≥3/4 semillas + ganancia agregada**, no subida del éxito de una variante. El denominador son **todos** los pares, incluidos atascos.

**Secundarios preregistrados:** controles \`normal\`, \`fixed_4\` y un control mecanístico **\`cue_lost_hidden_reset_at_junction\`**: al primer acceso al cruce, además de mantener la misma F4 anulada por la regla cue-lost, resetear explícitamente el estado recurrente H/C *antes de la acción*. Si en el grupo de tratamiento aparece una ventaja primaria pero desaparece bajo reset, hay evidencia funcional de dependencia de estado recurrente para esa decisión; por sí solo un reset también perturba el controlador/navegación, por lo que no prueba codificación específica de la identidad. Se reporta en cuántos episodios se aplica de hecho el reset. Este control es independiente y secundario: no sustituir el desenlace primario.
- **Ancla BC sin PPO**, obtenida del checkpoint original idéntico de F9-O, evaluada en los **mismos mundos nuevos** y condiciones \`normal\`, \`cue_lost_continuous\`, \`fixed_4\`. El ancla es comparador histórico de la fase de aprendizaje, no se reclasifica como tercero PPO. No escoger el ancla si supera al tratamiento; diagnóstico temporal, no criterio primario.
- Desglose F9-N: pista vista, cruce alcanzado, primer compromiso espacial, primera elección acertada entre *todos*, elección correcta condicional al compromiso **con denominador y sesgo de selección**, timeouts, éxito y pares ambos correctos. S7 como transferencia únicamente, sin reclamar generalización. Reportar diferencias por semilla y no tratar 320 episodios como 320 réplicas de entrenamiento.
- Todos los episodios autónomos usan solamente \`model.predict\` con input F4/public normalizado, sin acción de profesor ni memoria/etiquetas/geometría entregadas al actor. Las etiquetas y el momento del cruce son **solo** instrumentos de evaluación o manipulación externa preregistrada.

## Decisiones e interpretación

- Si \`ppo_blackout\` gana en **ambas primeras decisiones correctas por par** en S13 cue-lost en ≥3 semillas y con beneficio agregado, y el reset de H/C reduce la ganancia, habría evidencia de que el PPO con cortes largos preserva/mejora **memoria recurrente funcional** en este entorno.
- Si aumenta llegada/éxito individual pero no la métrica de pares, **no** se considerará éxito de memoria, por mucho que suba el retorno.
- Si ambos fallan, la posible memoria tras BC no sobrevivió al PPO con este diseño; no alterar retrospectivamente criterio ni redes usando 700000+.
- Este ensayo no prueba conciencia, comprensión ni memoria general; LSTM H/C puede contener trazas y navegación que no sean recuerdos utilizables. Confirmación \`700000+\` queda **consumida tras la primera ejecución**.
