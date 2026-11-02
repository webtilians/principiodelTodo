# INFINITO F9-Y — Valor causal de memoria MÁS ALLÁ del cue y la historia

**Preregistro 2026-11-02, antes de implementar, entrenar o abrir un solo mundo nuevo.** Rama aislada sobre F9-X (PR #40), sin reescribir ninguno de los experimentos anteriores. El siguiente resultado se debe publicar incluso si es negativo o no identificable.

## Pregunta y nivel de evidencia
Hipótesis `H_conditional`: dadas la identidad de pista (key/ball) y el régimen de historia F4 (normal/postcue_off), las señales neuronales previas H/C contienen información predictiva adicional sobre el valor causal local de READ, `Y=1[D=+1]`. Hipótesis `H_transfer`: tal información permite transferir la predicción a la cuarta combinación cue×historia cuando se entrena únicamente con las otras tres.

**No equivale a conciencia, metacognición general, lectura óptima de toda la trayectoria ni navegación end-to-end.** El entorno sigue incluyendo prefijo espacial docente hasta el primer fork, actor F9-P y F4 congelados; el evaluador define D mediante dos continuaciones READ/SKIP idénticas salvo una lectura F4 en el fork y F4=0 después.

## Factores y fuga de información
Cuatro estratos `(key,normal)`, `(ball,normal)`, `(key,postcue_off)`, `(ball,postcue_off)`. Cada mundo contribuye un ejemplo por estrato, por actor. Todas las continuaciones contrafactuales se ejecutan solo para etiquetar TRAIN y para **evaluar** DEV. `correct_side`, `D`, `cue` y `regime` no llegan al predictor desplegable. Los predictores estadísticos diagnósticos que reciben cue y régimen se marcarán expresamente **PRIVILEGIADOS**; nunca los compararemos como productos desplegables sin esta advertencia.

## Particiones estrictamente nuevas
- TRAIN `10_100_000 + 10_000*actor_index + j`, `j=0..127`: 128 mundos × 4 estratos por réplica; 512 estados/actor, 2048 total.
- DEV `10_200_000 + 10_000*actor_index + j`, `j=0..31`: 32 mundos × 4 estratos por réplica; 128 estados/actor, 512 total.
- CONFIRM `10_300_000 + 10_000*actor_index + j`, `j=0..63`: RESERVADO y sin entrada implementada en F9-Y. Solo autorizar si criterios DEV se cumplen; bloqueo irrevocable de arquitectura, pesos, umbral y evaluación antes de abrir.
- No reutilizar F9-V 8100000/8200000/8300000+, F9-W 9100000/9200000/9300000+, F9-X 9500000/9600000+, F9-T 6300000+, F9-U 7300000+ ni histórico 900000+. Los cuatro F9-P originales provienen de Actions #40960267485; las mismas fuentes BC #40957255987; validar SHAs y código F9-V/F9-Q originales.

## Diseño: cuatro actores independientes, mismo presupuesto por brazo
Se usa la misma MLP Tanh `(dim_entrada)→32→1`, Adam `lr=0.001`, BCEWithLogitsLoss **SIN pos_weight**, batch completo, 200 pasos exactos; umbral READ = 0.5, sin optimización por DEV. Inicialización `torch.manual_seed(113000 + actor_index + 100*fold)` fija y compartida entre los brazos de igual arquitectura.
Los siguientes brazos existen por réplica:
- `deploy_hc`: 148 público + 128 H/C; sin etiquetas cue/regime (276).
- `deploy_public`: 148 público + 128 ceros (276).
- `controlled_hc`: 148 público + 128 H/C + 4 indicadores de cue y régimen (280); **PRIVILEGIADO para diagnóstico**.
- `controlled_nuisance`: 148 público + 128 ceros + los mismos 4 indicadores (280); control privilegiado que conoce pista e historia.
Se entrenan los cuatro sobre TRAIN completo, **y cuatro pares de `controlled_hc`/`controlled_nuisance`** con entrenamiento leave-one-combination-out (LOCO): cada par ve exactamente tres de los cuatro estratos, luego se evalúa en DEV del estrato omitido. Se mantienen los cuatro actores separados, sin transferir pesos entre ellos ni usar DEV para ajustar la preparación.

**La comparativa principal es `controlled_hc` vs `controlled_nuisance` en LOCO**, con igual arquitectura, indicadores privilegiados, presupuesto, inicialización y conjuntos TRAIN; solo difiere H/C. No confundir esa prueba causal de variables con una política desplegable. `deploy_hc` vs `deploy_public` se analiza en DEV completo como comprobación de viabilidad sin cue/regime privilegiados.

## Estimadores y controles preregistrados
1. Métrica principal: reducción de *log-loss natural, sin ponderación* en DEV LOCO `delta = loss(nuisance) - loss(HC)`, promedio por estado. Reportar todos los actores, estratos, número positivo/negativo, log-loss/Brier, AUC cuando existe ambas clases y READ positivos/dañinos/neutros a umbral 0.5.
2. Soporte por estrato para **interpretación de generalización de D+**: >=4 etiquetas positivas y >=4 no positivas en DEV agregado, **y al menos dos actores con >=2 positivos** en dicho estrato. Sin ese soporte, `NO IDENTIFICABLE` para la hipótesis en ese estrato, aunque se reporta el log-loss. No seleccionar estratos ni seeds a posteriori. Si menos de 2 estratos tienen soporte, el gate primario queda **INCONCLUSO**, no `PASS`.
3. Contraste condicionado en mismos estratos: al evaluar `controlled_hc` original (entrenado en todos los estratos) y su LOCO correspondiente, permutar cíclicamente H/C **entre mundos distintos del mismo actor, cue y régimen**, conservando las 148 señales públicas y los indicadores. Comparar pérdida original vs pérdida permutada. Repetir por grupo H y C por separado. El actor no se vuelve a ejecutar; estos cruces pueden ser fuera de distribución, y no son una prueba absoluta de semántica.
4. Dos contrastes de transferencia: (a) la evaluación LOCO deja fuera *una combinación completa*, aunque sus dos factores aparecen en otras combinaciones TRAIN; (b) DEV usa mundos jamás vistos. Posibilidad de inferencia no identificable si un estrato carece de D+ estructuralmente.
5. Bootstrap por **bloques de mundo estratificado por actor**: 2.000 muestras, semilla 20261102; cada bloque conserva los cuatro estratos y las predicciones pareadas de todos los brazos. IC del delta LOCO entre grupos y del impacto de shuffle, calculados con el mismo conjunto de estratos elegibles determinado por soporte. No contar 512 estados como 512 mundos independientes. Denominadores por actor y estrato siempre visibles.
6. Estimación descriptiva por estrato de cómo cambia `P(D+|H/C, cue, régimen)` respecto a `P(D+|cue,régimen)`: comparación con baseline empírico Laplace de TRAIN y una prueba AUC intra-estrato. **Prohibido** declarar éxito por buena accuracy trivial con muchos SKIP.

## Gate científico fijado antes de resultados
`H_transfer` solo será *apoyada en DEV* cuando:
- al menos dos estratos tengan soporte;
- el `delta` LOCO (nuisance - HC) medio en esos estratos sea >=0.02 nats/estado, **IC95 % bootstrap con extremo inferior >0**;
- al menos tres de cuatro réplicas tengan `delta>0` y al menos dos estratos mejoren `delta>0`;
- permutar H/C dentro del mismo estrato haga subir log-loss LOCO >=0.02 nats/estado y el efecto tenga signo positivo en >=3 réplicas.
No convertir `SUCCESS` de Actions en éxito científico. Si hay soporte pero no se cumplen estos cuatro controles, **FAIL**; si no hay soporte suficiente, **INCONCLUSO**. La tasa de READ/precision/recall y el desempeño end-to-end son diagnósticos secundarios.

## Interpretación y límites
No permitir selección de unidades C32/C44/C8 descubiertas posthoc en F9-X: se usarán los 128 canales originales y arquitectura fija. No introducir filtrado de positivos DEV ni seleccionar épocas por resultados. La correlación entre H/C y D no demuestra causalidad estructural de una representación general (solo las intervenciones READ/SKIP identifican D localmente). Si los modelos solo imitan cue, condición o escasez, documentar negativo; si muestran ganancia condicional, exigir confirmación con pesos congelados 10300000+ antes de afirmación fuerte.
