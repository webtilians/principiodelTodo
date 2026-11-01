# INFINITO F9-W — ¿Faltan ejemplos positivos para aprender cuándo READ es útil?

**2026-11-01. PRE-REGISTRO: antes de implementar F9-W y ver cualquier TRAIN/DEV nuevo.** Experimento independiente de CONFIRM F9-V. La hipótesis nace de F9-V DEV (réplicas 1/2 tenían 5/128 etiquetas READ positivas, 0 consultas en DEV). Ningún resultado de CONFIRM 8300000+ se utiliza para diseñar o ejecutar este experimento.

## Hipótesis

`H_scarcity`: las cabezas F9-V con pocas muestras donde READ rescataba una primera decisión no aprendieron a consultar memoria porque el entrenamiento supervisado estaba pobremente informado sobre la clase positiva. **Predicción prospectiva**: ofrecer más **estados nuevos** de aprendizaje (no duplicar los existentes ni usar conjuntos de confirmación) aumenta la detección de READ beneficiosas en entornos nuevos, especialmente en índices 1 y 2, sin elevar consultas perjudiciales.

## Experimento factorial con control de dificultad

Mismo actor F9-P original **congelado** por índice `0..3`, misma memoria F4 y misma intervención de F9-V en primer cruce tras prefijo docente. Nada de LLM. Se comparan exactamente 4 cabezas MLP 276→32→1:
- `short32_hc`: datos de los primeros **32 mundos** nuevos TRAIN por índice, 128 estados (2 cues × 2 regímenes de historia por mundo), BCE clásico.
- `long128_hc`: primeros **128 mundos** nuevos TRAIN por índice, 512 estados; incluye exactamente los mismos 32 iniciales, 384 estados adicionales; BCE clásico.
- `short32_weighted_hc`: los mismos 32 mundos y 128 estados que short32_hc, BCE con `pos_weight = N_negative/N_positive` calculado en TRAIN (si 0 positivos, pos_weight=1 y registrar problema no-identificable). Contraste entre corregir desequilibrio y adquirir ejemplos nuevos.
- `long128_public`: 128 mundos, misma red e inicialización pero **H/C ceros** en la cabeza; control de utilidad del estado neuronal.

Todas las cabezas hacen **exactamente 200 épocas / actualizaciones Adam, lr=0.001, batch completo y mismo torch.manual_seed(91000+index)** antes de optimizar; misma arquitectura y umbral sigmoid 0.5. La única diferencia entre short32_hc y long128_hc es disponer de más mundos y ejemplos, por diseño mayor gasto de generación de datos. No afirmaremos igual coste total, ni atribución causal a «positivos» si también cambia cobertura de otros patrones.

Etiquetas `D=first_arm_correct(READ)-first_arm_correct(SKIP)` generadas por intervención causal SOLO sobre TRAIN. Etiqueta 1 si D=+1, 0 en otro caso. Las cabezas nunca reciben el oráculo, dirección correcta, contenido F4 previo a READ, régimen ni acciones docentes al predecir. En DEV ambas continuaciones contrafactuales se ejecutan SOLO para evaluar; nunca para seleccionar el READ del agente.

## Nuevas particiones aisladas

- TRAIN **9_100_000 + index*10_000 + j**, j=0..127. short utiliza j<32. Ninguna seed F9-V `8_100_000,8_200_000,8_300_000` reaparece.
- DEV **9_200_000 + index*10_000 + j**, j=0..15 (16 mundos×4 estados=64 por réplica; 256 en total). DEV nuevo, todos los métodos evaluados sobre idénticos estados.
- CONFIRM **9_300_000 + index*10_000 + j**, j=0..39. **SELLADO; no se implementa entrada a este conjunto**.
- F9-V CONFIRM 8_300_000+, F9-U 7_300_000+, F9-T 6_300_000+ y F9 histórico 900_000+ no se tocan. Crear F9-W en rama independiente basada en el commit F9-V DEV original `898b6e43f6f1dd08b970fe4a0e4651d271944e80`, NO sobre la rama de confirmación.

## Análisis preregistrado

Por réplica, informar número de positivos, negativos, neutros, D<0; evaluador: detección de positivos (recall), precisión de las lecturas, lecturas perjudiciales, consultas totales, primeras decisiones correctas, ambas elecciones correctas por par y régimen. Comparar short vs long con **exactamente los mismos mundos DEV**. También reportar coste en continuaciones del entorno (128×4×2 en entrenamiento de long, más 16×4×2 DEV por réplica), no ocultar aumento.

**Gate exploratorio H_scarcity**: para índices 1 y 2, el conjunto extendido debe contener ≥3 ejemplos **positivos adicionales** respecto a 32 mundos; `long128_hc` debe recuperar al menos **2 positivos DEV adicionales combinados** en esos índices sobre `short32_hc`, con como máximo **2 lecturas perjudiciales adicionales** agregadas en dichos índices, y no perder más de 2 aciertos totales en las 128 decisiones DEV de esos dos índices. Si falla, la hipótesis no queda apoyada; CONFIRM F9-W sellado. Si se cumple, solo justifica una prueba confirmatoria independiente, no una conclusión de que el desequilibrio sea la causa única.

Resultado observacional de F9-V previo no se utiliza para seleccionar semillas/arquitectura/hiperparámetros ni para reajustar nueva fase. Reportar condiciones de causalidad del simulador y origen supervisado, sin llamarlo aprendizaje autónomo de políticas.
