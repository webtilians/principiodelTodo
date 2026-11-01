# F9-V — lanzamiento de prueba empírica con 4 réplicas y mundos independientes

**2026-11-01.** Confirmación de la preflight [Actions #41345254362](https://github.com/webtilians/principiodelTodo/actions/runs/41345254362): **SUCCESS en contracts y smoke-real-actor**. Los checkpoints originales F9-O/P y las pruebas de integridad fueron validados. Protocolo registrado antes de cualquier test: commit `e1347f4603f781d7730222b13ae5e9bc09591291`.

**Smoke:** 1 mundo y ambos cues, historia F4 natural y blackout post-cue. D=READ correcto−SKIP correcto: **+1, −1, 0, 0**. Un pequeño oráculo de intervención recupera una primera decisión adicional con solo una consulta. Resultado TÉCNICO, no validación estadística. El efecto de lectura no es necesariamente siempre favorable, y los estados con F4 históricamente interrumpida pueden comportarse de modo diferente.

**Ejecución de desarrollo:** 4 checkpoints congelados de F9-P; 32 pares de entrenamiento por réplica × 2 cues × 2 regímenes → 128 estados de TRAIN/réplica y 256 continuaciones causales/réplica. La cabeza aprende a predecir preferencia READ/SKIP con BCE sin ver estados privilegiados durante decisión. DEV completamente disjunto: 16 pares por réplica ×2 cues×2 regímenes, 256 decisiones agregadas, todas con las mismas ramas READ y SKIP y oráculo diagnóstico medible. Cabezas: lex_hc, lex_public y scalar_hc; controles always/never/oracle.

**Limitación:** al menos hasta la primera decisión de cruce, la trayectoria la establece un profesor únicamente para alinear estados; el actor es autónomo después. No se mide navegación end-to-end ni aprendizaje autónomo de estructura de política. No LLM.

**Criterio preregistrado:** suficientes D positivos TRAIN y DEV, al menos 90 % de aciertos de always con al menos 20 % menos accesos a F4, 3/4 réplicas dentro de 2 aciertos de always, >3pp vs mismo predictor solo-público. Si falla, resultado NEGATIVO. No se abre CONFIRM F9-V `8300000+`, F9-U `7300000+`, F9-T `6300000+` ni histórico `900000+`.

Este archivo activa UNA única ejecución TRAIN/DEV con resultados publicados, sin reajustar a posteriori.
