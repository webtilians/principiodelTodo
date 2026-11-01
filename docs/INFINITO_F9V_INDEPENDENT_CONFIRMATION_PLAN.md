# F9-V — Congelación y confirmación independiente (registro previo a abrir CONFIRM)

**Fecha 2026-11-01.** Esta rama parte del commit exacto del experimento F9-V cuyo DEV fue ejecutado: `898b6e43f6f1dd08b970fe4a0e4651d271944e80`, run [#41345497955](https://github.com/webtilians/principiodelTodo/actions/runs/41345497955). No se edita `observatory/f9v_lexicographic_causal_f4.py`, la arquitectura de cabeza, los datos, el optimizador, el umbral, F4, el actor ni el protocolo F9-V.

**Justificación de apertura:** el criterio preregistrado de DEV dio `passes=true`: cabeza lex_hc acertó 183/256 con 15/256 lecturas, public 168/256, always 182/256. Este resultado es exploratorio y no garantiza reproducción en confirmación.

### Frozen provenance
- Hash SHA-256 código F9-V original: `5bf29a5c4b178c8bdc7c40c64901fd52488c482e3ed9860b0ba162f6bc513532`.
- Hash SHA-256 protocolo original: `73b3b19805f75edff093af169dcd485bc11307dbc72dc3a600c6b3ddb127ac2e`.
- Pesos **congelados, sin reentrenar**: se descargan desde artifacts originales de la ejecución DEV:
  - actor índice 0: artifact `13098763143`
  - actor índice 1: artifact `13098733338`
  - actor índice 2: artifact `13099027297`
  - actor índice 3: artifact `13099500331`
- SHA-256 de los 12 pesos `lex_hc.pt`, `lex_public.pt` y `scalar_hc.pt` fijados en `observatory/f9v_frozen_confirm.py` **ANTES** de ejecutar confirmación; mismatch = abortar, sin sustitutos.
- Checkpoints de actor F9-P recuperados de los mismos artifacts originales F9-O/P y verificados con `validate_sources` y hashes `source_shas` guardados en el informe DEV.

### Confirmación única (sin usarla para optimizar)
- Mundos sellados `8_300_000 + 10_000*index+j` con `index=0..3`, `j=0..39`. 40 pares originales/invertidos × 2 regímenes `history_f4_normal` y `history_f4_postcue_off` → **160 estados/replica; 640 estados en total**.
- Mismo prefijo docente **solo hasta el primer cruce**. Contrafactuales READ/SKIP forzados **solo en ese cruce**; continuación autónoma sin F4; mismo objetivo `first_arm_correct`.
- Predicciones de cabezas desde observación pública + H/C previa, no con labels ni estado futuro. Oráculo y prueba de daños `D` exclusivamente para evaluación.
- Métrica principal: elecciones individuales correctas y consultas por 640 estados, controles `always`, `never`, `lex_public`, `scalar_hc`, oráculo; por cuatro réplicas y por régimen, también `both_first_correct` por pares de pista. Positivos `D=+1`, negativos `D=-1`, neutrales `D=0`; recall/precision selectiva y decisiones equivocadas. Posiciones del profesor / pistas son observer-only.
- Antes de ejecutar se congela la condición de replicación: `lex_hc` >=90% de aciertos del brazo `always`, <=80% de las lecturas `always`; en >=3/4 réplicas no pierde más de 5 aciertos de 160 frente a `always` (escala de 2/64 DEV), y mejora al menos **3 puntos porcentuales (=20 aciertos/640)** frente a `lex_public`. Importante: este gate de CONFIRM es una **decisión operativa preregistrada para la réplica**, no pretende convertir la proporción no aleatoria en una prueba de significación.
- Para cuantificar incertidumbre sin independencia falsa de pistas/regímenes, se estimarán intervalos bootstrap 95% para la diferencia `lex_hc - never` y `lex_hc - always` mediante **remuestreo de mundos (4 estados por mundo), estratificado por réplica**, 2.000 bootstrap y RNG seed 20261101. Se publicarán diferencias puntuales e intervalos aunque contengan cero, sin elegir retrospectivamente métricas.

### Restricciones
No reentrenar cabeza, no reentrenar actor, no cambiar `src/`, no usar LLM externo, no tunear con confirmación; prueba se ejecuta una sola vez y se conserva como irreversible. Holdouts históricos `900000+`, F9-T `6300000+`, F9-U `7300000+` permanecen sin abrir. El otro experimento sobre escasez de muestras positivas tendrá semillas completamente diferentes y no usará `8300000+`.
