# INFINITO F9-T — Lanzamiento de entrenamiento neuronal real (solo TRAIN/DEV)

**1 noviembre 2026.** La investigación vuelve al núcleo F9: LSTM H/C + F4 real + siete acciones MiniGrid.

Protocolo congelado antes del código: `8975d68559dca321e9cc3ceba62162284eb6dc72`.
[CI de integridad y smoke con RecurrentPPO original #41326912971](https://github.com/webtilians/principiodelTodo/actions/runs/41326912971) **SUCCESS**. Se comprobó que la cabeza puede entrenarse y elegir READ/SKIP sin acceso a readout F4 antes de decidir. **Smoke no prueba aprendizaje útil** (dos cabezas 100% READ y 0/2 pares exitosos).

**Única ejecución TRAIN/DEV:** cuatro réplicas independientes, cada una actor `ppo_sham_after_rl.zip` congelado y validado con informe SHA desde F9-O/P; cabeza `learned_hc` vs `learned_public_only` con misma arquitectura e inicialización; 600 mundos S13 TRAIN×2 cues; 20 pares DEV nuevos×4 réplicas, más ablación H/C **solo a la cabeza**. Controles always/never/periodic_4/reloj de primer objeto público.

Entrenamiento exclusivamente por recompensa nativa y penalización fija 0.15 por fracción de lecturas. No hay LLM, acciones docentes, observaciones futuras ni etiquetas de pista para la cabeza.

El workflow registra precisión autónoma, pares invertidos, lectura por frame, número de episodios, métricas de entrenamiento y trazas. **NO abre confirmación `6300000+` ni histórico F9 `900000+`**. No cambiar hiperparámetros después de ver DEV para perseguir resultados favorables.

En caso de fracaso del gate, registrar resultado negativo; no afirmar "metamemoria" por tener una cabeza binaria capaz de consultar.
