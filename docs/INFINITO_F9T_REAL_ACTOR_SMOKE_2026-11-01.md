# INFINITO F9-T — Informe de integridad y smoke con actor REAL (no confirmación)

**1 de noviembre de 2026 — prueba técnica COMPLETADA, NO hay todavía evidencia de una nueva capacidad aprendida.**

- [Protocolo preregistrado](./INFINITO_F9T_LEARNED_F4_READ_GATE_PROTOCOL.md), antes del código: commit \`8975d68559dca321e9cc3ceba62162284eb6dc72\`.
- [GitHub Actions #41326912971](https://github.com/webtilians/principiodelTodo/actions/runs/41326912971): integrity **SUCCESS**, smoke de cabeza neuronal conectada al checkpoint RecurrentPPO real **SUCCESS**.
- [Artifact de smoke: dos pesos de read-head, detalles por dos pares, traza de READ y F4](https://github.com/webtilians/principiodelTodo/actions/runs/41326912971/artifacts/13091682321).
- El smoke descargó pesos originales F9-O \`run 40957255987\` y F9-P \`run 40960267485\`, verificó hashes/consistencia de origen y usó específicamente el actor congelado \`ppo_sham_after_rl.zip\` de semilla 0. **No es un actor LLM ni un imitador simulado.**
- Cabeza READ/SKIP de **276→32→1** con \`raw_148 + H_prev_64 + C_prev_64\`, entrenamiento únicamente de esa cabeza mediante recompensa MiniGrid y coste de lectura. Control de la misma cabeza con H/C igualados a cero; sin lectura F4 anticipada.
- Original \`src/\` y políticas congeladas no modificados, y el histórico F9 \`900000+\` no abierto.

## Resultado de dos pares SMOKE (cada uno dos historias de pista invertida)

| Brazo | Ambos primeros brazos correctos | Ambos episodios terminales correctos | Media de lecturas/frames |
|---|---:|---:|---:|
| Head entrenada H/C, **solo 1 mundo de entrenamiento** | 0/2 | 0/2 | **100%** |
| Head entrenada solo públicos, **solo 1 mundo** | 0/2 | 0/2 | **100%** |
| Actor siempre lee F4 | 0/2 | 0/2 | **100%** |
| Actor nunca lee F4 | 0/2 | 0/2 | 0% |
| Actor lee cada 4 frames | 0/2 | 0/2 | 25,09% |
| Reloj basado en primer objeto público y luego cada 4 frames | 0/2 | 0/2 | 25,09% |

**Interpretación: NO es una victoria científica.** Un mundo TRAIN es demasiado poco para afirmar aprendizaje. Las dos cabezas recién inicializadas terminaron consultando F4 en todos los frames del smoke, algo que NO satisface el criterio de reducir accesos al menos un 25%. También el actor congelado con acceso continuo falló esos dos pares. El smoke prueba solamente que las decisiones, el entrenamiento diferenciable, F4 real y un actor recurrente real **funcionan juntos sin errores técnicos**. No inferir rendimiento ni generalización a partir de 2 pares.

## Próximo paso ya protocolizado

Entrenar **cuatro** cabezas H/C y cuatro controles público-solo, cada una con 600 mundos TRAIN S13 originales/invertidos, manteniendo las mismas constantes. Después evaluar 20 pares DEV por réplica y ablación **H/C solo en el gate**, con actor H/C intacto, y reportar acierto/apertura vs siempre, nunca, reloj de 4 y reloj público. Proteger CONFIRM nuevos \`6300000+\` hasta superar todos los gates y verificar de nuevo pesos/hashes. No ajustar pesos ni hiperparámetros usando la confirmación; si el gate no aprende a economizar lecturas sin perder capacidad, declarar **resultado negativo**.

El único criterio de mérito es que la política **neuronal de lectura del propio INFINITO** mejore la frontera entre éxito autónomo y coste de consultar F4, con evidencia de que H/C contribuye y no solo la señal pública. No confundir "el agente puede consultar" con "ha aprendido que necesita recordar".
