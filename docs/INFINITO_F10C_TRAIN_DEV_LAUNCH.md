# F10-C — Único lanzamiento prospectivo TRAIN y DEV
**2026-11-02.** El [prerregistro completo](./INFINITO_F10C_VERSIONED_RELATIONAL_PREREG.md) se congeló en el commit `2bd7393223b3c917746769f4264e39c57ecd420b` **antes** de generar nuevas semillas y antes del ejecutor. Se abre únicamente:
- TRAIN `14_100_000 + 10_000*index+j`, `index=0..3, j=0..511`, 512 mundos por actor.
- DEV `14_200_000 + 10_000*index+j`, `index=0..3, j=0..63`, 64 mundos por actor.
- CONFIRM `14_300_000+` permanece **sellada** y sin ejecutor.

Un mundo contiene los mismos 60 informes públicos para cuatro consultas finales no anticipables; cada clave consultada tiene 5 registros con credenciales, revisiones y contradicciones públicas. El valor correcto se calcula solo a partir del último evento firmado. Entrenar `relational_read`, `unary_read_control`, `recurrent_only_control` en las mismas muestras nuevas, sin filtrar; comparar en nuevos DEV frente a los **checkpoints originales congelados** F10-B del run [#41443064754](https://github.com/webtilians/principiodelTodo/actions/runs/41443064754), sin evaluar ningún mundo de F10-B otra vez.

READ/SKIP, obsolescencia y firma falsa cambian solo contenido F4 en el punto final, nunca la GRU, historia pública, query, pesos ni verdad. Gate prospectivo de cuatro estratos/actor, prioridad temporal acreditada, soporte causal y ventaja relacional sobre control unario reentrenado; bootstrap por mundo completo. No se entrena ninguna cabeza READ. El éxito en esta tarea sintética no equivale a transferencia a INFINITO integrado.

La publicación de este archivo dispara exactamente la ejecución de `.github/workflows/infinito-f10c-relational-versioned.yml`.

## Recuperación de preflight sin datos (2026-11-02)

La ejecución #41447020599 falló en la prueba inicial, **antes del paso de entrenamiento o de abrir mundos nuevos**, por la ausencia de la clave de compatibilidad `latest_unsigned` que usa la función original de F10-B al simular credenciales falsas. El commit `e989651cec2456955ca6d1309beb17db66519f70` añade exclusivamente ese alias a la misma bandera pública `last_unsigned`; no cambian casos, seed, reglas, modelos, presupuesto ni gates. Esta edición lanza el **primer intento que puede llegar a TRAIN/DEV**, no una repetición sobre datos observados.
