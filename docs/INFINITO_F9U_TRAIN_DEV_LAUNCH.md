# INFINITO F9-U — Lanzamiento único TRAIN/DEV (no confirmación)

**Fecha 2026-11-01.** Experimento basado en el actor LSTM propio de F9-P y memoria real F4, **sin LLM externo**.

El protocolo preregistrado se escribió **antes del código** y está fijado por commit `4fc7f8abaa41d747892299728163cfbe923ad314`: [F9-U — recompensa contrafactual por pares](./INFINITO_F9U_PAIRED_COUNTERFACTUAL_REWARD_PROTOCOL.md).

[**GitHub Actions #41335668515**](https://github.com/webtilians/principiodelTodo/actions/runs/41335668515) completó con **SUCCESS** el control de integridad, tests de gradiente y un smoke con los checkpoints neuronales originales F9-O/P y el almacén F4 real.

**Smoke con UN mundo de entrenamiento, DOS pares de prueba por cabeza**: todos los brazos dieron **0/2 pares correctos**, y las cuatro cabezas aprendieron tasas de READ de aproximadamente 6,25%, 8,77%, 15,38% y 12,82%, respectivamente. Eso **NO valida la hipótesis de aprendizaje**; confirma que cuatro cabezas entrenan y que no se ha introducido fuga de información. El actor con READ siempre tampoco resolvió los dos pares de smoke.

Lanzamiento reservado: 4 réplicas independientes, 4 cabezas (individual/paired x H/C/public), **600 mundos TRAIN×2 pistas por cabeza y 600 actualizaciones**, 20 pares DEV×4 réplicas, evaluación determinista y estocástica, controles always/never/periódico/observación pública y ablación H/C solo en cabeza. Emplea las mismas fuentes de pesos congelados que F9-T. Reportar también tasa de decisiones idénticas con pistas invertidas y coste de consultas.

**NO** se abren las 40 parejas/semilla de CONFIRM `7300000+` en este workflow; tampoco los holdouts históricos `900000+` ni reserva F9-T `6300000+`.

**Puerta científica de desarrollo**: si el brazo paired+H/C no mejora de manera consistente el control individual equiparado, no retiene los éxitos del actor always, no ahorra al menos 25% READ y no supera la cabeza paired sin H/C, registrar F9-U como NEGATIVO y no abrir CONFIRM. No ajustar beta=0.5 ni read_cost=0.15 después de observar DEV.

El entrenamiento del actor se mantiene congelado. No fusionar a producción; PR #36 borrador.
