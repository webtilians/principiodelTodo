# INFINITO F9-B — Registro de resultados (29 octubre 2026)

Preregistration: [F9-B causal memory protocol](INFINITO_F9B_CAUSAL_MEMORY_PROTOCOL.md).
Código: [f9b_causal_memory.py](../observatory/f9b_causal_memory.py).
Objetivo: una prueba refutable de **memoria explícita F4 Store/FIFO** sobre MiniGrid Memory S7; la prioridad F4 LearnedRelevance y el predictor F5 no están integrados.

## Validación técnica: 128 transiciones / 1 semilla

[CI de memoria y seis condiciones](https://github.com/webtilians/principiodelTodo/actions/runs/40869016140): **PASÓ**.
Cuatro tests incluyen:
- Todos los agentes reciben exactamente la misma observación parcial actual (148 características).
- Sólo las configuraciones con memoria reciben 12 características adicionales del pasado. El control de ceros iguala el ancho de entrada.
- La intervención \`lstm_f4_corrupt\` altera únicamente los colores guardados, NO las observaciones actuales, y comparte checkpoint con \`lstm_f4\`.
- Los seis tratamientos se entrenan/evalúan de verdad en un smoke; **0 éxito para las cinco políticas entrenadas**, coherente con 128 pasos de prueba técnica.
- La barrera de competencia impide correr las ablaciones confirmatorias con baselines sin entrenar. El smoke especial está marcado como no elegible para heldout.

## Primer presupuesto de desarrollo: 20.000 transiciones/modelo

[GitHub Actions, ejecución real](https://github.com/webtilians/principiodelTodo/actions/runs/40869619780): **COMPLETADO**.

MiniGrid-MemoryS7-v0. Una sola semilla de entrenamiento, 50 episodios de evaluación por política, misma representación parcial:

| Referencia | Éxitos | Retorno medio | Pasos/episodio |
|---|---:|---:|---:|
| random | 16/50 (32%) | 0.253216 | 150.14 |
| PPO MLP | 0/50 (0%) | 0 | 245 |
| RecurrentPPO LSTM | 0/50 (0%) | 0 | 245 |

**Resultado de competencia: RECHAZADO.**
LSTM <70%, no supera al random en ≥15 pp y existe una única semilla de entrenamiento. No se puede atribuir ninguna mejora a INFINITO ni conectar F4 a un baseline competente aún.

Que random alcance el 32% y PPO/LSTM cero señala una posible dificultad de exploración/aprendizaje con recompensa escasa, no una demostración de que la memoria neur(on)al sea deficiente. El presupuesto de 20k puede ser insuficiente; no se han calculado intervalos por semillas a partir de 50 episodios.

## Siguiente presupuesto preregistrado: 100.000 transiciones/modelo

[CI de desarrollo, 100k](https://github.com/webtilians/principiodelTodo/actions/runs/40870053190). Se registrarán sus resultados aunque también fallen. **No abrir el conjunto retenido.** No optimizar hiperparámetros mirando la prueba retenida.

## Estado de los cinco pasos acordados

1. **Baseline competente**: en curso; aún no demostrado.
2. **Integración**: hecho en la rama experimental para componentes reales \`F4.Store\`, \`F4.FIFO\`, \`F4.Event\`, sin retroalimentación ficticia. LearnedRelevance no integrado; comparar esto con aprender a seleccionar recuerdos sería incorrecto.
3. **Seis variantes y controles**: implementados y aprobados en smoke; comparaciones completas pospuestas hasta superar el gate.
4. **Holdout**: conservado cerrado, sólo código disponible; exige ≥5 semillas y ≥500 episodios por semilla, freeze SHA/deps/checkpoints, marcador de una única apertura.
5. **Decisión retener/rediseñar/descartar**: NO TOMADA. Faltan resultados externos validados; un empate o una mejora de F4 frente a PPO sin LSTM competente no justificaría conservar el módulo.

**No desplegar automáticamente** la rama ni modificar trabajo activo en el VPS. El nuevo protocolo de entrenamiento de baselines está registrado en el portal, pero requiere que PC Windows y VPS ejecuten exactamente la misma revisión de esta rama mediante un despliegue revisado.
