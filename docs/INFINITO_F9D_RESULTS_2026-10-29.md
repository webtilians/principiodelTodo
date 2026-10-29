# F9-D — Resultados observados, 29 de octubre de 2026

**Resultado: se aprende adquisición de pista y navegación S7, pero no se demuestra
uso causal de la pista para escoger salida. No hay referencia competente para F4.**

[CI completa](https://github.com/webtilians/principiodelTodo/actions/runs/40902425412)
sobre `ac036e4b6e90009f1830844759e3c303b17563e9`: 6 pruebas nuevas, 4 controles
F9-C, cuatro entrenamientos y agregación terminados con éxito.
Éxito de CI significa ejecución válida, no éxito científico.

Base: PR #9 fusionada en `experiment/f9b-behavior-audit` a las 16:21:30 de Madrid,
commit `bb0508bea461f82c5ea067def25fad70fb40efe8`. No en master.

## Datos y presupuestos realmente ejecutados

Cuatro semillas independientes; 600 mundos S7 por semilla, dos variantes de pista
por mundo; 40 épocas de imitación. Cada política recibe 20.096 transiciones RL
reales (20.000 solicitadas, redondeo de rollout de SB3). Cuatro políticas por
semilla: PPO y RecurrentPPO desde cero y tras imitación.
PPO: 27.912 parámetros; recurrente: 118.408. No equivalencia de cómputo afirmada.
Cada evaluación: 10 mundos pareados por tamaño y semilla, 80 episodios por
condición agregada contando original+invertida. S13 es transferencia sin entrenar.

El acierto combinado promedia original e invertida, no solo la condición favorable.
Los 40 pares están agrupados en cuatro semillas de entrenamiento: no son 40
réplicas independientes de aprendizaje. No se presenta significación estadística.

| Modelo | Etapa | S7 acierto combinado | S13 acierto combinado | Cambios S7 / S13 |
|---|---|---:|---:|---:|
| random | random | 25.0% | 22.5% | 0/40 / 0/40 |
| ppo_scratch | after_rl | 37.5% | 25.0% | 0/40 / 0/40 |
| lstm_scratch | after_rl | 2.5% | 0.0% | 0/40 / 0/40 |
| ppo_bc | before_rl | 50.0% | 50.0% | 0/40 / 0/40 |
| ppo_bc | after_rl | 50.0% | 50.0% | 0/40 / 0/40 |
| lstm_bc | before_rl | 50.0% | 50.0% | 0/40 / 0/40 |
| lstm_bc | after_rl | 50.0% | 28.7% | 0/40 / 0/40 |

## LSTM tras imitación + RL, por semilla

| Semilla | S7 original | S7 invertida | S7 pista vista | S13 combinado | S13 timeout |
|---|---:|---:|---:|---:|---:|
| 0 | 70.0% | 30.0% | 100.0% | 50.0% | 0.0% |
| 1 | 20.0% | 80.0% | 100.0% | 15.0% | 70.0% |
| 2 | 50.0% | 50.0% | 100.0% | 50.0% | 0.0% |
| 3 | 70.0% | 30.0% | 100.0% | 0.0% | 100.0% |

En S7, las cuatro semillas ven la pista en el 100% de episodios y llegan a una
salida sin timeout. Sin embargo, ninguna cambia de salida al invertir la pista:
0/40 pares correctos en ambas variantes. Ocultar la pista mantiene exactamente el
mismo acierto por semilla. Reiniciar el estado recurrente en cada paso produce
0% y 100% timeout. Por tanto, el estado recurrente ayuda a ejecutar una secuencia
de navegación, pero estos resultados NO acreditan que retenga la identidad de
la pista para decidir.

Antes de RL, la LSTM también obtiene solo 50% combinado y cero cambios, tanto S7
como S13. No se puede culpar únicamente al ajuste RL de no aprender la pista.
En S13 el promedio pasa de 50% antes de RL a 28,75% después, con degradación en
las semillas 1 y 3. Es compatible con pérdida de navegación transferible durante
ajuste S7; no prueba olvido de la pista porque esa capacidad no se demostró antes.

## Integridad y límites

- Actor y crítico reciben solo 148 valores de imagen parcial y brújula.
  El profesor BFS usa geometría y objetivo privilegiados exclusivamente para
  etiquetas de acciones de entrenamiento. No está presente en inferencia.
- Siete acciones originales, poses iniciales nativas, horizonte y recompensa
  originales; sin teleport, acciones forzadas ni shaping en el agente.
- La variante contrafactual modifica pista y objetivo consistente: diagnóstico
  intervenido separado del episodio original.
- Visibilidad mediante comparación de imágenes al mismo pose; trazas con
  acciones, posiciones, hashes de observación, visibilidad y recompensa.
- Máscara y reinicio son controles fuera de distribución. El reinicio puede
  romper navegación; no es evidencia suficiente de memoria de pista.
- Entrenamiento/desarrollo disjuntos dentro de los índices permitidos. Holdout
  900000+ sin abrir. No cambios ni ejecución local en PC/VPS.
- Se guardaron checkpoints finales de RL. Las métricas previas a RL están
  registradas, pero no se guardó checkpoint separado antes de RL: para estudiar
  olvido en detalle, añadirlo al próximo experimento.
- CI específica de este experimento; no implica que toda la antigua suite
  del repositorio haya sido ejecutada.

## F4 y siguiente experimento

La puerta de competencia devolvió false; F4 no se entrenó ni evaluó. Compararlo
ahora no resolvería si un cambio se debe a memoria o a fallos de navegación.
El módulo existente usa Store/FIFO real, no LearnedRelevance/F5. Además,
corromper solo colores es un control insuficiente cuando todos los objetos son
verdes; hacen falta borrado y perturbación de identidad/orden.

Próximo experimento propuesto, aún NO ejecutado:

1. Medir pérdida/precisión de imitación por fase y por acción decisiva; guardar
   checkpoint pre-RL. Hipótesis: la pérdida global está dominada por navegación
   y permite ignorar la decisión dependiente de pista.
2. Comparar pérdida equilibrada por fase y codificación one-hot genérica de
   objetos visibles, manteniendo los mismos controles reactivos/recurrentes.
   Pesos/fases solo en la pérdida, nunca añadidos a observaciones del actor.
3. Exigir éxito contrafactual autónomo antes de afinar con RL. Comprobar además
   el mismo frame de cruce tras historias de pista diferentes, con estado
   recurrente intacto y borrado; marcar ese replay como diagnóstico, no benchmark.
4. Si la imitación supera el criterio, probar ajuste con anclaje BC/KL y comparar
   antes/después bajo recompensa original. Solo después replicar con 100 pares
   de desarrollo por semilla y ejecutar las ablaciones F4 con canales igualados.
   No ampliar presupuesto a ciegas ni abrir holdout.

## Evidencia conservada

Los valores completos por semilla, dependencias y hashes de checkpoints están
en [el JSON de resultados](INFINITO_F9D_RESULTS_2026-10-29.json).
Artifacts en la ejecución: `f9d-seed-0..3` (checkpoints y trayectorias) y
[`f9d-aggregate`](https://github.com/webtilians/principiodelTodo/actions/runs/40902425412/artifacts/12909334539).
Retención observada: hasta 2026-11-28.

Se verificaron también logs previos: [GRU supervisada F9-C](https://github.com/webtilians/principiodelTodo/actions/runs/40885330611)
100% S7/S13 con cambios 100/100 y 30/30; [PPO/LSTM de dos vistas](https://github.com/webtilians/principiodelTodo/actions/runs/40885125879)
50% S7 y 0/100 cambios. Son tareas controladas distintas, no resultados nativos.
