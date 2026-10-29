# F9-B — Resultados de auditoría conductual S7/S13 (2026-10-29)

**Estado:** CI completado correctamente. No se entrenaron modelos nuevos;
se reutilizaron los checkpoints congelados de 100k/300k.
[Run #40881408115](https://github.com/webtilians/principiodelTodo/actions/runs/40881408115).
[Artifact con report.json y trayectorias JSONL](https://github.com/webtilians/principiodelTodo/actions/runs/40881408115/artifacts/12899988758).

Protocolo y advertencias: [INFINITO_F9B_BEHAVIOR_AUDIT_PROTOCOL.md](INFINITO_F9B_BEHAVIOR_AUDIT_PROTOCOL.md).

## 1. S7 original y contrafactual (50 episodios / política)

| Política congelada | Éxito original | Éxito señal cambiada | Salida original | Cambios de salida | Secuencias de acciones idénticas |
|---|---:|---:|---|---:|---:|
| PPO 100k | 27/50 = 54% | 23/50 = 46% | inferior 50/50 | 0/50 | 50/50 |
| LSTM 100k | 27/50 = 54% | 23/50 = 46% | inferior 50/50 | 0/50 | 50/50 |
| PPO 300k | 27/50 = 54% | 23/50 = 46% | inferior 50/50 | 0/50 | 50/50 |
| LSTM 300k | 27/50 = 54% | 23/50 = 46% | inferior 50/50 | 0/50 | 50/50 |

Las cuatro políticas tienen en S7: 0 acciones izquierda, 50 derecha y
162 avanzar en los 50 episodios (media 4,24 pasos). Los 27 éxitos surgen
porque la salida inferior era la correcta en 27 de los 50 escenarios,
**no porque el modelo haya utilizado la memoria**. El objeto inicial cambia
en el mundo en cada intervención pero la observación inicial parcial sólo
difiere en 5/50 semillas; incluso en esas cinco, la salida sigue siendo
inferior y la secuencia de acciones no cambia. La ausencia de cambios de
acción no prueba que los agentes siempre vieron la señal: el protocolo
registra precisamente cuántas veces la observación inicial cambió.

Éxitos condicionados por objeto inicial (idénticos para los cuatro agentes):
- key: 20 éxitos / 32 episodios.
- ball: 7 éxitos / 18 episodios.

## 2. S13 transferencia zero-shot (30 episodios / política)

| Política congelada S7 | Éxito original S13 | Éxito señal cambiada | Salida escogida | Cambios de salida |
|---|---:|---:|---|---:|
| PPO 100k | 14/30 = 46,67% | 16/30 = 53,33% | inferior 30/30 | 0/30 |
| LSTM 100k | 14/30 = 46,67% | 16/30 = 53,33% | inferior 30/30 | 0/30 |
| PPO 300k | 14/30 = 46,67% | 16/30 = 53,33% | inferior 30/30 | 0/30 |
| LSTM 300k | 14/30 = 46,67% | 16/30 = 53,33% | inferior 30/30 | 0/30 |

PPO 100k/300k y LSTM 100k: 30 derecha y 181 avanzar (media 7,03 pasos).
LSTM 300k: 40 izquierda, 70 derecha y 181 avanzar
(media 9,70 pasos); **no cambia la elección final**, sólo el camino.
En S13, la observación parcial inicial cambia en apenas 1/30 parejas.
Los cuatro agentes mantienen intacta la secuencia completa de acciones
entre original e intervención en los 30 episodios.

## Interpretación falsable

- La hipótesis de que estos baselines ya resuelven MiniGrid Memory
  utilizando el objeto recordado queda **refutada en el conjunto evaluado**:
  las elecciones finales no dependen del objeto y los éxitos no superan
  lo esperable por escoger siempre una de dos salidas.
- El progreso 0%→54% entre 20k y 100k reflejaba aprender una ruta fija
  que finaliza el episodio, **no aprender memoria**.
- El aparente éxito en S13 no es transferencia cognitiva: es la misma
  política fija de salida inferior sobre otra geometría.
- **No** se ha probado F4 Store/FIFO en esta auditoría: ningún resultado
  autoriza a concluir que INFINITO mejore o empeore frente a PPO/LSTM.
- El entorno sigue siendo externo a INFINITO y estos son datos de
  desarrollo, no holdout. Una sola semilla de entrenamiento impide una
  comparación estadística poblacional entre arquitecturas.

## Diagnóstico de investigación recomendado

No seguir escalando pasos **sin** inspeccionar por qué la política no
explora la señal inicial. Separar:
1. **Acceso a la pista:** si la pista se ve y cuánto tiempo se conserva,
   y si la posición inicial hace que quede fuera del campo de visión.
2. **Exploración/objetivo:** medir visitas al cuarto inicial, exploración
   y recompensa para rutas que dependen realmente de la señal.
3. **Baselines competentes:** configurar una referencia que **aprenda
   condicionalmente** de la pista con varias semillas de entrenamiento,
   sin darle estado oculto ni guías privilegiadas.
4. **Intervenciones causales de memorias:** recién después del punto 3,
   ejecutar las seis condiciones incluidas F4 y controles de coste.
5. **Holdout:** permanece sin abrir; ninguna modificación futura puede
   justificarse por resultados observados del test retenido.

El resultado negativo es útil: evita atribuir un 54% de éxito
a una capacidad que los agentes no están mostrando.
