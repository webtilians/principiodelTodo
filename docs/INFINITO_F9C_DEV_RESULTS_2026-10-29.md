# F9-C — Registro de desarrollo (29 de octubre de 2026)

## Diseño del diagnóstico
[Protocolo pre-registrado](INFINITO_F9C_VISIBLE_CUE_PROTOCOL.md).
Dos vistas simbólicas (MiniGrid Memory S7/S13), una inicial donde la pista
es realmente visible, otra en una bifurcación en la que el objeto inicial
está fuera del campo visual. El cambio contrafactual afecta solo al objeto
inicial y a cuál de los dos destinos es correcto; **la imagen de decisión
es bit a bit idéntica**. La acción de la primera observación no puntúa,
la segunda consiste en elegir arriba/abajo. Es un *ensayo controlado*,
no el MiniGrid original con navegación de siete acciones.

## Ensayo 20k/modelo, semilla 0, desarrollo

[GitHub Actions (éxito técnico)](https://github.com/webtilians/principiodelTodo/actions/runs/40883765649).
Tres tests causal/visibilidad: PASAN.

| Agente | S7 original | S7 con pista invertida | Pares que cambian decisión | S13 transferencia sin entrenar |
|---|---:|---:|---:|---:|
| PPO | 50/100 (50%) | 50/100 (50%) | 0/100 | 16/30 (53,3%) |
| LSTM | 50/100 (50%) | 50/100 (50%) | 0/100 | 16/30 (53,3%) |

**Resultado: FAIL de memoria en ambos modelos a 20k pasos**.
No hay evidencia de que los agentes usen la pista; la mejor explicación
es que siguen una respuesta fija. No extrapolar un fallo del LSTM general,
ni una mejora de módulos INFINITO (todavía no integrados).

## Siguiente escala autorizada
[Ensayo de 100k/modelo](https://github.com/webtilians/principiodelTodo/actions/runs/40884217413):
mismas observaciones, intervención causal, configuraciones y 1 semilla,
sin abrir el conjunto retenido. Si tampoco aprende, primero diagnosticar
el aprendizaje o ensayar una referencia supervisada con etiquetas de
entrenamiento separadas, **declarando explícitamente que es aprendizaje
supervisado y no PPO o transferencia plena**.
