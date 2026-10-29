# F9-C — Resultados verificados: memoria visible y decisión condicionada (29 de octubre de 2026)

## Situación anterior
[PR #8](https://github.com/webtilians/principiodelTodo/pull/8) demostró
que PPO y LSTM de 100k/300k pasos en MiniGrid Memory S7 **siempre elegían
la salida inferior** y, por ello, obtenían el 54% por sesgo de salidas
correctas. No había evidencia de memoria.

## Protocolo desarrollado y advertencias
[INFINITO_F9C_VISIBLE_CUE_PROTOCOL.md](INFINITO_F9C_VISIBLE_CUE_PROTOCOL.md).

Se desarrolló un ensayo de dos observaciones **reales** MiniGrid Memory
(S7 y S13): cámara forzada al objeto inicial, después al cruce mirando
hacia la salida superior, eligiendo arriba/abajo. Son imágenes parciales,
sin utilizar el estado verdadero del mundo como input. El entorno etiqueta
el objetivo solo para recompensa/entrenamiento supervisado; ninguna
política recibe dicho objetivo durante la inferencia.

**No es** el problema original de navegación de siete acciones: el entorno
mueve la cámara entre dos vistas. Esta distinción no se puede omitir.

## Auditoría de suficiencia informativa

Un experto simbólico **sin aprendizaje**, que extrae exclusivamente dos
píxeles de las mismas imágenes del agente (pista y objeto candidato),
acierta todas las parejas de prueba S7 y S13. Se comprobó además en los
tests que la observación del cruce permanece idéntica al invertir la pista,
que la salida correcta cambia y que un agente que elige siempre lo mismo
no puede acertar ambas condiciones de un par. No se recibe la etiqueta
oculta en el experto ni en las entradas de los modelos.

La primera implementación de cámara al cruce mirando al este resultó
inadecuada porque podía no presentar el objeto de la salida. Fue corregida
a **cámara mirando al norte con comprobación explícita de visibilidad**;
los resultados iniciales de aquel protocolo roto no se consideran
evidencia interpretable.

## F9-C por refuerzo: un resultado negativo

[CI PPO y RecurrentPPO reales, 20k con cámara corregida](https://github.com/webtilians/principiodelTodo/actions/runs/40885125879): tests de visibilidad y
control positivo pasados; el entrenamiento 20k/modelo generó:

| Modelo | S7 original | S7 señal invertida | Cambios de elección | S13 original |
|---|---:|---:|---:|---:|
| PPO | 50/100 = 50% | 50/100 = 50% | 0/100 | 16/30 = 53,3% |
| RecurrentPPO LSTM | 50/100 = 50% | 50/100 = 50% | 0/100 | 16/30 = 53,3% |

**No hay memoria adquirida por PPO/LSTM** en esta condición y presupuesto.
Una ejecución 100k anterior (GitHub Actions #40884217413)
utilizaba el diseño inicial defectuoso y NO permite inferir resultados
de la cámara corregida.

## F9-C supervisado — positivo verificado

[CI #40885330611](https://github.com/webtilians/principiodelTodo/actions/runs/40885330611):
**éxito**; [artifact report y modelos](https://github.com/webtilians/principiodelTodo/actions/runs/40885330611/artifacts/12902606171).

Se entrenaron **dos** modelos supervisados con el mismo conjunto de
entrenamiento: 1.200 semillas de generación independientes (train),
cada una con las dos variantes de pista original e invertida =
2.400 ejemplos perfectamente balanceados (1.200 arriba, 1.200 abajo),
25 épocas y entropía cruzada. La señal de entrenamiento es la decisión
correcta, pero en evaluación los modelos reciben SOLO dos imágenes
parciales codificadas como **98 canales de ocupación visible**
(49 píxeles × key/ball) por observación.

La red MLP usa exclusivamente **la última vista**; la GRU procesa
secuencialmente ambas vistas y puede conservar una representación interna.

| Modelo supervisado | S7 éxito original | S7 éxito señal invertida | Cambio de decisión S7 | S13 zero-shot éxito | Cambio S13 |
|---|---:|---:|---:|---:|---:|
| Sin memoria, solo vista actual | 50/100 | 50/100 | 0/100 | 16/30 | 0/30 |
| **GRU con memoria temporal** | **100/100** | **100/100** | **100/100** | **30/30** | **30/30** |

Además, la GRU acertó **ambos miembros de cada par contrafactual**
(100/100 S7; 30/30 S13). Sus checkpoints y el informe están en el
artifact de CI, con SHA-256. El test reservado original F9-B
(semillas 900.000+) **nunca se abrió**.

**Conclusión limitada:** hemos conseguido entrenar una red que retiene
eficazmente la pista de una observación anterior y la utiliza en una
decisión posterior. Demuestra que la observación contiene información
suficiente y que la tarea temporal es aprendible. **No** demuestra que
RecurrentPPO aprenda memoria con recompensa escasa, ni que la GRU haya
aprendido a desplazarse por el entorno original, ni ninguna ventaja de
F4/F5/INFINITO sobre una arquitectura convencional.

## Próximo objetivo falsable

Usar los resultados para preparar un **currículum de navegación real**:
primero un entorno con desplazamiento hasta la pista y la bifurcación,
después el MiniGrid Memory S7 original de siete acciones y finalmente
S13 sin entrenamiento adicional. PPO/LSTM deben competir bajo las mismas
observaciones, recompensas, presupuesto de interacción y semillas; el
control de memoria F4 se ensayará **solo después** de disponer de una
referencia de navegación que responda a la pista.

Si el problema de exploración continúa, puede estudiarse aprendizaje
por demostración o una fase auxiliar supervisada, pero se debe declarar
por separado el origen de cada ejemplo y nunca confundirlo con
entrenamiento PPO de recompensa pura.
