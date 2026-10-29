# F9-I — resultados verificados (29-10-2026)

## Conclusión

No se demuestra que el bajo éxito original en S13 se explique por pérdida
generalizada de la pista por FIFO, ni que la selección contextual resuelva
la retención inicial en trayectorias largas.

El control primario de 64 eventos obtiene 40% con FIFO y 45% con selección
contextual; ambos miembros correctos en solo 1/40 y 3/40 pares respectivamente.
La diferencia contextual por semilla es +5, +20, −5 y 0 puntos porcentuales.
Con 8 eventos, el aumento medio es mayor, pero se concentra en una semilla y
el primer evento de pista NO llega retenido a la bifurcación en ningún episodio
contextual S13. Una mejora conductual no prueba el mecanismo propuesto.

## Ejecución y trazabilidad

- [Entrenamientos y 19 tests](https://github.com/webtilians/principiodelTodo/actions/runs/40928547873):
  integridad y las ocho celdas completas aprobadas.
- Código experimental: `e5fd56ab794e77dbe680bec14ce8a5a276682da3`.
- [Verificación y agregación corregida](https://github.com/webtilians/principiodelTodo/actions/runs/40930660060):
  aprobada, con comprobación de las ocho celdas y su SHA.
- [Informe completo, JSON y Markdown](https://github.com/webtilians/principiodelTodo/actions/runs/40930660060/artifacts/12921428449).
  SHA256 ZIP: `ff0b07bca0013ea045fffad9ab50b3d24386140f88ef8f347c77d4b67020092b`.
- La agregación inicial falló por incluir el artifact smoke entre los ocho
  experimentos. Se corrigió el filtro y se verificaron los mismos artifacts,
  sin repetir entrenamientos ni seleccionar semillas.
- Los artifacts del entrenamiento incluyen checkpoints y trazas por episodio.
  El JSON adjunto en este repositorio conserva métricas y diferencias de las
  ocho celdas y los cuatro checkpoints originales. Los artifacts caducan en 30 días.

## Protocolo cumplido

Cuatro semillas, 600 mundos S7 con ambas pistas, 40 épocas BC y 20.000 pasos PPO
solicitados por política, mismo presupuesto efectivo entre políticas.
FIFO/contextual/reservoir/zero; 64 eventos primarios y 8 de sensibilidad.
660 entradas, misma arquitectura y parámetros entrenables incluso al reducir
el almacén; sin alterar reward, transiciones ni las siete acciones.
S7/S13 emparejados, diez pares por semilla/tamaño; S13 sin entrenamiento.
Splits y RNG preasignados; sin heldout, PC ni VPS.

Los 19 tests comprueban, entre otras cosas, paridad exacta del encoder FIFO
con F9-H, capacidad, identidad excluida de prioridad contextual, intervención
de exactamente dos canales de un único evento sin mutar el almacén,
splits y transiciones/recompensas nativas. También pasó un entrenamiento mínimo
completo. Ese smoke no aporta evidencia científica.

## Comparación autónoma después de PPO

Promedios de cuatro semillas; columna de pares sobre 40 pares en total.

| Eventos | Política | S7 éxito | S13 éxito | S13 ambos correctos |
|---:|---|---:|---:|---:|
| 64 | zero | 50.00% | 42.50% | 0/40 |
| 64 | fifo | 95.00% | 40.00% | 1/40 |
| 64 | context | 91.25% | 45.00% | 3/40 |
| 64 | random | 100.00% | 35.00% | 0/40 |
| 8 | zero | 50.00% | 35.00% | 0/40 |
| 8 | fifo | 98.75% | 53.75% | 10/40 |
| 8 | context | 96.25% | 72.50% | 21/40 |
| 8 | random | 97.50% | 52.50% | 6/40 |

Con 8 eventos, contextual−FIFO en S13 por semilla: −5, +5, +75, 0 puntos.
La media +18,75 puntos está dominada por la semilla 2; no es evidencia confirmatoria.
La mejora viene acompañada de una reducción S7 de 98,75% a 96,25%.

Retención S13 en la primera llegada a la bifurcación, sobre 80 episodios:
contextual8 conserva el primer evento 0 veces y alguna ocurrencia de pista
1 vez; FIFO8, 0 y 0; reservoir8, 44 y 66.
El recurrente puede conservar información que ya no está en el almacén.
No se ha aislado causalmente qué información mantiene su estado oculto.

Los controles zero S13 son 42,5% a capacidad nominal64 y 35% a nominal8,
aunque ambos reciben los mismos canales vacíos por construcción.
Los reentrenamientos en runners distintos no son réplicas numéricas exactas
(por ejemplo FIFO64 nuevo 40% frente al checkpoint original 38,75%).
No se ha aislado la causa de esa variación de ejecución; las semillas iguales
no garantizan equivalencia bit a bit. Por ello, la comparación entre capacidades
es una sensibilidad exploratoria, no un efecto causal puro de capacidad.
La comparación principal de selectores comparte runner, arquitectura,
presupuesto y semillas dentro de cada celda. No se seleccionó ningún checkpoint
por resultados de desarrollo.

## Diagnóstico de los checkpoints originales de F9-H

Se cargaron sin reentrenamiento los cuatro checkpoints y se reprodujeron
exactamente los éxitos por semilla:

- S7: 80%, 100%, 100%, 100%; media **95%**.
- S13: 50%, 25%, 45%, 35%; media **38,75%**.
- Corrección aritmética: la descripción inicial de este PR decía 36,25%;
  esa cifra era incorrecta. Los valores por semilla y la verificación automatizada
  siempre usan 50/25/45/35.

En S13 llegan a la bifurcación 67/80 episodios. De los 61 que habían escrito
una pista al llegar, 60 conservaban su primer evento y 61 alguna ocurrencia.
Los otros seis habían llegado sin escribir una pista hasta ese punto.
En las trazas completas hay pérdida del primer evento en 16/80 episodios,
pero pérdida de TODAS las ocurrencias de pista solo en 3/80.
Trece de los 16 con pérdida inicial no llegan a la bifurcación.
La pérdida tardía durante bucles no establece que FIFO causara el fallo.
No debe confundirse desaparición de un evento con pérdida total de información.

## Intervención en un único evento inicial

Mismos checkpoints originales, sin reentrenar. Solo cambia la identidad
key/ball del primer evento relevante retenido, cuando la pista no es visible.
No cambia otros eventos, píxeles, orden, edad ni estado recurrente directamente.
La navegación posterior puede divergir como consecuencia de esa intervención.

| Tarea | Normal | Corromper evento inicial | Cobertura |
|---|---:|---:|---:|
| S7 | 95% | 53,75% | 80/80 |
| S13 | 38,75% | 37,50% | 78/80 |

S7: éxito con corrupción por semilla 30%, 60%, 55%, 70%; ambos correctos
bajan de 36/40 a 15/40. Esto apoya sensibilidad causal a ese evento en S7
bajo esta intervención. No demuestra una memoria general ni ausencia de
efectos por distribución de entrada.
S13: éxito con corrupción 50%, 10%, 50%, 40%; efectos mixtos y media −1,25 puntos.
Otros eventos y la LSTM permanecen: un efecto nulo no demuestra ausencia de uso
de toda la memoria. La intervención original de swap global de F9-H no es
equivalente a esta alteración selectiva.

## LearnedRelevance real

Se inspeccionó y probó la clase real de `observatory/selective_memory.py`.
Usa ocho categorías y feedback ex post de la clave consultada. Los IDs de
ocurrencias F9-G/H son crecientes: una novena clave puede provocar IndexError
al puntuar, y reciclar claves sobreescribe eventos, rompiendo el control.
MiniGrid no proporciona un queried_key histórico observable con su reward.
No se inventó ese feedback a partir de la posición oculta de la pista.

La adaptación requiere separar ID de evento y categoría contextual observable,
y un contrato de consulta observable posterior a la decisión; aprendizaje
solo en train y congelación antes de evaluar. Reward escalar exigiría otro
método de asignación de crédito. La selección contextual implementada es
una regla fijada de antemano, **no LearnedRelevance**.

## Alcance

Desarrollo exploratorio de cuatro semillas, sin heldout ni afirmación
confirmatoria. No extrapolar el 95% S7 a S13. El próximo estudio debería
separar el uso del estado recurrente del uso del almacén y establecer primero
el feedback observable necesario para relevancia aprendida; los resultados
actuales no justifican afirmar que conservar el primer evento resuelve S13.
