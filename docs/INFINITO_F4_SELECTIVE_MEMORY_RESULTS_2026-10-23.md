# INFINITO F4 — Aprender relevancia para memoria selectiva (sin oráculo)

**Fecha:** 23 octubre 2026  
**Rama aislada:** \`experiment/emergence-observatory-v0-1\`  
**Código:** [observatory/selective_memory.py](../observatory/selective_memory.py)  
**Pruebas:** [tests/test_selective_memory.py](../tests/test_selective_memory.py)  
**GitHub Actions (24 semillas):** https://github.com/webtilians/principiodelTodo/actions/runs/39862304502

## Motivación

F3 reveló que conservar el primer recuerdo y no sobrescribirlo
permitía recuperar información diferida que se perdía con escrituras
continuas: 50% -> 81,25% en la sonda V5.1 reducida. Pero
esa regla **conocía** que el primer estímulo sería el útil.

F4 elimina tal conocimiento del controlador. La decisión sobre
qué guardar utiliza únicamente el acontecimiento **actual**, su
memoria limitada y consultas **anteriores**. La consulta futura
es inaccesible hasta que ha terminado la fase de almacenamiento.

**Nota de alcance:** F4 es una **prueba aislada de política de
priorización en memoria simbólica clave/valor**. Todavía NO es una
versión entrenada de la red V5.1 ni mide el rendimiento
end-to-end de sus 16 slots de escritura distribuida.

## Hipótesis evaluables

H1: Una política que aprende de necesidades anteriores de
recuperación supera a FIFO y reservorio aleatorio en la retención
de acontecimientos útiles cuando hay regularidades no anunciadas.

H2: Cuando cambian las necesidades de información sin aviso,
una política con actualización continua se adapta más que una
política entrenada solo en el primer contexto y congelada.

H3 negativa: Si las futuras consultas son verdaderamente uniformes
e independientes del pasado, ningún mecanismo de selección puede
identificar un subconjunto más útil de claves.

## Entorno, observabilidad y presupuestos

- Ocho tipos de evento, con una clave observable y un bit aleatorio
  independiente como contenido. Cada episodio presenta los ocho
  en un orden aleatorio; no existe una posición que identifique
  a los relevantes.
- Capacidad estricta **tres** elementos (3 de 8).
- La pregunta sobre el contenido de un evento llega **después**
  de que se hayan tomado **todas** las decisiones de guardado.
- Durante 120 episodios, dos tipos ocultos al selector reciben
  el 86% de las consultas (el resto el 14%). Cambian los dos
  tipos dominantes en los episodios 120 y 240, sin avisar a
  la política ni reiniciar su conocimiento.
- Después de 360 episodios, se congela el entrenamiento y se
  evalúan 160 episodios nuevos por semilla en el último régimen.
- 24 semillas independientes con pares de tipos distintos en
  cada una. Misma secuencia, preguntas y contenido para los
  cuatro algoritmos de cada semilla.

**Políticas:** frecuencia de consultas con olvido exponencial
(decay 0,97), FIFO, reservorio aleatorio, y frecuencia aprendida
solo durante el primer régimen, después congelada.

Aprendizaje online: tras contestar una pregunta, el selector
recibe **la clave que se consultó** (haya acertado o fallado)
y actualiza su expectativa de demanda. No recibe futuras claves,
etiquetas «relevante» ni la regla de cambio. Las señales de
feedback son exógenas a la decisión futura, pero están
disponibles **después** de la decisión del episodio actual.

Este algoritmo **aprende probabilidades de demanda asociadas a
claves**, no la semántica de los recuerdos. No es curiosidad
espontánea ni optimización neurobiológica de consciencia.

## Resultado: recuperación diferida en el conjunto no visto

| Política | Hit de clave solicitada | Acierto de valor binario* |
|---|---:|---:|
| **Aprendizaje adaptativo** | **87,99%** | **94,04%** |
| Reservorio aleatorio | 37,76% | 68,62% |
| FIFO (últimos tres) | 37,06% | 68,88% |
| Política inicial congelada (régimen 1) | 18,59% | 59,97% |

*Si falta el dato, el sistema adivina un bit; la métrica primaria
es **hit** porque refleja si el recuerdo existía, no la suerte
con la respuesta binaria.

Diferencia emparejada adaptativa menos aleatoria:
**+50,23 puntos porcentuales**; bootstrap al nivel de semilla,
intervalo exploratorio del 95% **[48,93; 51,67] puntos**.
Son 24 unidades de aleatoriedad independientes, no se toman
los episodios como 3.840 unidades independientes para calcular
ese intervalo.

Diferencia adaptativa - FIFO: +50,94 puntos, IC exploratorio
[49,11; 52,73].

**Evolución tras los cambios ocultos:**

| Régimen | Primeras 25 consultas | Últimas 25 |
|---|---:|---:|
| 0 | 83,50% | 88,50% |
| 1 (nuevo) | 47,33% | 89,17% |
| 2 (nuevo) | 49,00% | 89,17% |

Al cambiar el patrón, la política pierde acierto al principio
y posteriormente se adapta por la experiencia de las consultas
anteriores. La actualización exponencial evita el bloqueo en
la utilidad aprendida del primer régimen.

## Control negativo crucial

Se genera el mismo tipo de prueba pero haciendo las preguntas
**uniformes** (sin relación con las consultas anteriores).

| Política | Recuperación, datos no vistos |
|---|---:|
| Adaptativa | 35,65% |
| Reservorio | 37,11% |
| FIFO | 37,79% |
| Estática | 37,16% |

El techo teórico de un selector que guarda tres claves cualesquiera
ante una pregunta independiente y uniforme es **3/8 = 37,5%**.

La diferencia adaptativa - reservorio fue -1,46 puntos,
intervalo exploratorio bootstrap [-3,83; +0,73].
Esto concuerda con la ausencia de una señal aprendible; evita
confundir el mero funcionamiento del mecanismo con un descubrimiento
de importancia cuando no existe.

## Estado de las pruebas

- 7 tests unitarios completados en la ejecución GitHub Actions.
- 24 semillas en demanda cambiante + 24 semillas en control negativo.
- Ejecutable con Python estándar, sin dependencias ML:
  \`python -m observatory.selective_memory --seeds 24\`
- Datos auditables: \`report.json\` y trazas individuales de decisiones
  en \`drifting_relevance_per_seed.jsonl\` y
  \`uniform_negative_control_per_seed.jsonl\` como artefactos de Actions.
- Ninguna modificación a master ni a V5.1 original.

## Lo que demuestra, lo que no

**Sí:** una política **con memoria estadística de usos pasados**
puede aprender a priorizar eventos sin etiquetas prefijadas de
«importancia», con presupuesto estricto y utilidad cambiante.

**No:** no demuestra emergencia libre ni conciencia. Hemos
elegido expresamente un ambiente donde el tipo (clave) del evento
predice probabilísticamente qué se preguntará. Tampoco se ha
aprendido qué significa un acontecimiento ni si debe priorizarse
sin feedback explícito de consultas anteriores.

La precisión en test está condicionada por permanecer en el último
régimen aprendido. No se ha probado generalización a claves
totalmente nuevas o a un mundo donde la importancia depende
de relaciones causales semánticas complejas. Solo validamos
un primer mecanismo explicable de utilidad futura.

## Próximas fases delimitadas (no se presentan como completadas)

**F5 — memoria neuronal:** enlazar el selector a
\`EnhancedExternalMemory\` mediante una política de escritura con
límite de slots real, sin permitir que una escritura difusa
conserve recuerdos supuestamente «expulsados». Instrumentar
lectura, escritura, recuperación y predicción bajo igual costo.

**F6 — relevancia contextual:** sustituir el ID como pista
suficiente por características y relaciones aprendibles,
introducir nuevos tipos de eventos, señales de interés
no explícitas y consultas condicionadas al estado del entorno.

**F7 — curiosidad epistémica:** solo después estudiar si un
sistema busca datos nuevos para reducir incertidumbre sobre el
mundo o sus propios estados, con controles contra rewards
artificiales y respuestas disparadas por triggers.

La investigación sigue en términos de **mecanismos verificables**.
