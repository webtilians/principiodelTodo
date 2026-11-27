# M2: memoria por capas con resúmenes, para historias de 20 años (protocolo)

Escrito el 24 de noviembre de 2026, antes de que exista el generador de datos, el harness o cualquier dato. Nada de
lo que sigue se cambia después de ver un resultado. Un arreglo del harness que no cambie un brazo, un parámetro o los
datos se anota en el informe. Las únicas decisiones abiertas están en «Qué puede decidir DEV», y cada una se toma en
DEV, con una regla escrita aquí, antes de abrir CONFIRM.

## Pregunta

La pregunta del mantenedor (memoria de INFINITO, 24/11/2026), con sus palabras: *la memoria debería ir de lo general
a lo específico. Las primeras capas, pequeñas y con lo más actual; debajo, esquemas por época; y un almacén más
profundo que no se visita siempre, al que se baja solo cuando hace falta un detalle antiguo* («¿con quién jugaba al
pádel en 2014?»). Y su objeción a M1: *no se trata de si mejora al RAG; o la información se estructura así, o no vas
a poder trabajar con 20 años de memoria.*

Precisión que este protocolo da por buena antes de medir: **ni el RAG ni INFINITO envían 20 años de contexto.** Cada
consulta manda un presupuesto fijo de tokens. Lo que crece con los años es otra cosa:

1. **Las preguntas que abarcan años** («¿qué deportes he practicado en mi vida?», «¿en cuántas ciudades he vivido?»).
   Su respuesta está repartida en miles de recuerdos sueltos, y ningún top-k de recuerdos sueltos la cubre con un
   presupuesto fijo. Una capa de resúmenes podría cubrirla en pocas líneas.
2. **Lo que el servidor tiene que puntuar y leer en cada consulta.** En plano, todo; por capas, solo lo que cuelga de
   las ramas elegidas.

M2 mide las dos cosas, con historias de 20 años, y compara siempre contra el RAG plano con el mismo presupuesto, como
exige [AGENTS.md](../AGENTS.md).

Tres hipótesis, que se juzgan por separado:

- **H1, preguntas que abarcan años.** Con 20 años de historia, la memoria por capas con resúmenes (L) responde las
  preguntas de alcance largo mucho mejor que el RAG plano (A0), sin perder en detalle, estado actual ni abstención.
- **H2, el crecimiento.** El RAG plano empeora en esas preguntas a medida que la historia crece de 2 a 20 años, y la
  memoria por capas no. Es la tesis del mantenedor: que la estructura es necesaria *por* el volumen.
- **H3, capas frente a resúmenes en un índice plano.** Bajar por capas (L) rinde como poner los mismos resúmenes en
  el índice plano (A0+), puntuando una fracción pequeña de los registros. Si es así, las capas son la arquitectura;
  si no, basta con añadir resúmenes al índice plano.

## Qué se sabe ya y qué no cubre (M1)

[M1](INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md) (resultados del 23/11/2026, rama
`experiment/m1-fractal-memory-run-20261120`, pendientes de PR) no pasó ninguna hipótesis: recorrer primero grupos o
un árbol de tiempo perdió cobertura frente al RAG plano con 1.024 tokens (A0 38,6 %, árbol temporal 37,1 %, grupos
descubiertos 26,3 % en preguntas relacionales). M2 no repite M1:

| | M1 | M2 |
|---|---|---|
| Historia | conversaciones de LoCoMo, unos 600 turnos y meses | 20 años, unos 22.000 recuerdos por persona |
| Capas | grupos (Louvain) o un árbol de sesiones, **sin resúmenes** | mes, año y vida, **con resúmenes** que citan sus fuentes |
| Preguntas | las de LoCoMo; ninguna pide agregar años | incluye preguntas de alcance largo con respuesta exacta |
| Escala | un solo tamaño | 2, 5, 10 y 20 años de la misma persona |
| Juez | un LLM (auditoría incompleta) | puntuación determinista, sin juez |
| Coste | ms por paquete | registros puntuados y bytes leídos por consulta |

Lo que M1 enseña y M2 respeta: el recorrido por grupos pierde lo que no cae en la rama elegida, así que L se compara
con A0 en preguntas de detalle con una puerta de no inferioridad. M1 también dejó escrito que no prueba la generación
de conceptos; M2 es esa prueba, en su forma más estrecha (resúmenes por periodo).

Otros antecedentes:

- **E13-A** (5/11): una vista derivada con procedencia contestó 632–638 de 640 consultas frente a 547 del RAG, casi
  todo en relaciones de dos saltos, pero su extracción estaba escrita para la gramática de su generador
  (FINDINGS, lección 2). Aquí los resúmenes los escribe un modelo que no conoce el generador.
- **Bemula-Lite** (3/11): olvidar una inferencia solo llegó al 29 % porque los hechos de origen seguían. Un resumen
  es una inferencia sobre sus fuentes: el olvido en cascada es una puerta dura.
- **Contrato de las reflexiones** (`src/infinito3/remote_reflections.py`): lo derivado cita sus fuentes y se borra
  con ellas. Los resúmenes de M2 siguen ese contrato.
- **Medición del 24/11/2026** en el PC del mantenedor: la búsqueda exacta sobre vectores de 256 números tarda 5,9 ms
  con 250.000 recuerdos y 288 ms con 10 millones. Puntuar todo no es caro a la escala de una persona; por eso el coste
  se informa pero no decide H1.

## Datos: historias sintéticas de 20 años

No existe un banco público de memorias personales de 20 años con respuestas exactas, y las conversaciones de CONFIRM
de LoCoMo ya están consumidas. M2 usa un generador propio, escrito y probado en DEV, y congelado antes de generar
CONFIRM.

**Personas.** 12, con semilla h_i = SHA-256 de `INFINITO/M2/v1/persona/<i>`, i = 0…11. Ordenadas por h: las 2 primeras
son DEV, las 8 siguientes CONFIRM y las 2 últimas reserva (no se abren en M2). Ninguna semilla de experimentos
anteriores se reutiliza.

**Calendario.** Cada historia va del 1/1/2006 al 31/12/2025. «Ahora» es el último día de la historia.

**Hilos de vida** (internos del generador; ningún brazo los ve):

| Hilo | Contenido |
|---|---|
| Residencia | 3 a 6 ciudades en 20 años, con barrio |
| Trabajo | 3 a 7 empresas, puesto y jefe |
| Deporte | 4 a 7 deportes con sus periodos, de 1 a 3 compañeros cada uno, club o lugar y frecuencia |
| Personas | amigos y pareja con nombre, que aparecen en varios hilos |
| Viajes | destino, año y acompañantes |
| Salud | lesiones ligadas a un deporte, con fecha |
| Únicos | por persona, 10 hechos que se dicen en un solo recuerdo, con un nombre que no aparece en ningún otro (para el olvido) |

Los nombres salen de listas fijas (ciudades de España y Latinoamérica, deportes, nombres de pila, empresas
inventadas), elegidos por la semilla.

**Recuerdos.** Una media de 3 al día (Poisson), unos 21.900 por persona. Alrededor del 70 % es vida diaria (comidas,
recados, series, tiempo, compras, reuniones), que da volumen y distractores: «vi el partido de fútbol» no es practicar
fútbol. Cada sesión de deporte nombra al compañero con probabilidad 0,3. Cada deporte activo deja al menos un recuerdo
al mes. Texto en español, primera persona, con al menos 20 plantillas por tipo de suceso y sinónimos. Cada recuerdo es
`(<fecha>) <texto>`, como en M1.

**Respuestas exactas.** Salen del estado de los hilos, nunca del texto. Un test comprueba que cada elemento de cada
respuesta aparece en al menos un recuerdo dentro de su ventana.

**Prefijos.** La misma persona se mide con 2, 5, 10 y 20 años de historia, siempre desde el 1/1/2006: el prefijo de N
años termina el 31/12 del año 2006 + N − 1, que es su «ahora». Las preguntas de cada prefijo se generan para su
«ahora» y solo con lo que existe en él.

**Tipos de pregunta** (plantillas fijas; 8 por tipo, persona y prefijo, ordenadas por SHA-256 de
`INFINITO/M2/v1/q/<persona>/<prefijo>/<tipo>/<índice>`; todas si hay menos):

| Tipo | Ejemplos | Respuesta |
|---|---|---|
| **S, alcance largo** | «¿Qué deportes he practicado en mi vida?», «¿En qué ciudades he vivido?», «¿En cuántas ciudades he vivido?», «¿A qué países he viajado?» | conjunto o número |
| **T, estado en una época** | «¿Qué deporte practicaba en 2014?», «¿Dónde trabajaba hace 6 años?» (al menos 2 años antes de «ahora») | conjunto |
| **D, detalle antiguo** | «¿Con quién jugaba al pádel en 2014?», «¿Con quién viajé a Lisboa?» (evidencia de al menos 3 años antes de «ahora»; en el prefijo de 2 años, cualquiera) | conjunto |
| **C, estado actual** | «¿Qué deporte practico ahora?», «¿Dónde vivo?» | conjunto |
| **A, sin respuesta** | las plantillas de D y T con un deporte, año o destino que no existen en esa persona | «No consta» |

## Escritura en memoria

Cada persona y prefijo tiene su propio almacén en el harness, con los mismos vectores para todos los brazos
(`MeaningRanker`, potion-multilingual-128M int8, la carpeta de R1 y M1). No se usa la ruta de escritura del producto:
el 24/11/2026, un bucle de 100, 1.000 y 5.000 llamadas a `AgentMemory.remember` no terminó en 10 minutos en el PC del
mantenedor, y aquí hacen falta unas 260.000 escrituras. Eso queda como dato para el producto, fuera de M2. Los tokens
se cuentan como en M1: ceil(caracteres / 4) sobre todo el paquete, incluidos ids, fechas y etiquetas.

## Resúmenes (la capa general)

Los escribe **`llama3.1:8b`** (digest `46e0c10c…`, ya instalado), de otra familia que el lector. Es un modelo más
pequeño que el del motor nocturno, y se elige por coste: si sesga algo, sesga **en contra** de L. Ollama, temperatura
0, semilla fija, contexto de 8.192, sin razonamiento visible.

| Nivel | Entrada | Límite | Cita |
|---|---|---|---|
| Mes | los recuerdos del mes, en orden, con sus ids | 60 palabras | ids de recuerdos |
| Año | los 12 resúmenes de mes | 100 palabras | ids de mes |
| Vida | los resúmenes de año del prefijo | 150 palabras | ids de año |

Instrucción (fija): «Resume lo importante de este periodo para recordarlo años después: actividades habituales,
cambios, personas, lugares y hechos que no se repiten. Cada frase termina con los ids que la apoyan entre corchetes. No
añadas nada que no esté en el texto. Como máximo N palabras.» Si la entrada de un mes pasa de 6.000 tokens, se resume
en dos mitades de 30 palabras que se concatenan. Nunca se trunca.

El modelo no recibe hilos, categorías ni preguntas. Los resúmenes de mes y año de un periodo son los mismos en todos
los prefijos que lo contienen; el de vida se escribe una vez por prefijo.

## Brazos

Todos terminan en un **paquete** de como mucho B tokens. Todos usan el mismo analizador de tiempo: si la pregunta
nombra un año («en 2014»), «hace N años» o «el año pasado», cada brazo solo considera lo fechado en ese año (L
conserva además su resumen de vida). Así el filtro por fecha no favorece a nadie.

| Brazo | Qué hace |
|---|---|
| **A0, RAG plano** (línea base) | todos los recuerdos ordenados por significado; se añaden en ese orden mientras caben; se muestran en orden de fecha |
| **A0+, resúmenes en el índice plano** | un solo índice con los recuerdos y todos los resúmenes (mes, año y vida), ordenados por significado; cada resumen lleva su periodo |
| **L, por capas** | el descenso de abajo; no puntúa nada fuera de las ramas que elige |
| L-barajada (control, solo fase 1) | L con los resúmenes de mes y de año permutados entre periodos de la misma persona (semilla 20261124) y sin sus citas, para que una cita no devuelva a los originales correctos |

**Descenso de L:**

1. Resumen de vida, siempre.
2. Resúmenes de año ordenados por similitud con la pregunta. Se añaden en ese orden, como mucho 3, mientras los
   resúmenes no pasen de B / 2.
3. Del mejor año incluido, sus resúmenes de mes ordenados por similitud. Se añaden como mucho 3, con el mismo límite
   de B / 2.
4. Recuerdos originales candidatos: los de los meses incluidos, los citados por esos resúmenes de mes y los de los
   últimos 90 días (la capa «ahora»). Se ordenan por similitud y se añaden mientras caben en B.
5. Si queda presupuesto, se repiten los pasos 3 y 4 con el siguiente año incluido, hasta 3 años.

L no tiene relleno plano. Lo que no cuelga de las ramas elegidas no se puntúa ni se lee, que es justo la propuesta del
mantenedor. El paquete muestra primero los resúmenes (vida, años, meses, con su periodo) y luego los originales en
orden de fecha.

Todos los números (60, 100 y 150 palabras, 3 años, 3 meses, B / 2, 90 días, 6.000 tokens) se fijan ahora, sin ver
datos.

**Presupuesto.** B = 1.024 tokens es el decisivo salvo que la regla de DEV lo baje a 512. Se informa también de 512 y
2.048.

**Traza.** Por pregunta y brazo: ids incluidos con la ruta que los trajo (vida, año, mes, original citado, original
del mes, capa «ahora», índice plano), tokens usados, registros puntuados y bytes de texto leídos.

## Modelos (locales, sin API de pago)

- **Lector** (todos los brazos): `qwen3.8-motor`, digest `7991c412…`, como en M1. Ollama, temperatura 0, semilla
  fija, contexto de 8.192. Instrucción fija: «Responde usando solo la memoria de abajo. Si no contiene la respuesta,
  responde exactamente: No consta. Si la pregunta pide varios elementos, sepáralos con punto y coma. Si pide un número,
  responde solo el número. Como máximo 25 palabras.»
- **Resúmenes:** `llama3.1:8b`, arriba.
- **Sin juez.** La puntuación es determinista (abajo). Es la lección de M1: un juez sin auditar deja la fase 2 en
  INCONCLUSIVE.

## Métricas

**Fase 1, sin modelo lector** (todas las preguntas, todos los brazos, los cuatro prefijos):

- **Cobertura de elementos:** fracción de los elementos de la respuesta cuyo nombre aparece en el texto del paquete
  (originales o resúmenes), tras pasar a minúsculas y quitar tildes. **Cobertura completa:** todos los elementos.
- Para D, además, la **cobertura de evidencia**: si está alguno de los recuerdos originales que dicen el dato.
- **Coste por consulta:** registros puntuados, registros leídos y bytes leídos.
- Coste de construir: llamadas al modelo de resúmenes, segundos y tokens.

**Fase 2, el lector responde** (prefijos de 5 y 20 años; A0, A0+ y L):

- Conjuntos (S, T, D, C): F1 entre los elementos de la respuesta y los de la referencia. Un elemento cuenta si su
  nombre normalizado aparece en la respuesta. Se informa también del acierto exacto del conjunto.
- Números: acierto exacto.
- A: correcto solo si la respuesta es exactamente «No consta». Cualquier otra cosa es un recuerdo falso.

**Fidelidad de los resúmenes** (todos los resúmenes de las personas de CONFIRM):

- **Citas válidas:** fracción de ids citados que existen en el periodo del resumen.
- **Nombres inventados:** fracción de los nombres de las listas del generador que salen en un resumen sin aparecer en
  ninguna de sus fuentes.

**Olvido** (copia aparte de cada persona de CONFIRM, prefijo de 20 años). Se olvidan los 5 hechos únicos con menor
SHA-256 de `INFINITO/M2/v1/forget/<persona>/<índice>`: se borra su recuerdo y, en cascada, se reescriben el resumen de
su mes, el de su año y el de vida, sin esa fuente. Las 5 preguntas sobre esos hechos se hacen a A0, A0+ y L.
Comprobaciones, todas deterministas, porque el nombre de un hecho único no aparece en ningún otro recuerdo:

1. ningún resumen, de ningún nivel, contiene el nombre olvidado;
2. ningún paquete lo contiene;
3. ninguna respuesta lo contiene;
4. ningún resumen cita el id borrado.

**Incertidumbre.** Diferencias pareadas con intervalo del 95 % por bootstrap sobre las 8 personas de CONFIRM (5.000
remuestreos, semilla 20261124). Todo se informa también por persona.

## Tests del harness (escritos y en verde antes de DEV)

- Determinismo: la misma semilla da la misma historia, las mismas preguntas y los mismos paquetes.
- Respuestas: cada elemento de cada respuesta aparece en un recuerdo dentro de su ventana; los hechos únicos aparecen
  en un solo recuerdo.
- Presupuesto: ningún paquete pasa de B y ningún registro sale dos veces.
- L no puntúa nada fuera de sus ramas (lo comprueba la traza en un almacén pequeño hecho a mano).
- Analizador de tiempo, con ejemplos fijos.
- Puntuación: casos escritos a mano de conjuntos, números, abstención y tildes.
- Olvido en cascada en un almacén pequeño: las cuatro comprobaciones.
- Sin fugas: el escritor de resúmenes y los brazos no reciben hilos, preguntas ni respuestas (por firma y con un test
  que falla si leen el fichero de preguntas).

## Qué puede decidir DEV (solo con estas reglas, antes de CONFIRM)

1. **Errores del generador** en los que la respuesta no coincide con el texto, encontrados leyendo 20 recuerdos y
   10 preguntas por tipo de cada persona DEV. Se corrigen antes de congelar y se anotan. No se cambian densidades,
   tipos ni plantillas de pregunta.
2. **El presupuesto decisivo:** B = 1.024 salvo que A0 tenga, en DEV y con 20 años, una cobertura completa de S de al
   menos el 80 %. Entonces B = 512, para no decidir en el techo. Solo se mira A0.
3. **El análisis de las salidas** del lector y del escritor de resúmenes, nunca el contenido de las instrucciones.
4. **Comprobar la puntuación:** el mantenedor puntúa a mano 30 respuestas de DEV elegidas por SHA-256, sin ver el
   brazo, antes de congelar. Si coincide con la puntuación automática en menos de 28, se corrige el análisis de
   salidas y se repite con otras 30. Sin ese paso no hay congelación.
5. **El límite de tiempo:** si DEV extrapola CONFIRM a más de 48 horas de cálculo, la fase 2 baja de 8 a 6 preguntas
   por tipo.

Después de DEV, un commit de congelación guarda el generador, el harness, los tests, las instrucciones, los digests de
los modelos, el presupuesto y las salidas de DEV. El runner se niega a generar las personas de CONFIRM si ese commit no
está en Git.

## Puertas (en CONFIRM, con el presupuesto decisivo)

Cada diferencia es la cifra del brazo menos la del comparador, en puntos. «IC > 0» quiere decir que el extremo
inferior del intervalo del 95 % está por encima de cero.

**H1 pasa si, con 20 años:**

1. la cobertura completa de S de L es ≥ A0 + 15, IC > 0;
2. el F1 de S de L es ≥ A0 + 15, IC > 0;
3. no inferioridad: el F1 de D, T y C de L no queda más de 3 puntos por debajo de A0 en ninguno; la cobertura de
   evidencia de D, tampoco;
4. la abstención correcta en A no queda más de 3 puntos por debajo de A0.

**H2 pasa si**, en cobertura completa de S:

1. A0 con 20 años queda al menos 10 puntos por debajo de A0 con 2 años, IC > 0;
2. L con 20 años no queda más de 5 puntos por debajo de L con 2 años;
3. la ventaja de L sobre A0 con 20 años supera en al menos 10 puntos a la ventaja con 2 años, IC > 0.

**H3 pasa si, con 20 años**, L no queda más de 3 puntos por debajo de A0+ en el F1 de S, D, T y C ni en la abstención,
y la mediana de registros puntuados por L es como mucho el 10 % de la de A0+.

**Puertas duras. Si falla una, no pasa ningún brazo:**

- **olvido:** las cuatro comprobaciones no encuentran nada en ningún brazo;
- **fidelidad:** citas válidas ≥ 95 % y nombres inventados ≤ 2 %, sobre todos los resúmenes de CONFIRM;
- **puntuación:** el paso 4 de DEV se hizo y pasó;
- **integridad:** CONFIRM se generó después del commit de congelación, los modelos coinciden con sus digests y cada
  paquete está en la traza.

L-barajada no tiene puerta: si L-barajada cubre S tanto como L, la ganancia no viene del contenido de los resúmenes, y
el informe lo dice así.

## Qué cambia cada resultado

- **H1 y H3 pasan:** la memoria por capas es la arquitectura. Un protocolo de producto (flags apagados, primero en
  sombra) haría que el motor nocturno escriba resúmenes de mes, año y vida como registros derivados, con el contrato de
  las reflexiones (citan sus fuentes y se reescriben al olvidar), y que `recall` baje por ellos. Después, y solo cuando
  el volumen lo pida, el almacenamiento por bloques cifrados siguiendo las mismas capas; eso es ingeniería y se
  comprueba con tests de equivalencia, no con otro experimento.
- **H1 pasa y H3 no:** los resúmenes ayudan, pero bajar por capas pierde algo. El producto añadiría los resúmenes al
  índice plano (A0+), y el coste de escala se resolvería con un índice aproximado, no con capas.
- **H2 pasa:** queda medida la tesis del mantenedor: el plano se degrada con los años en las preguntas de alcance largo
  y las capas no. Sin H2, la ventaja de los resúmenes no depende de la escala y se presenta así.
- **H1 no pasa:** con este escritor de resúmenes, la capa general no cubre esas preguntas mejor que el plano. Se cierra
  esta forma. Probar un escritor más fuerte exige un protocolo nuevo con las personas de reserva, no repetir CONFIRM.
- **Falla la no inferioridad en D, T o C:** las capas no sustituyen al plano. Como mucho se usan para las preguntas de
  alcance largo, y decidir cuándo usarlas sería otro protocolo.
- **Falla una puerta dura:** no se integra nada, sea cual sea la precisión. Primero se arregla lo que falló.

Un resultado positivo es la prueba para proponer un cambio de producto, no el cambio. El núcleo sigue siendo solo
biblioteca estándar: los resúmenes serían datos escritos por el motor, no una dependencia de `src/infinito3/`.

## Límites conocidos de antemano

- **Datos sintéticos.** El generador y los brazos los escribe el mismo diseñador. Se mitiga con tres cosas: los brazos
  y el escritor de resúmenes no ven los hilos, la estructura de L es solo de tiempo (mes, año, vida) y no de temas, y la
  puntuación sale de los hilos, no del texto. Aun así, un aprobado prueba el mecanismo, no su tamaño en vidas reales.
  Hoy no existe ninguna memoria real de 20 años, y la del mantenedor tiene tres semanas.
- **Plantillas.** Por variadas que sean, un texto de plantilla es más regular que el real, lo que puede favorecer a A0
  (las palabras se repiten) y a los resúmenes (los sucesos son claros).
- **Escritor de resúmenes de 8B**, elegido por coste; el producto usaría el del motor nocturno.
- **Un lector local de 27B**, no los asistentes que usa la gente.
- **Solo español** y una densidad supuesta de 3 recuerdos al día.
- **Ocho personas** son pocas unidades independientes, y el intervalo se calcula sobre ellas.
- **M2 no prueba el almacenamiento:** ni el cifrado por bloques ni los índices aproximados. Solo qué se elige y cuánto
  se puntúa.

## Coste y alcance

Sin modelos de pago y sin red: los dos modelos ya están instalados y los datos se generan en local. Sin datos
personales, sin cambios en `src/infinito3/` y sin despliegue. Se ejecuta en el PC del mantenedor, enchufado, fuera de
las horas del motor nocturno.

Estimación, que DEV mide de verdad: por persona, unas 264 llamadas de resumen (240 de mes, 20 de año y 4 de vida),
hasta 240 del lector (40 preguntas × 2 prefijos × 3 brazos) y unas 30 de olvido. Con las medianas de M1 (11,7 s el
lector) y unos 10 s por resumen, sin medir todavía, sale entre 1,5 y 2 horas por persona: unas 3 horas de DEV y entre
12 y 16 de CONFIRM.

M2 se aparta del siguiente paso que dejó M1 (la comparación pública, punto 2 de la hoja de ruta) porque el mantenedor
lo pidió el 24/11/2026. No lo bloquea, porque usa otros datos, pero compite con él por horas de cálculo en el mismo PC.

Reutiliza del harness de M1 (`observatory/m1_fractal_memory.py`, commit `87292ea`, todavía sin publicar) la llamada a
Ollama con reintentos, el guardado con reanudación, la petición a Windows para no suspender, los digests y el
bootstrap. Por eso el PR de resultados de M1 va antes que el código de M2.

Código: `observatory/m2_layered_memory.py`, con tests en `tests/`. Informe:
`docs/INFINITO_M2_LAYERED_MEMORY_RESULTS_<fecha>.md`, que se publica salga lo que salga.
