# F9-G — Memoria F4 contextual de observaciones, antes de entrenar

**Fase:** desarrollo exploratorio, sin optimización de políticas ni apertura del holdout.

## Qué ha demostrado F9-F

En 40 pares MiniGrid Memory S7, el Store/FIFO de F9-E guardó llave
y pelota sin contexto de dónde/cuándo las observó. La lectura de los
12 canales fue **exactamente igual bajo inversión de la pista** en
40/40 pares antes y después de PPO. Un decodificador de dichos canales
no puede recuperar cuál era la pista entre dos mundos con idéntico
contenido. No implica que LSTM carezca de memoria ni prueba nada
sobre LearnedRelevance o F5, todavía no integrados.

## Qué cambia F9-G

El archivo observatory/f9g_contextual_memory.py incorpora un adaptador F4 de
*eventos visuales con contexto*. Recibe únicamente el vector público
148-dimensional original, que contiene imagen simbólica egocéntrica
7×7×3 y orientación. Cada evento almacena una fotografía de **98 bits**
(49 posiciones visibles × [llave,pelota]) que conserva *dónde* estaban
ambos tipos en el campo de visión, aunque aparezcan simultáneamente.

Tres slots permiten observar la evolución temporal. Se conservan las
dos primeras vistas distintas que contengan llave/pelota, y el tercero
representa la última vista distinta posterior. Cada slot incluye
la antigüedad (pasos desde su observación, normalizada) y un bit
de ocupación. El orden de los slots también codifica antigüedad.

Este diseño utiliza Event, Store y FIFO reales de F4, pero cambia
**de manera explícita** la política de admisión y actualización: dos
eventos tempranos quedan fijados y el tercero se refresca. **No es el
mismo FIFO de F9-E**, ni un selector de relevancia aprendido, ni F5.
Mantiene un presupuesto fijo de tres instantáneas; aumenta la dimensión
de memoria de 12 a **300** canales. Una futura comparación de rendimiento
debe incluir un control de **300 canales a cero** y el mismo número
de parámetros.

La representación es genérica, no contiene etiqueta de pista,
inicio de pasillo, meta, success_pos, failure_pos o coordenadas
absolutas. Nunca ve el futuro ni conoce la respuesta correcta.

## Gate de suficiencia informativa (sin entrenar)

Se utiliza el profesor BFS **únicamente en las pruebas** para
recorrer trayectorias nativas de siete acciones y localizar una
bifurcación donde, tras invertir la pista inicial, la **observación
actual es bit a bit idéntica** pero las acciones que enseñaría el
profesor divergen. El profesor no puede escribir ni leer el
contenido que recibe la memoria.

En los mismos mundos de desarrollo original/invertido se exige:

1. La nueva memoria se diferencia entre ambos miembros del par justo
   antes de la decisión que depende de la pista, aunque la observación
   actual sea idéntica.
2. Esa diferencia sobrevive al trayecto real hasta la bifurcación,
   sin teletransporte ni modificación de la recompensa.
3. El modo erase produce memoria de salida nula; swap_identity altera
   solo la lectura, no los 148 valores de observación actual.
4. El modo zero tiene la misma anchura que la memoria contextual
   y ningún canal informativo.
5. Reset limpia los tres eventos. Ningún sensor privilegiado entra
   en el codificador.

**Gate técnico**: 50 pares en S7 y 50 pares en S13, todos con memoria
distinta en el punto de decisión e imagen actual idéntica.
Se detendrá el desarrollo de entrenamiento si falla.

La prueba no acredita que una política entrenada utilice esos datos.
Solo acredita la *posibilidad de recuperar la diferencia* del contenido.

## Siguiente experimento después del gate

Comparación igualada en navegación autónoma con siete acciones:
- LSTM + 300 canales a cero.
- LSTM + memoria F4 contextual de 300 canales.
- El mismo checkpoint F4 con lectura borrada y con identidad alterada.

Deberá evaluarse éxito **en los dos miembros de cada par**,
cambios de rama al invertir pista, visibilidad y retención,
propiedades de navegación, S7 y S13 sin entrenamiento adicional.
La integración posterior con LearnedRelevance será un experimento
diferente, porque requiere feedback sin oracle.

Ninguna prueba aquí abre el conjunto reservado, usa Work,
modifica el PC o actualiza VPS.
