# INFINITO F9-Z-A — Factibilidad de positividad causal por actor, pista e historial

**PREREGISTRO 2026-11-02, antes de escribir código experimental o abrir mundos 11_100_000+.** F9-Y devolvió `INCONCLUSIVE_SUPPORT`, fundamentalmente por falta de D+ en múltiples celdas actor×cue×historia. No se corrige haciendo MLP mayor, cambiando umbral o filtrando por resultado.

## Pregunta experimental
¿Puede la variación prospectiva de dos variables físicas **no basadas en etiquetas** —capacidad de memoria F4 y duración de la consulta F4 en el cruce— generar, para cada actor F9-P original y cada celda cue×historia, tanto experiencias donde consultar ayuda (`D=+1`) como experiencias donde no ayuda (`D=0/-1`), sin elegir episodios en función de D?

**Esto es un PILOTO DE FACTIBILIDAD de una tarea nueva, no una repetición de F9-Y ni una prueba de que H/C representa valor abstracto.** Hasta comprobar el soporte, no se entrenará ninguna nueva cabeza.

## Diseño fijo, íntegramente factorial
- Actores F9-P originales índices 0..3, pesos exactos y hashes originales bloqueados, entorno original `MiniGrid-MemoryS13-v0`. Actor y selector F4 `SelectiveF4("context", capacity=N)` sin modificación de parámetros entrenables o de la lógica del selector.
- Mundo piloto `11_100_000 + 10_000*actor_index + j`, `j=0..23`. Exactamente 24 nuevos mundos por actor, 96 mundos únicos; para cada uno, dos intervenciones de cue contrafactual e idénticas geometrías y dos regímenes de historia (F4 normal o F4 bloqueada desde postcue).
- Factor A **capacidad F4 en número de eventos**: `N ∈ {1,4,8}`. La anchura de entrada al actor siempre es 660 (el resto queda a 0), como F9-I. El selector es el mismo para todos; únicamente cambia el máximo de eventos retenidos.
- Factor B **ventana READ**: `W ∈ {1,4,8}` pasos autónomos a partir del cruce. `READ_W` recibe el readout del escritor F4 durante los primeros W pasos autónomos y ceros después. `SKIP_W` recibe F4=0 **en todos** los pasos autónomos. Ambos parten de exactamente el mismo estado pre-cruce, escritor, entorno y vista pública; el escritor sigue actualizándose en las dos ramas independientemente de si se lee, y se detiene la evaluación al elegir el primer brazo o al terminar el episodio. Un READ de W pasos es una **nueva intervención de política de consulta**, no la misma decisión de un paso de F9-V.
- Todos los `3×3=9` tratamientos, en los `4` estados del mundo, en las cuatro réplicas y los 24 mundos, **sin excluir** casos D=0 ni resultados desfavorables. Máximo `96×4×9=3456` casos experimentales, con `6912` continuaciones READ/SKIP por pares. Los estados comparten mundo, no son independientes estadísticamente.
- Referencia verificable: `capacity=8, W=1` debe coincidir **bit a bit** (decisión y primer brazo) con la implementación original F9-V `one_seed_cases` para una comprobación de mundo por actor, sin evaluar semillas antiguas.
- No se introducen pistas de calidad, etiquetas de D ni contenidos falsificados. La capacidad y la ventana son configuraciones **externas de intervención** que deberá conocer cualquier futuro controlador que seleccione ventanas; no afirmar que ya se codifican en H/C. Se analizará el caso con ventana como tratamiento, no como información intrínseca oculta del estado.

## Variables, contabilidad y validez
`D_{N,W} = 1[primera_salida_correcta(READ_W)] - 1[primera_salida_correcta(SKIP_W)] ∈ {-1,0,+1}`. Comparar `SKIP_W` entre ventanas del mismo mundo/estado/capacidad; deben ser exactamente iguales al ser idénticas ramas sin lectura. Comparar los estados `(cue_original, cue_invertido)` con la misma vista pública en el fork. H/C puede diferir por cue/régimen/capacidad **porque es el efecto causal de sus historias**, no por información futura.

Para cada uno de los 16 estratos `actor × cue × régimen`, contar `D+ / D0 / D-` en cada **tratamiento individual** `(N,W)`, denominador fijo 24 mundos. Reportar opcionalmente agregados marginales (sin usar para decisión), separando número de mundos únicos y episodios dependientes. Incluidos los efectos perjudiciales.

## Gate ex ante del piloto (y stop)
- *Soporte positivo para un tratamiento común*: existe al menos **un único par N,W** que aporta, dentro de cada una de las 16 celdas `actor×cue×régimen`, **>=3 mundos D+ y >=3 mundos D<=0**. La comprobación recorre los nueve tratamientos completos; no se permite seleccionar un tratamiento por celda, actor o mundo.
- *Reproducibilidad contractual*: todas las comparaciones emparejadas verificadas, cada SKIP idéntico entre W, y `capacity=8,W=1` reproduce F9-V nuevo con idéntico actor/fork.
- Solo si ambos se cumplen, declarar **`SUPPORT_FEASIBLE_EXPLORATORY`**. La elección del tratamiento en este piloto es exploratoria y exige otro protocolo prospectivo nuevo y mundos completamente distintos para TRAIN/DEV. Fijar orden de desempate `(8,1),(8,4),(8,8),(4,1),(4,4),(4,8),(1,1),(1,4),(1,8)`.
- Si no existe ese par, **`SUPPORT_NOT_ACHIEVED`**; no abrir confirmaciones ni entrenar predictor nuevo con los mundos piloto. Separar explícitamente si falta soporte pese a W=8 o si los actores no cambian la decisión ante F4.
- Si no se puede validar la invariancia SKIP/fork/anchoring, **`INVALID_EXECUTION`**, con auditoría, sin interpretar los conteos.
- Un resultado negativo **no puede arreglarse** aumentando después ventanas, capacidades, cambiando el criterio de 3/3 o seleccionando subconjuntos dentro de los mismos mundos para presentar un PASS. El siguiente estudio, de existir, tendrá que declarar una tarea diferente de forma prospectiva.

## Exclusiones y semillas futuras
Ninguna semilla de F9-V 8100000/8200000/8300000+, F9-W 9100000/9200000/9300000+, F9-X 9500000/9600000+, F9-Y 10100000/10200000/10300000+, ni F9 histórico 900000+, F9-T 6300000+, F9-U 7300000+. Otras fases potenciales `11_200_000+` TRAIN, `11_300_000+` DEV, `11_400_000+` CONFIRM quedan **reservadas** y no tendrán ejecutor ni se abrirán en este piloto.

Este diseño busca **verificar si es posible obtener positividad**. No garantiza que los actores congelados sean físicamente capaces de aprovechar la lectura en todas las situaciones, y no inventará un efecto positivo si su política es insensible a la memoria.
