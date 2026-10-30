# F9-E — Primera integración exploratoria de F4 en navegación autónoma

Fecha de inicio: 29/10/2026. Rama: experiment/f9e-f4-autonomous.
Base: experiment/f9b-behavior-audit (F9-D y F9-C fusionadas).
Esta rama no debe confundirse con master/main.

## Pregunta experimental

¿Almacenar de forma explícita los símbolos visibles con el
F4 real Event → FIFO → Store mejora la toma de decisiones de una LSTM que
navega sin ayuda en MiniGrid-MemoryS7-v0? ¿Desaparece cualquier aparente
ventaja si el almacenamiento se elimina o se altera el objeto recordado?

**No esperar** a que una LSTM alcance primero un umbral arbitrario para
explorar F4; distinguir exploración de ensayo confirmatorio.

## Condiciones entrenadas por separado

1. raw: RecurrentPPO/LSTM con 148 variables de la observación pública original.
2. zero: **misma** RecurrentPPO/LSTM, 160 variables, 12 canales adicionales a
   cero; control principal para el tamaño de la entrada.
3. f4: misma red de 160 variables, 12 canales procedentes del F4 real
   selectivo FIFO de capacidad tres (tres slots × [key, ball, color, ocupado]).
   El extractor ve únicamente tipos y colores presentes en la imagen
   parcial original. No ve coordenadas absolutas, meta, etiqueta de pista
   ni futuro. No impone un criterio oculto de relevancia.

Una escritura F4 guarda las identidades llave/pelota (y color) observadas
y conserva el orden FIFO del **primer avistamiento**; no reconoce por
construcción cuál de ellas era la pista del cuarto inicial. Si hay
ambigüedad en lo observado, debe quedar reflejada en el resultado.
Se utiliza F4 Store/Event/FIFO real de observatory/selective_memory.py;
**NO** se evalúa todavía LearnedRelevance ni F5 predictiva.

## Intervenciones causales sobre el MISMO checkpoint F4

- **erase**: se sigue procesando el episodio pero los doce canales de
  salida del almacenamiento quedan en cero; la vista actual no cambia.
- **swap_identity**: la identidad key/ball se invierte **solo en la
  salida almacenada**, conservando ocupación, slots y color. Se trata
  de una intervención de distribución fuera de entrenamiento; una
  degradación, por sí sola, no demuestra causalidad específica.
- **mask_cue**: se borran los píxeles que cambian bajo el intercambio
  de la pista inicial en la misma pose antes de escribir en F4 y antes
  de presentar la observación a la política. No recibe estado
  privilegiado, pero el observador usa un entorno contrafactual para
  localizar la región que se debe ocultar.

Las variantes raw y zero también se evalúan con mask_cue para controlar
los efectos de la ablación visual. S7 y S13 se evalúan sin entrenamiento
en S13, conservando las siete acciones y recompensa nativas.

## Igualdad y límites experimentales

- Misma fuente de demostraciones F9-D (ruta BFS con estado privilegiado
  SOLO para obtener acciones supervisadas), mundos y variantes pareadas,
  presupuesto de aprendizaje imitativo y transiciones PPO/RecurrentPPO.
- Modelo zero y F4 tienen **igual** anchura de entrada y parámetros;
  raw tiene menos entradas y se declara por separado.
- El modelo puede usar la memoria interna LSTM además de F4. Una
  ventaja de F4 sobre zero indicaría valor adicional, no que F4
  fuera la única memoria utilizada.
- F9-D detectó 0/40 elecciones dependientes de pista. Un 50% de
  acierto puede conseguirse eligiendo una salida fija. La señal más
  discriminante es acertar los DOS miembros del par contrafactual
  y cambiar de lado cuando la pista se invierte.
- F4 FIFO guarda *símbolos* en orden; no clasifica relevancia aprendida,
  por lo que es posible que otros objetos ocupen los slots antes de la
  pista. No manipular la política ni incorporar etiquetas de pista para
  forzar una mejora.

## Evaluación y gates

Primero una batería de integridad con entrenamiento técnico diminuto.
Esta prueba confirma API y ausencia de filtraciones, NO capacidad
cognitiva. Si pasa, ejecutar estudio de desarrollo con índices de
entrenamiento 0..3, presupuesto máximo F9-D: 600 mundos/semilla,
40 épocas BC, 20k transiciones PPO por modelo/semilla, 10 pares de
evaluación por tamaño y semilla. Registrar **todos** los resultados,
también negativos.

Reportar S7/S13: éxito original, éxito pista invertida, éxito combinado,
pares correctos en ambas variantes, cambios de salida, pista visible,
memoria escrita y tiempo agotado; registrar por modelo y semilla los
pesos pre-RL y pos-RL, SHA y parámetros. No considerar 40 pares como
40 réplicas independientes del entrenamiento.

Se priorizará una repetición con 100 pares de desarrollo por semilla si
F4 muestra una posible mejora y las ablaciones son congruentes. Después
sería necesario un estudio confirmatorio con las semillas de entrenamiento
y capacidad de navegación congeladas. El holdout preasignado nunca se
abre en esta fase. No se cambia PC ni VPS: CI GitHub-hosted.

**Criterio de evidencia exploratoria prometedora:** mejora consistente
F4 vs zero por semilla, aumento de pares correctos en ambas variantes y
de cambios de salida, con descenso frente al propio checkpoint al
eliminar memoria y ante la corrupción pertinente. Una única diferencia
de media sin cambios de decisión no autoriza a afirmar superioridad.
