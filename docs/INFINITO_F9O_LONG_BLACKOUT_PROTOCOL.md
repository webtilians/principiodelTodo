# INFINITO F9-O — Protocolo preregistrado: blackout largo frente a misma dosis fragmentada

**Fecha: 29/10/2026, antes de entrenar y antes de abrir la nueva confirmación.** Continuación de F9-N (PR #21). F9-N confirmó que S13 con supresión sostenida de F4 mejoraba principalmente la **llegada al cruce** (28,75 % → 81,25 %), mientras la primera elección correcta condicionada al compromiso era cercana al azar (50 % frente a 51,54 %). No reutilizar sus 400000+ para elegir ni validar.

## Hipótesis primaria

Cuando dos políticas reciben **exactamente ocho observaciones de demostración con readout F4 a cero** por episodio, ¿enseñar una **ausencia continua de ocho pasos** conduce a mayor capacidad de escoger *ambas* salidas contrafactuales en S13 autónomo sin F4 posterior a la pista, frente a repartir los ocho pasos en cuatro ventanas cortas de dos? Priorizar **pares con ambas primeras salidas correctas**; el éxito aislado de una variante, el porcentaje de primeras elecciones correctas y el éxito final son secundarios.

## Dos brazos controlados y presupuesto

- `fragmented8` (control de **dosis**, no sham): bloquear F4 al actor en offsets **0,1,3,4,6,7,9,10** respecto al primer frame donde empieza la ventana.
- `continuous8` (tratamiento): bloquear F4 al actor en offsets **0,1,2,3,4,5,6,7** desde el **mismo** frame.
- Inicio fijo basado solo en observaciones públicas: primera aparición de cualquier key/ball detectada por el *mismo encoder* `SelectiveF4`; empezar exactamente dos frames después. El reloj no recibe etiquetas, posición, recompensas futuras, orientación a objetivo ni acciones del profesor. El escritor de F4 trabaja intacto; solo se modifican sus 512 canales **de lectura en el actor**, jamás la imagen parcial de 148 entradas ni el almacén.
- Cada **demostración de S13 train** con ambos cues presenta 8 frames efectivos de blackout en cada brazo. Las mismas observaciones públicas y acciones originales, sin selección o exclusión por éxito. **Fallar integridad si no se alcanza esa igualdad exacta**; no sustituir la igualdad por promedios. Verificar que efectivamente existían canales F4 no cero antes de anularlos.
- Cuatro índices de entrenamiento 0–3, dos políticas nuevas por índice (8 en total), **misma arquitectura/380552 parámetros/estado inicial SHA** y RNG de orden de mini-batches. Nuevos mundos **S13 TRAIN** `500000+index*10000+j`, `j=0..599`; dos cues por mundo, **1200 episodios de teacher por índice**, 40 épocas de BC sobre esos datos y **20.000 pasos solicitados de PPO** por política.
- **PPO es limpio e idéntico en horario:** no hay *ninguna* máscara en ninguno de los brazos PPO. Esto deja exactamente igualada la **dosis de blackout de entrenamiento** durante BC; los ejemplos de PPO visitados pueden divergir por las políticas y no se presentarán como transiciones idénticas. Esta elección favorece la interpretación causal sobre la **distribución temporal de ocho frames en BC**, no sobre la persistencia bajo blackout en PPO. Guardar resultados tanto **después de BC** como **después del PPO limpio** para detectar si el segundo deshace lo aprendido.
- PPO utiliza exclusivamente `500000+` TRAIN en S13, con recompensa nativa, siete acciones, sin maestro online ni reward shaping. Inicializaciones por índice iguales en los dos brazos. No hay ninguna acción/semilla de `development`, `300000+`, `400000+` ni `900000+` en entrenamiento.

## Comprobación de integridad ANTES de confirmar

- Probar máscaras, ocho frames realmente nulos en ambos brazos en **todos los 1200 teacher episodios**, distancias, igualdad de acciones y raw, pesos iniciales, recuentos por brazo y SHA-256 de ocho modelos. Al menos un episodio de entrenamiento de comprobación se usa para los tests, *nunca* el nuevo rango confirmatorio.
- `train` completo sobre cuatro índices; **gate** separado verifica el 100 % de los checkpoints y partidas presupuestarias antes de descargar cualquier confirmación. Si hay bug antes de confirmación, arreglar y volver a entrenar todo.
- Ninguna prueba, script o job del conjunto de confirmación se invoca en el entrenamiento.

## Confirmación prospectiva sellada

- Exclusivamente `600000 + index*10000+j` con `j=0..39` para S13 (**160 pares por brazo/condición**, 320 episodios por brazo/condición) y `j=0..19` para S7 (**80 pares**). **No** es F9-M 300000+, F9-N 400000+, TRAIN 500000+ ni holdout histórico 900000+, que permanece cerrado.
- Evaluación autónoma determinista con modelos congelados, sin profesor, mediante instrumentación de etapas F9-N: pista vista, llegada al cruce, compromiso espacial de primera salida, primer acierto, éxito, timeout y pares completos. Condiciones predefinidas:
  1. `cue_lost_continuous` (primaria): F4 suprimida en cada frame en que se vio antes la pista inicial y actualmente no es visible (igual que F9-N). **Ambas primeras salidas correctas por par** es el desenlace central, denominador total de pares sin filtrar no-llegadas.
  2. `normal`: coste en navegación íntegra.
  3. `fixed_4`: control de perturbación exógena de cuatro pasos t=3..6.
- Confirmar en dos etapas: primero los modelos, y después (por primera vez) estas semillas. No investigar resultados y reajustar umbrales, presupuestos, horarios o semillas con los mundos de confirmación consumidos.
- Agregación por **cuatro semillas** de política con deltas por índice. **Criterio descriptivo predeclarado:** aumento de pares S13/cue_lost con ambas primeras elecciones correctas en **al menos tres de cuatro índices** y aumento de pares global, no solo subida del éxito; informar por separado caso S7, llegada, first-choice conditional, full blackout coverage, casos sin decisión, y coste de navegación normal. Número total de pares no equivale a réplicas de entrenamiento.

## Riesgos y límites

- La intervención se inicia por **eventos públicos de key/ball**, no por el `cue_visible` privilegiado; en MiniGrid S13 el objeto cue puede no ser el primer key/ball percibido. El disparador es idéntico entre brazos y *no se calibrará sobre confirmación*. BC teacher-forced no reproduce distribuciones de trayectorias autónomas.
- Tras BC, el PPO limpio puede facilitar navegación pero degradar el código recurrente: reportar BC y PPO si se observan diferencias, sin elegir retrospectivamente el mejor modelo.
- El dropout CONTINUOUS8 no enseña necesariamente la desaparición permanente F4, solo ocho frames; fallo en el control `cue_lost_continuous` no significa que toda memoria interna sea imposible.
- La elección condicional a llegada sufre selección y no se interpretará como efecto causal entre las mismas historias. Las parejas con no-llegada siguen en denominador de desenlace primario.
- Cualquier resultado confirmado aquí se considera validación **consumida**; no reajustar y reutilizar 600000+ ni abrir el holdout 900000+.

**Interpretación:** si las políticas continuous8 resuelven con más frecuencia ambos cues S13 con F4 permanentemente fuera de visión, hay evidencia de retención/uso más robustos que con el **mismo número de exposiciones fragmentadas**. Si solo mejora la llegada, la hipótesis de memoria no queda confirmada, en línea con F9-N.
