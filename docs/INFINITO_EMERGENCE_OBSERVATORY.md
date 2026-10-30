# INFINITO — Observatorio de emergencia v0.1

**Rama aislada:** experiment/emergence-observatory-v0-1, creada desde master.

## Objetivo

Observar estados escalares de simulaciones de diferentes generaciones de INFINITO,
encontrar cambios temporales que puedan merecer una explicación, etiquetarlos
provisionalmente y preparar experimentos causales posteriores.

**No se intenta inferir conciencia, sentimientos, intención ni vida a partir de
estas métricas.** Los resultados son candidatos a fenómenos funcionales.

## Lo que implementa esta primera versión

1. Esquema de observación independiente del simulador: JSONL con run_id, step
   y métricas numéricas finitas.
2. Importador de logs de recursión del INFINITO original y CSVs escalares.
3. Análisis ciego: sustitución de IDs y eliminación de metadata de condición.
4. Detector *exploratorio* de saltos de media entre ventanas adyacentes.
5. Comparación con permutaciones de orden temporal **dentro de cada ejecución**.
   Se prueba la máxima diferencia a lo largo de todos los posibles cortes, para
   que no baste seleccionar a posteriori el instante más conveniente.
6. Ajuste BH de valores p entre métricas medidas en una misma ejecución.
7. Agrupación descriptiva de candidatos repetidos en diferentes ejecuciones.
8. Datos sintéticos con transiciones conocidas y controles aleatorios.
9. Suite unitaria y CI exclusivos; no requieren instalar PyTorch, CUDA ni LLMs.

La salida se etiqueta «unclassified_transition», nunca «consciousness».

## Empezar sin modificar INFINITO

Requiere Python 3.10+; no hay dependencias externas para este observatorio.

~~~bash
python -m unittest discover -s tests -p 'test_emergence_observatory.py' -v

# Seis pruebas sintéticas: tres con un salto artificial y tres de ruido.
python -m observatory demo --output-dir observatory_results/demo

# Análisis sobre los datos cegados:
python -m observatory analyze \
  --input observatory_results/demo/blinded.jsonl \
  --output observatory_results/demo/audit.json \
  --window 6 --permutations 199 --alpha 0.05 --seed 42
~~~

Los archivos de observatory_results están ignorados por Git.

## Convertir experimentos históricos a observaciones

### Logs de la primera generación

Se reconocen líneas de consola como:

~~~text
Recursion 2: Clusters 4, Entropy 5.300, Loss 0.376
R   10: C 7 | E 1.23 | L 0.042 | ...
~~~

~~~bash
python -m observatory import-log \
  --input mi_ejecucion.log \
  --output observatory_results/v1_raw.jsonl --run-id v1_seed_0042
~~~

Las variables originales se guardan como clusters_reported, entropy_reported
y loss_reported: **no se corrige ni reinterpreta la definición histórica**.

### CSVs para cualquier versión

~~~text
step,loss,stability,innovation
0,2.1,0.1,0.03
1,1.9,0.2,0.06
~~~

~~~bash
python -m observatory import-csv \
  --input observaciones.csv \
  --output observatory_results/v51_raw.jsonl \
  --run-id v51_seed_0042
~~~

La importación acepta solamente columnas numéricas y una columna de paso
(step o recursion). Para comparar experimentos deben exportarse **las mismas
métricas con la misma definición y unidad**. Jamás se deben mezclar scores Φ
heurísticos de versiones distintas como si fueran una magnitud común.

### Cegar antes de examinar los resultados

~~~bash
python -m observatory blind \
  --input observatory_results/v1_raw.jsonl \
  --output observatory_results/v1_blind.jsonl \
  --key-file observatory_results/PRIVATE_v1_mapping.json
~~~

**Guardar PRIVATE_v1_mapping.json fuera del repositorio.** Se crea con
permisos de propietario y no se sobrescribe. No contiene conclusiones, sino
la clave y equivalencias de identificadores. La versión ciega expone solamente
ID seudónimo, paso y métricas numéricas. Revisar que los nombres de métricas
no revelen condiciones de estudio.

No hay un mecanismo de «apertura» automática de la clave: debe hacerse al
final del análisis y dejando constancia de qué criterios se congelaron antes.

## Qué significa una alerta estadística

El observatorio busca la mayor diferencia de medias antes/después entre
ventanas temporales de longitud fija. Obtiene un valor p empírico comparando
ese máximo con máximos de datos temporalmente permutados, bajo la hipótesis
nula de **intercambiabilidad temporal**. Utiliza el ajuste BH por ejecución
y además exige un tamaño de cambio normalizado.

Esta prueba **rompe la autocorrelación** en los controles. Puede marcar
incorrectamente procesos estacionarios con dependencia temporal.
Tampoco corrige la multiplicidad de todos los experimentos, semillas,
tamaños de ventana ni decisiones hechas durante una exploración.

Por eso, «flagged: true» significa únicamente: «merece seguimiento,
según este protocolo concreto». No es demostración de novedad real,
causalidad, adaptación, autorreferencia ni experiencia subjetiva.

## Protocolo científico para el siguiente paso

1. **Congelar definiciones y código** antes de abrir las condiciones.
2. Ejecutar varias semillas de cada configuración; registrar hash del código,
   configuración, versión de Python, hardware y todas las observaciones.
3. Cegar la versión y condición durante la primera detección.
4. Anotar candidatos neutrales: E-0001, E-0002, etc., sin interpretación.
5. Repetir con un conjunto independiente de semillas no vistas.
6. Comparar con controles adecuados para series autocorreladas
   (surrogates espectrales, block bootstrap, cadenas nulas calibradas).
7. Aplicar intervenciones controladas: congelar pesos, desactivar memoria,
   cortar feedback, sustituir entradas, cambiar escalas temporales.
8. Comprobar si el fenómeno predice estados futuros o altera desempeño
   frente a esos controles. Registrar también resultados negativos.
9. Solo después interpretar si corresponde a aprendizaje, adaptación,
   autoorganización, autorreferencia o un nuevo fenómeno.

**Causalidad** requiere intervenciones bien diseñadas y asignación/controles;
las permutaciones por sí solas no la establecen.

## Límites conocidos y próximos hitos

- Esta v0.1 detecta transiciones escalares, **no** morfología espacial del
  «mar binario», clusters móviles, grafos causales ni dinámicas multiescala.
- Aún no reproduce las simulaciones originales de V1 a V5.1; importa
  sus trazas cuando están disponibles. El demo es sintético.
- No hay integración a nivel de callback con los simuladores. Importamos
  registros para mantener la instrumentación fuera del objeto investigado.
- Las distribuciones nulas autocorreladas y los controles de ablación
  deben implementarse como una versión posterior, sin sobreajustar sobre los
  ejemplos inspeccionados.
- En el primer experimento histórico, revisar especialmente la entropía,
  el paso de estado entre recursiones y los indicadores llamados «phi».
- La siguiente capacidad distintiva sería registrar estados 2D, analizar
  persistencia/estructura espacial y efectuar intervenciones con pares
  controlados bajo semillas iguales.

La pregunta abierta es **qué mecanismos causan las nuevas propiedades
observables**, no «¿cuánto porcentaje de conciencia tenemos?».
