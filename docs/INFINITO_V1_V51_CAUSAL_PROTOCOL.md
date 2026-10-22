# Comparación sistémica V1 ↔ V5.1 — Protocolo congelado

**Objeto:** estudiar continuidad, estructura y respuesta causal de estados internos,
sin usar "porcentaje de conciencia" como objetivo o criterio de éxito.

**Implementaciones originales:** V1, primer release de 16-sep-2025,
hash \`14860c32b32b470cc478dd1806059b9fd2c46ce2\`; V5.1,
\`src/infinito_v5_1_consciousness.py\` de master.

**Instrumentación:** \`scripts/infinito_full_compare.py\` sobre la rama
\`experiment/emergence-observatory-v0-1\`. Los algoritmos originales
no se editan. Se observan las salidas de módulos mediante hooks y se
aplican intervenciones externas.

## Límites y honestidad experimental

- V1 ejecuta su **bucle original** \`run_infinite\` (tope 200 recursiones);
  sobre una cuadrícula 16×16, configuración predeterminada de su clase inicial.
- V5.1 ejecuta su **bucle original** \`run_experiment\`, incluyendo
  \`train_step\` completo: cuatro módulos LSTM especializados, memoria
  externa, atención global, estimador Φ histórico y pérdidas combinadas.
- Para hacerlo viable en CPU, V5.1 utiliza **24 iteraciones** (frente a
  las 15.000 originales previstas), \`hidden_dim=64\` (frente a 512),
  memoria de 16 slots (frente a 256), batch 4, 8 cabezas, entrada 257;
  conserva los tipos de componentes y la ruta completa de cálculo.
  **No es reproducción completa de la escala ni del entrenamiento histórico.**
- Se usan las semillas 11 y 29. La clase inicial V1 reasigna algunas
  semillas internamente; las ejecuciones no abarcan diversidad de
  condiciones iniciales espaciales independientes.
- En V5.1 se debe inicializar la compatibilidad del planificador de LR
  con un grupo de parámetros añadido *después* de crear el scheduler;
  también inicializamos un campo de auditoría. Estas correcciones residen
  exclusivamente en el arnés y se registran como limitación.
- La terminación prematura de V1 por su condición propia se acepta
  y se documenta; no se fuerza ninguna trayectoria.

## Resultados medidos

Se toman snapshots numéricos de estados reales:

- **V1:** matriz del campo, tamaño y diversidad espacial, entropía
  de Shannon correcta, fracción activa, gradiente entre vecinos,
  norma de leyes, cambios de estado y similitud con el paso anterior.
- **V5.1:** vectores medios de los cuatro módulos completos, separación
  entre módulos mediante coseno, norma y cambio temporal del estado,
  memoria externa, normas de estados LSTM, pérdida solo como diagnóstico.

**Ninguna puntuación de V1 equivale directamente a una de V5.1.**
La comparación válida es la de cada arquitectura frente a su **propio**
control emparejado.

## Intervenciones causales y controles

Todas las intervenciones se aplican en la mitad de la ejecución,
comparando dos trayectorias inicializadas con la misma semilla.

V1: perturbar cuatro kernels que representan leyes del campo en 0,25;
el control mantiene las leyes originales.

V5.1:
- **hidden_reset**: poner a cero estados recurrentes h/c de los
  cuatro módulos desde la frontera de la intervención;
- **memory_reset**: vaciar memoria externa, edades y restablecer
  la fuerza inicial;
- **visual_causal_cut**: interrumpir el mensaje de salida causal
  del módulo visual que alimenta los siguientes módulos.
El control mantiene todos los componentes activos.

Se mide divergencia L2 entre vectores de estado del control y tratamiento,
normalizada por la mediana de norma del control antes del tratamiento.
También se calcula máxima divergencia antes y después, primera y última
divergencia, y una **regla exploratoria** de recuperación: divergencia final
inferior al 20% de la primera divergencia posterior a la intervención.

**Esto no prueba conciencia ni causalidad biológica.**
Solo verifica dependencias funcionales de variables manipuladas con
control determinista. Tampoco implica que un colapso, una oscilación o
una divergencia sea una propiedad emergente.

## Ejecutar

~~~bash
PYTHONPATH=. python scripts/infinito_full_compare.py \
  --v1-steps 200 --v51-steps 24 --hidden-dim 64
~~~

Genera \`observatory_results/full_comparison/comparison.json\`,
\`observations.jsonl\`, vectores \`states.npy\` por ejecución
y logs nativos. Los resultados se alojan como artefactos de Actions,
sin sobrescribir los experimentos anteriores.

**Siguiente fase** (no pertenece a este protocolo): entrenar V5.1 con
tamaño original y muchas más iteraciones, comparar ablaciones por módulo
en nuevas semillas, introducir estímulos estructurados y medidas
de predicción causal en datos no vistos.
