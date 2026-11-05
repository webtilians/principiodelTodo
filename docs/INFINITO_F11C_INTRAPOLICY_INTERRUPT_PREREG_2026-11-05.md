# INFINITO F11-C — Interrupción aprendida de políticas dentro del episodio
**Prerregistro prospectivo, 2026-11-05. Este commit se escribe ANTES del código y los datos F11-C.**

## Pregunta
F11-B escogió A2/A3 al inicio del episodio: selector aprendido 60,66 %, una regla elemental de cambio tras fallo 60,95 %. ¿Puede el agente aprender **en tiempo de ejecución** a interrumpir una opción A3 cuando el estado observable contradice su trayectoria esperada y replanificar con A2, mejor que una guarda fija? No se busca reentrenar un LLM ni explotar información del simulador.

## Por qué otro estudio
No reciclar semillas, pruebas heldout, umbrales ni resultados de F11-A o F11-B. F11-C es independiente y solo aprovecha como controles congelados la familia sintética, el planificador A2 y las opciones A3 de F11-A v3. El A3 previo ya abortaba por **guarda exacta programada**: esta NO es una decisión aprendida; debe ser el baseline principal, y no atribuirnos esa capacidad previa como aprendizaje.

## Particiones F11-C, nuevas y aisladas
Cuatro réplicas r0..r3; semillas TRAIN exclusivamente **63_100_000+10_000*r+i**:
- índices 0..255: 256 mundos/r para aprender A2 y almacenar opciones A3 con 30 episodios de recolección aleatoria; misma evidencia para ambos;
- índices 256..383: 128 mundos/r de entrenamiento para detector/selector de interrupción, 30 episodios/mundo, experiencia solo de la trayectoria que realmente se ejecuta;
- índices 384..511: 128 mundos/r nuevos de **TRAIN-heldout**; pesos congelados antes de generar resultados; 2 condiciones de perturbación por mundo (sin cambio de reglas + cambio de fase oculto en un paso 3..7 del episodio).
Alumnos jamás reciben seed, fase, etiquetas de anomalía, true roles, cola de efectos ni plan óptimo. El auditor puede guardar seed, paso de cambio y etiquetas solo **fuera** del estado del actor. Restringir cada episodio a 12 acciones.

## Comparación y controles
- **P**: siempre A2 — planificar desde el inicio;
- **C**: siempre A3; continúa acciones de la opción si encuentra observaciones distintas, sin abortar (*control de falta de interrupción*);
- **G**: guarda exacta fija de A3 v2/v3 que aborta ante el primer estado diferente a la expectativa y cambia a A2 (*baseline fuerte no aprendido*);
- **H**: regla simple de sorpresa observable (desviación en variables de recursos/ruta, no cola) que aborta en la primera contradicción sustancial;
- **L**: detector logístico entrenado por REINFORCE a partir de recompensa/éxito del **episodio realmente ejecutado**, evalúa cada paso antes de la siguiente acción y puede abortar una única vez para ceder control a A2. Las características visibles son diferencia entre estado actual y guardia de trayectoria recordada, sorpresa en recompensa anterior, progreso, cola pendiente y persistencia de discrepancia. Nunca conoce fase/seed ni el resultado contrafactual.
- **L-no-sorpresa**: ablación del mismo detector sin indicadores de sorpresa; distingue aprendizaje de mera gestión del horizonte.
- **A4** oráculo solo como techo descriptivo, separado de la selección; no entrena modelos.

El controlador L no recibe más experiencia o presupuesto por episodio que los comparadores. Entrenamiento usa exploración estocástica en abortar/continuar, registra una sola trayectoria y el gradiente de log-probabilidades de decisiones tomadas. El control G exacto puede ganar legítimamente; NO lo debilitar de manera artificial. Si las opciones no se activan, registrar el porcentaje y la comparación como poco informativa.

## Estímulos y causalidad
El simulador de F11-A v2 aplica roles según la fase. Para F11-C la fase puede cambiar **solo desde la capa de mundo, después de iniciar la opción**, con instante variable (pasos 3..7) que NO se pasa al controlador. Episodios de control mantienen fase fija; ruido/slip del entorno es benigno. La misma semilla de ruido por mundo/escenario se usa para cada brazo. Control adicional: pruebas con cambios de fase que afectan efectivamente una acción posterior (no asumir que cada cambio altera decisiones si el episodio ya finalizó).

## Criterios para progresar (no PASS confirmatorio)
Se requieren al menos **3/4 réplicas** con:
1. L supera tanto a G como a P en **>=5 puntos porcentuales** de éxito en los episodios con cambio real, y no pierde más de 5 puntos en episodios sin cambio;
2. L supera a L-no-sorpresa por **>=5 puntos**, señal de que utiliza la discrepancia observable;
3. L no tiene más decisiones/episodio, y su coste total medido <=1,25x el del mejor control relevante. Informar modelos y opciones por separado;
4. registros trazables de anticipación, interruptores, falsos positivos, pasos postcambio y violaciones (cero), y variabilidad por réplica.

Si falta alguna condición, **NO-GO**, informe negativo sin buscar ajustes sobre heldout. No revelar o consumir DEV ni CONFIRM originales F11-A (61_200_000+, 61_300_000+, 61_400_000+) ni F11-B. Si el simulador hace trivial el problema (G=P=L o ninguna opción activada), etiquetar **INCONCLUSO** e investigar un nuevo mecanismo en otro experimento.

## Límites
Aun con ventaja: solo evidencia preliminar dentro de TRAIN, nuevos mundos de la misma familia sintética, no transferencia estructural, LLM, autónomo abierto ni seguridad de producción. Medir por mundo base, no fingir independencia de todos los pasos.

**Estado: preregistrado sin código de F11-C, sin resultados.**
