# INFINITO — Revisión de rumbo y resultados F7 / F7b

**Fecha:** 23 de octubre de 2026  
**Rama:** experiment/f7-active-information-value  
**Código F7:** observatory/active_information_f7.py  
**Código F7b:** observatory/active_recalibration_f7b.py  
**Protocolo F7 anterior a resultados:** docs/INFINITO_F7_ACTIVE_INFORMATION_PROTOCOL.md  
**Protocolo exploratorio F7b registrado tras F7:** docs/INFINITO_F7B_RECALIBRATION_PROTOCOL.md  

**Ejecuciones de CI:** F7 https://github.com/webtilians/principiodelTodo/actions/runs/39868941669; F7b https://github.com/webtilians/principiodelTodo/actions/runs/39869421088. Ocho y cuatro tests unitarios, respectivamente, superados; JSON completos disponibles como artefactos de Actions.

## 1. Auditoría F1–F7: no mezclar problemas

**F1–F3 (V5.1).** El borrado de memoria altera ligeramente la integración, pero más de lo que mostraban las medidas parciales de módulos. La congelación de escrituras permitió decodificar la pista diferida en 81,25% frente a 50% del control en un modelo V5.1 pequeño y sin entrenar. **Se entrenó un decoder lineal externo**: no es generación autónoma de respuestas, ni memoria autobiográfica. El congelamiento sabía cuándo la pista importante llegaba.

**F4.** La estrategia de demanda aprendida logra 87,99% de hits; el óptimo analítico del diseño es 88,33%, porque dos de ocho claves concentran 86% de consultas y caben tres en memoria. Es aprendizaje de frecuencias, no una teoría nueva de inteligencia.

**F5.** Potencial de EcoDispatch para memoria predictiva: existe protocolo propuesto, **no resultado de integración neuronal**. No adjudicarle mejora cuantitativa o validación que no se ha ejecutado.

**F6.** El agente aprende transiciones y planifica, pero un control reactivo diseñado por una persona fue mejor y la prueba sufrió desajuste de parámetros físicos entre los entornos de entrenamiento y evaluación. En concreto, el simulador determina su física con seed//1000; los bloques físicos del test no son los mismos que los del entrenamiento, aunque comparten rangos. Su "in_distribution" es una familia común de parámetros, **no identidad de mecanismos**. El baseline Q y MPC no fueron optimizados con equivalente cuidado. Antes de estudiar transferencia en F6, emparejar mundo físico y corregir la comparación; no reivindicar superioridad general de planificación.

**F7.** En una tarea de información costosa, señales parcialmente observables y dos sensores uno de ellos inútil, el controlador Bayesiano aprendió fiabilidades y decidió si pagar por un sensor **antes** de observar la lectura. Esto prueba una implementación funcional de *valor de la información*, no conducta no programada. La familia probabilística y el coste están codificados por el investigador.

**F7b.** Se añadió DESPUÉS de observar el fallo de F7: detector de error estadístico sostenido más campaña explícita de 40 observaciones alternas. Su recuperación tras un cambio es una demostración exploratoria de identificación y control adaptativo. **No** es prueba preregistrada independiente ni curiosidad espontánea.

## 2. F7 — 24 semillas y 600 episodios por semilla/régimen

Recompensa neta media por episodio; más alto es mejor. La exploración cuesta 0,12; el pago por acertar/errar la decisión final es proporcional a las consecuencias observables (0,2 o 1).

| Política | Mundo estable | Inversión de sensor (total) | Ambos sensores ruido |
|---|---:|---:|---:|
| Valor de información adaptativo | **0,3633** | 0,2519 | **0,2484** |
| VOI congelado tras calibración | 0,3667 | 0,2050 | 0,2470 |
| Nunca inspeccionar | 0,2447 | 0,2447 | 0,2470 |
| Inspección aleatoria | 0,2442 | 0,1876 | 0,1859 |
| Inspeccionar siempre | 0,3422 | 0,1108 | 0,1211 |
| Inspeccionar fuente impredecible | 0,1229 | 0,1209 | 0,1286 |
| VOI con fiabilidades verdaderas (oráculo) | 0,3684 | 0,3734 | 0,2491 |

En estable, diferencia emparejada VOI adaptativo menos sin inspección **+0,11864** recompensa/episodio; IC exploratorio bootstrap por semilla 95% [0,10891; 0,12839]. Menos inspeccionar siempre **+0,02111** [0,01618; 0,02638]. Fracción de inspecciones 45,56%; fracción en sensor inútil 0%.

Control sin información: el adaptativo realizó 0% inspecciones, sin ventaja atribuible a sondar ruido. Su ligera diferencia frente a «nunca» se debe a **calibración online del indicador gratuito**, no a obtener información de sensores (F7 permitió que el adaptativo aprendiera ese indicador en test, a diferencia de otros baselines). Esta asimetría debe corregirse en futuras comparaciones confirmatorias.

Tras invertir la fuente informativa, en las últimas 100 decisiones la política adaptativa obtuvo **0,1576** frente a **0,2248** de actuar sin inspección, y **0,3643** del oráculo. F7 no sabía encontrar a tiempo el nuevo sensor, aunque perdió parte de su confianza en el anterior. En toda la segunda mitad, VOI adaptativo menos no inspeccionar = **−0,10847** [−0,13026; −0,08417]. Hipótesis principal estable cumplida, pero **robustez a cambio fallida**.

## 3. F7b — detectar sorpresa y abrir investigación (EXPLORATORIO)

Misma batería, mismos generadores de episodios, semillas y baselines, con un controlador adicional. Se rastrea error entre fiabilidad esperada y observada, se detectan discrepancias persistentes y se activan 40 observaciones alternas con coste. Esto **incorpora manualmente** cuándo una discrepancia obliga a reexaminar el modelo.

| Condición | Recompensa F7b total | Últimas 100 decisiones | Detecciones por semilla (media) |
|---|---:|---:|---:|
| Estable | 0,36333 | 0,35132 | 0,0833 |
| Inversión sorpresiva | **0,33596** | **0,34987** | 1,0 |
| Sensores sin información | 0,24839 | 0,2400 | 0 |

Tras el cambio, últimas 100 decisiones: F7b menos F7 original **+0,19228** recompensa [0,16518; 0,22192]; menos no inspeccionar **+0,12503** [0,09748; 0,15548], usando 24 semillas. Frente al oráculo aún queda **−0,01445** [−0,03102; −0,00030].

**Atención:** esta mejora se obtuvo tras diseñar F7b con conocimiento de los resultados F7. Aunque se usa la misma prueba, no es una confirmación independiente. F7b debe replicarse con un cambio de tipo distinto, costes distintos y nuevas semillas, sin volver a ajustar el detector.

## 4. Qué aporta EcoDispatch realmente

La utilidad de anticipar consecuencias y proteger flexibilidad futura es un puente válido entre la función de potencial de EcoDispatch y el valor de información en un problema parcialmente observable.

Pero el algoritmo F7 es un cálculo Bayesiano sencillo de utilidad esperada; NO implementa teoremas de OpenAI sobre k-median/k-server. Su comportamiento esperado tiene precedentes en control dual, POMDP y exploración dirigida por progreso del aprendizaje. No afirmar novedad matemática.

EcoDispatch además mostró que proteger una capacidad futura puede ser perjudicial si el contexto real no la necesita: en su replay histórico λ positivo empeoró el coste modelado. En F7 ocurre una situación análoga: inspeccionar ruido o comprar información que no mejora una decisión derrocha recursos.

## 5. Veredicto y criterio de continuidad

**La ruta sí tiene sentido para investigar mecanismos de autonomía funcional**: observación -> modelo causal/predictivo -> detección de discrepancias -> búsqueda dirigida de información -> decisión. Los experimentos revelan efectos contrastables y fallos.

**No tiene aún sentido construir una nueva «capa de conciencia» en V5.1.** Ningún experimento ha producido un comportamiento imprevisto que resista controles y justifique una afirmación de experiencia subjetiva.

Recomendación de próximos hitos, con una condición de parada:

1. **Reparar F6 y emparejar física** en un protocolo de test nuevo, sin sesgos de versión; comparar MPC con política simple y Bayes adaptativo en mundo compartido.
2. **Verificar F7b fuera de muestra**: mecanismo desconocido, nuevas condiciones, cambios más graduales y sensores ruidosos sin señal. Repetir con modelos calibrados por todos de igual modo y aprendizaje del indicador gratuito simétrico entre políticas.
3. **Unir ambas pruebas solo después**, en un mundo donde observar consecuencias (por experiencia) es costoso, las acciones alteran el entorno y la información se necesita a lo largo de múltiples pasos. Medir retorno total y error causal de pronóstico.
4. Si un controlador sencillo (inferencia Bayes, MPC, reglas reactivas) explica todo, documentar el resultado y **detener la búsqueda de supuesta emergencia** en este diseño. Investigar estructuras recurrentes causales solo si aparecen fenómenos nuevos no explicados por baselines.

**No confundir:** aprendizaje de fiabilidad, motivos instrumentales, inteligencia general ni conciencia son propiedades diferentes. El laboratorio tiene valor científico si mantiene esa distinción y publica resultados negativos.
