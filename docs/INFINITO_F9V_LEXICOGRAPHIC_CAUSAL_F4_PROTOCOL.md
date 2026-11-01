# INFINITO F9-V — Valor causal de leer F4 y optimización lexicográfica inspirada en EcoDispatch

**Preregistro 2026-11-01 — se publica ANTES del código, los tests y cualquier TRAIN/DEV.** Rama F9-V encima de F9-U; núcleo del actor F9-P y F4 congelados. No LLM.

## Conexión matemática verificable

EcoDispatch [docs/math_spec.md](https://github.com/webtilians/ecodispatch/blob/master/docs/math_spec.md) emplea (1) prioridad lexicográfica **máxima gravedad cubierta → máximo número cubierto → mínimo coste secundario** y (2) una política con término de exposición futura `coste inmediato + λ·Δpotencial de cobertura`. No existe garantía de transferencia de algoritmos k-median, matching ni k-server a navegación con memoria; solamente exploramos **el principio de ordenar objetivos y comparar consecuencias**. En EcoDispatch histórico 1.7.1, λ>0 no superó λ=0, por lo que no presuponemos que el potencial mejore los resultados.

En F9-T/U el único retorno de aprendizaje se basaba en recompensas globales + READ-cost, y F9-U compartía un retorno entre dos pistas distintas, sin aislar la contribución de READ frente a SKIP desde un mismo estado.

**Hipótesis primaria F9-V:** un predictor neuronal del valor causal de **una lectura de F4 en el primer cruce** puede (a) anticipar qué decisiones de lectura rescatan una primera elección, (b) mantener los aciertos del control READ en situaciones nuevas y (c) reducir consultas respecto a READ siempre, priorizando acierto sobre ahorro.

## Intervención y alcance (decisión aislada en cruce, no end-to-end)

- Fuente por réplica `index=0..3`: checkpoint `ppo_sham_after_rl.zip` original F9-P, SHA e informes originales validados con `validate_sources()`. Misma red LSTM H/C 64+64, mismo vector de 148 observación pública + 512 F4.
- Utilizar código F9-Q `matched_pair`, `history_pair` y `replay_to_decision` para generar dos historias con pista key/ball invertida y prefijo espacial **idéntico del profesor hasta el primer cruce**. El profesor SOLO mueve el entorno durante el prefijo, no suministra acciones a la política. La continuación desde el cruce es autónoma hasta la primera elección de brazo o fin.
- Dos regímenes: `history_f4_normal` y `history_f4_postcue_off` (F4 visible normalmente vs ausente para el actor durante los últimos 10 pasos post-cue). El escritor F4 permanece activo en ambos.
- En el cruce, el actor recibe **solo**: READ = los 512 canales F4 del snapshot; SKIP = 512 canales cero. Observar el 148 público de ambos contrafactuales es obligatorio idéntico. H/C, entorno y trayectoria previa del actor son exactamente iguales antes de la decisión; una única intervención difiere en la lectura. **TODAS las observaciones posteriores tienen F4=0 en ambos brazos**, para aislar la decisión y no cambiar una política de consultas futura.
- La diferencia causal observada por el evaluador es `D = first_arm_correct(READ) − first_arm_correct(SKIP)` en `{-1,0,+1}`, sin omitir no-compromisos. El estimador del futuro es una trayectoria nativa y determinista por intervención hasta primer compromiso, no una predicción perfecta de recompensa total ni una garantía de que el actor llegue autónomamente al cruce.
- El coste físico de leer una vez es `1`, de SKIP `0`. Orden lexicográfico EcoDispatch adaptado para **el mismo cruce**: `(first_arm_correct, -read_cost)`. Si ambos aciertan o ambos fallan, escoger SKIP; si READ rescata, escoger READ; si READ perjudica, escoger SKIP. Esta regla de selección es **ORÁCULO DIAGNÓSTICO generado por intervenciones**, no agente desplegable.

## Capacidad aprendida vs evaluación privilegiada

- El comparador/oráculo de desarrollo ve las dos continuaciones y el `correct_side` del observador exclusivamente **para generar etiquetas durante entrenamiento/medición**.
- La cabeza neuronal `ReadHead` existente MLP 276→32→1 ve **SOLO** `concat(raw_public_148, H_previous_64, C_previous_64)` antes de la lectura; no F4, no futuro, no etiqueta correcta, no coordenadas privilegiadas, no bandera del régimen, no `D` en inferencia.
- Entrenar dos predictores de la misma arquitectura, presupuesto y optimización, uno `hc` (H/C) y otro `public` (H/C anulados solo en la entrada de cabeza). Supervisión: elegir READ únicamente cuando READ produce un primer brazo correcto y SKIP no. Etiquetas con `D=0` prefieren SKIP; etiquetado por lexicografía, no por suma de recompensas.
- **Control aprendible de prioridad escalar**: tercera cabeza `scalar_hc` con misma entrada y arquitectura que `lex_hc` pero etiquetas que maximizan `0.15·success − 0.1·read_cost`; representa coste escalar exagerado frente a fiabilidad (éxito vale 0.15). Así difiere en casos en que READ salva una primera elección (0.15−0.1>0); esta parametrización por defecto **NO difiere de lex** y por tanto **no se usará como contraste principal**. Para ensayo de sensibilidad realmente contrastivo, fijar `scalar_hc` a `success−1.1·read_cost` (prioriza ahorro aunque READ salve), deliberadamente fuerte y desfavorable. Este brazo sirve de prueba de prioridad, **no como baseline competitivo ni evidencia general de superioridad**. El baseline principal es `lex_public`, `always`, `never`, y el oráculo.
- La cabeza entrenada es una **imitación del etiquetado por intervención**, no generación autónoma de nuevas políticas ni RL; cada predicción de necesidad se debe validar prospectivamente.

## Presupuesto y particiones estrictas

TRAIN `8_100_000 + 10_000*index + j`, j=0..31: 32 mundos×2 cues×2 regímenes = 128 estados intervenidos por réplica; 256 continuaciones READ/SKIP/semilla.

DEV **independiente** `8_200_000 + 10_000*index + j`, j=0..15: 16 mundos×2 cues×2 regímenes=64 estados/semilla. Solo se abre tras entrenar y fijar pesos. Se generan ambos futuros también en DEV **solo para auditar el oráculo**, nunca para seleccionar READ/SKIP en la cabeza.

CONFIRM `8_300_000 + 10_000*index + j`, j=0..39, **NO accesible** desde código/CI actual y permanecerá sellado. No abrir F9-U 7_300_000+, F9-T 6_300_000+ ni F9 histórico 900_000+.

TRAIN y DEV solo con los cuatro pesos existentes, mismo preprocesamiento, sin ajuste de fuente checkpoint tras ver resultados. Entrenamiento de cabeza: `ReadHead`, Adam `lr=0.001`, **200 épocas**, `BCEWithLogitsLoss`, batch completo, `torch.manual_seed(91000+index)` para inicialización idéntica por cabeza. Sin selección por DEV ni ajuste de umbral; READ si `sigmoid(logit)>=0.5`. También registrar positivas D=+1, negativas D=−1, neutrales D=0 y cobertura por régimen/semilla. Si TRAIN solo contiene una clase, aún ejecutar, reportar degeneración. El modelo no recibe el régimen ni cronómetro privilegiado.

DEV métricas: primera acción espacial correcta para ambos cues en 16 pares por régimen (32 pares por réplica), número de READ, precisión contra el oráculo de estados, recall y precision para READ necesarias, matriz de errores por D, coste de entrenamiento/rollouts, cantidad de registros F4 vacíos, **par de primeras salidas correctas** y primer brazo correcto por episodio. Cada estrategia se evalúa con los **mismos** dos contrafactuales por estado; no hace falta un tercer rollout del agente puesto que en cada estado se puede elegir el resultado del brazo correspondiente solo para evaluación; al ejecutar la política real, la decisión se toma sin ver los dos resultados.

**Condición necesaria para progreso científico, pre-registrada**: debe haber ≥8 estados positivos D=+1 en TRAIN agregados y ≥4 en DEV, de lo contrario no hay señal de aprendizaje de READ. `lex_hc` debe lograr al menos 90% de los aciertos del brazo `always` en total y ahorrar ≥20% de lecturas, en al menos tres de cuatro réplicas no debe perder >2 aciertos individuales vs always, y conseguir al menos 3 puntos porcentuales más de elecciones correctas que `lex_public`. La exactitud y equilibrio por cue deben acompañar todo resultado. Si falla, documentar resultado negativo, no abrir 8_300_000+.

## Limitaciones y controles de falsación

- El protocolo usa prefijo guiado hasta el cruce; NO mide navegación autónoma hasta ese punto.
- `correct_side` lo usa el evaluador para construir la etiqueta del futuro, NUNCA lo recibe el actor ni la cabeza; el entrenamiento supervisado con etiquetas causales es una intervención de investigación, no metacognición autoorganizada.
- La lectura F4 de un solo instante puede no explicar efectos acumulados de múltiples lecturas. Si D=0 en casi todos los estados, resultado de no-identificabilidad, no prueba de que F4 sea inútil; una segunda fase requeriría comparar ventanas READ de varios pasos con nuevos datos.
- La "lexicografía" opera sobre acierto de primera elección, no traslada literalmente la severidad de incendios; la prioridad de éxito sobre coste está diseñada, no emerge del agente.
- No modificar `src/` ni reentrenar políticas F9-P/F4. Publicar cuatro réplicas, incluso negativas; nunca presentar éxito CI como éxito científico.
