# INFINITO F9-L — Resultados: dependencia temporal de F4 para la memoria recurrente

**Fecha:** 29 de octubre de 2026. **Estado:** F9-L ejecutado; GitHub Actions [#40948523471](https://github.com/webtilians/principiodelTodo/actions/runs/40948523471) **SUCCESS** en integrity, cuatro semillas y agregado. [JSON completo versionado](./INFINITO_F9L_RESULTS_2026-10-29.json); [artifact original #12930850960](https://github.com/webtilians/principiodelTodo/actions/runs/40948523471/artifacts/12930850960). [PR #19](https://github.com/webtilians/principiodelTodo/pull/19), borrador sobre #18.

## Pregunta y procedimiento

F9-K encontró identidad key/ball linealmente decodificable de H (hidden+cell LSTM) en S13/fork en **96,25 %**, con la observación actual y readout F4 idénticos entre cues invertidos. Pero F4 seguía siendo informativo durante buena parte de la trayectoria. **F9-L pregunta si H conserva la información cuando se suprime ese suministro externo hasta llegar al cruce.**

Se reutilizaron sin reentrenar los cuatro checkpoints F9-I context8 y los cuatro zero originales, validados contra los SHA-256 de la ejecución fuente #40928547873. Cada modelo mantiene los siete movimientos nativos, observación de 148 componentes, LSTM y 512 canales externos originales. El **selector F4 sigue funcionando** y recibiendo frames públicos; la intervención pone a cero *solo los canales F4 entregados a la política* desde el momento de corte hasta el frame anterior al fork. En el **mismo fork**, F4 original se restaura, de modo que al decidir el input actual O+M es el mismo que en control natural y entre las dos historias de cues invertidos.

Las dos historias avanzan mediante el prefijo **idéntico** de acciones nativas del profesor hasta la primera siguiente acción distinta. Son recorridos forzados diagnósticos, **no éxitos autónomos**. Ninguna etiqueta oracle ni acción futura entra a la política o al selector.

Se usaron 80 **pares** S7 train/semilla exclusivamente para entrenar dos probes ridge lineales (alpha=10), uno ajustado sobre H natural y **congelado para todas las intervenciones**, y otro **readaptado a cada intervención** con entrenamiento en S7 train. Luego se evaluó en 10 pares S7 development y 10 pares S13 development por semilla (40 pares, 80 decisiones por tamaño). **Ni S13 ni development entrenaron un probe**. Se mantuvo cerrado heldout. Sin PC/VPS, ni entrenamiento del agente.

## Resultado S13: supervivencia sin realimentación F4 hasta el cruce

Todas las métricas son agregados de **cuatro semillas de política**, 80 casos por condición, 40 pares. Distancia del corte = pasos sin readout F4 antes de restaurarlo en fork.

| Condición | Pasos sin F4 | Probe natural fijo, acc. | Probe readaptado, acc. | Probe readaptado, pares ambos correctos | Primera acción correcta | Pares ambas acciones correctas |
|---|---:|---:|---:|---:|---:|---:|
| Natural, sin cortar | 0 | **96,25 %** | **96,25 %** | 37/40 | **88,75 %** | 31/40 |
| Cortar solo paso anterior al cruce | 1 | 61,25 % | **66,25 %** | 13/40 | **77,5 %** | 22/40 |
| Cortar desde medio recorrido | 5 | 50 % | **52,5 %** | 2/40 | 53,75 % | 9/40 |
| Cortar desde desaparecer la pista | 10 | 50 % | **50 %** | 0/40 | 51,25 % | 1/40 |
| Control neutral, tocar padding F4 que ya era cero | 10 (ninguno efectivo) | **96,25 %** | **96,25 %** | 37/40 | **88,75 %** | 31/40 |
| Política zero original, no recibe F4 | 0 | 50 % | 50 % | 0/40 | 50 % | 0/40 |

**Variabilidad entre las cuatro semillas, probe readaptado S13:**
- Natural: **100 %, 100 %, 95 %, 90 %**.
- Interrupción de un paso: **50 %, 50 %, 95 %, 70 %** (gran dispersión; no universalizar 66,25 %).
- Interrupción de cinco pasos: **55 %, 50 %, 50 %, 55 %**.
- Interrupción de diez pasos: **50 %, 50 %, 50 %, 50 %**.

**Lectura:** en S13, al bloquear F4 desde justo después de la pista durante diez pasos, ni el probe entrenado sobre estados interrumpidos recuperó la identidad por encima del 50 % en estos mundos, y la acción del agente también quedó cerca del azar. Con cinco pasos sin F4 ocurre algo similar. Interrumpir solo un paso dañó menos la acción, pero el probe readaptado ya mostró caída y fuerte variación entre semillas.

No es un tiempo de vida o vida media de memoria medido en general; son tres ventanas discretas *bajo prefijo guiado* en una arquitectura entrenada con F4. Tampoco es prueba de borrado absoluto: podría quedar información no lineal, o cambiar el código interno de H bajo la intervención OOD.

## Resultado S7: el contraste decisivo representación/uso

| Condición | Pasos sin F4 | Probe natural fijo, acc. | Probe readaptado, acc. | Primera acción correcta | Pares ambas acciones correctas |
|---|---:|---:|---:|---:|---:|
| Natural | 0 | 100 % | 100 % | 96,25 % | 37/40 |
| Corte temprano | 4 | **50 %** | **100 %** | **55 %** | 4/40 |
| Corte a mitad | 2 | **50 %** | **100 %** | **62,5 %** | 14/40 |
| Corte solo último paso | 1 | 60 % | **100 %** | 95 % | 36/40 |
| Control neutral (padding ya cero) | 4, sin cambio efectivo | 100 % | 100 % | 96,25 % | 37/40 |
| Zero (sin F4) | 0 | 87,5 % | 87,5 % | 50 % | 0/40 |

En S7, el probe natural fijo cae al 50 % al bloquear F4 durante 2–4 pasos, **pero el probe readaptado consigue 100 % de decodificación en las mismas condiciones**. Esto es evidencia explícita contra equiparar la caída de un probe fijo con pérdida de información: un cambio de representación debido a la intervención puede invalidar la lectura anterior. **El modelo sigue albergando información linealmente recuperable de la pista, pero no elige ambas acciones correctas de forma fiable** (4/40 pares con corte temprano).

En S13 los probes readaptados también fallan con cortes de 5–10 pasos, pero no podemos distinguir si es desaparición total de la información, pérdida de decodificabilidad lineal o desajuste en la transferencia S7→S13 de ese probe.

## Integridad confirmada

1. **Paridad exacta con F9-K:** reproducidas por semilla las accuracies naturales H/fork S7 y S13, tanto context8 como zero; las divergencias causaban error y fallaban el job.
2. En cada par, observación actual O y F4 externa M **idénticas entre key y ball en el fork**. En cada condición se restaura M sin modificar selector/almacenamiento, y se cotejan O y M contra el control natural en los tests.
3. Control \`neutral_tail\` en todo el train y desarrollo: H y próxima acción **exactamente iguales** al natural, al no tocar ninguno de los 64 canales activos. Esto verifica la instrumentación, pero NO equivale a un control OOD no trivial.
4. Se exige cobertura efectiva del blackout y longitud de ventanas, sin selección por acierto. El formato del modelo y sus pesos permanecen idénticos. Se reportan todas las semillas y condiciones; el holdout no se abrió.

## Conclusión delimitada y próximo experimento

La interpretación más compatible con lo observado es que **F4 desempeña un papel continuo durante el recorrido largo S13** para sostener o actualizar estados recurrentes que desembocan en decisiones dependientes de la pista. Quitar la realimentación F4 durante cinco o diez pasos reduce la decodificabilidad lineal transferida a S13 y degrada la primera decisión al azar.

A la vez, **F9-L demuestra que información decodificable y comportamiento correcto pueden separarse**: en S7/H el probe readaptado identifica perfectamente la pista aun cuando el actor no sabe convertirlo en una decisión adecuada tras la perturbación.

**No afirmar:** que la información sea totalmente borrada, que el agente sea consciente, que el resultado esté confirmado en otros entornos o que el cambio sea una medición de memoria sin OOD. Los probes son lineales y entrenados solo en S7, el actor se sometió a entradas artificiales, la geometría y las semillas son de development reutilizadas y los agentes fueron entrenados por separado.

**Siguiente dirección (posterior a F9-L, sin ejecutarla aquí):** comparar entrenamiento con *dropout temporal F4* (políticas nuevas, no los checkpoints originales) contra un control de entrenamiento igualado, para saber si el agente **aprende** robustez a interrupciones y pasa a usar representaciones internas de forma estable. Requeriría nuevo protocolo, presupuesto, controles y validación fuera de development; no interpretar retrospectivamente F9-L como si ese reentrenamiento ya hubiera ocurrido.
