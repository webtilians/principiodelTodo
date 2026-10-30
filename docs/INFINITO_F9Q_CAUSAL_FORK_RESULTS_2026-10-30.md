# INFINITO F9-Q — Memoria recurrente: representación, decisión y causalidad

**Resultado confirmado el 30/10/2026.** [GitHub Actions #41043258450](https://github.com/webtilians/principiodelTodo/actions/runs/41043258450) **SUCCESS, 11/11 jobs**: integridad, entrenamiento de probes exclusivamente TRAIN, gate de los 12 checkpoints congelados y probes, cuatro confirmaciones independientes y agregado. [PR #24](https://github.com/webtilians/principiodelTodo/pull/24) en borrador, apilada sobre F9-P #23; no fusionada.

- [Protocolo preregistrado antes de confirmación](./INFINITO_F9Q_CAUSAL_FORK_RECALL_PROTOCOL.md).
- [Datos agregados JSON, 4 semillas, todos los brazos/controles](./INFINITO_F9Q_RESULTS_2026-10-30.json).
- [Artifact agregado original #12971321986](https://github.com/webtilians/principiodelTodo/actions/runs/41043258450/artifacts/12971321986); los **cuatro artifacts por semilla** contienen todos los casos originales/invertidos y cada intervención.
- Los mundos 800000+ **ya están consumidos** y no se utilizarán para elegir un nuevo modelo y volver a confirmarlo.

## Respuesta breve a la hipótesis de INFINITO

**Sí existe una contribución causal observable de H/C a la elección dependiente de la pista cuando F4 estuvo disponible en el historial pero se suprime en la bifurcación.** Bajo un prefijo de acciones físicamente idéntico y vistas públicas de decisión idénticas, intercambiar H/C entre la historia key y la ball hace que ambas decisiones cambien hacia la **pista donante** en **112/160 pares BC** y **112/160 pares PPO sham**, en cuatro de cuatro réplicas. Sin reemplazar memoria externa, la representación recurrente es accionable en este contexto controlado.

**No se conserva ese acceso funcional cuando F4 se bloquea durante los 10 pasos previos a la decisión**, pese a que un decodificador entrenado en mundos distintos sigue identificando correctamente key/ball desde C en **157/160 pares (BC/PPO sham)** y **160/160 pares (PPO blackout)**. Bajo ese intervalo sin F4, los tres checkpoints dan **0/160 pares de primeras salidas correctas**, y el trasplante H/C no cambia ninguna de ambas primeras salidas. **Decodabilidad ≠ utilización para actuar.**

Esto delimita el fallo: **la información sobre la pista puede permanecer accesible para un lector lineal externo dentro de H/C aunque la política no sea capaz de utilizarla en la elección**. No demostramos que ese sea el único factor ni que se conserve en memoria humana/semántica/general.

## Diseño y sellado

- Cuatro semillas y tres checkpoints **congelados** por semilla: F9-O \`continuous6_after_bc.zip\` (run #40957255987) y F9-P \`ppo_sham_after_rl.zip\`, \`ppo_blackout_after_rl.zip\` (run #40960267485). SHA-256 y presupuesto/comienzo PPO verificados antes de confirmar.
- **Ninguna política se reentrena en F9-Q.** Probes lineales H y C, ridge α=10, ajustados **solo con 80 pares nuevos TRAIN S13/semilla** \`750000+index*10000+j\`. Dos regímenes × tres checkpoints × dos componentes por semilla = 48 probes, validados antes de abrir confirmación. Las métricas de entrenamiento no seleccionan modelos.
- Confirmación S13 \`800000+index*10000+j\`, j=0..39, 4 semillas, **160 pares** de mismo mundo y cue invertida por grupo y régimen, distinta de TRAIN y confirmaciones previas F9-M/N/O/P y holdout histórico 900000+, **sin abrir**.
- A cada par se le presenta **el mismo prefijo de acciones del profesor** hasta la primera divergencia temporal de rutas key y ball. Esas acciones solo mueven el simulador: la LSTM actualiza su estado a través de \`model.predict\` con su propia observación, nunca a partir de la etiqueta docente.
- Los 148 valores observables son idénticos en el punto de decisión y los **512 valores del readout F4 están forzados a cero** en la decisión y durante toda la continuación. Las únicas intervenciones entre controles son intercambiar o anular H y/o C. La continuación desde el cruce es **autónoma**, mientras que la llegada al cruce es controlada, **no** éxito end-to-end.
- \`history_f4_normal\`: F4 disponible durante todo el prefijo, cero en fork/continuación.
- \`history_f4_postcue_off\`: F4 anulado durante **10 observaciones previas** a la bifurcación y también en ella/continuación, para ambos cues. El almacén/escritor F4 no se modifica; solo lectura al actor.
- Se midieron \`intact\`, \`swap_HC\`, \`swap_H\`, \`swap_C\`, \`zero_HC\`, \`zero_H\`, \`zero_C\`; no se omitieron casos difíciles, ni se filtraron por éxito.

## Regimen 1 — Historia F4 disponible; decisión y futuro F4=0

Cada celda representa **pares completos / 160**, no episodios individuales.

| Resultado S13 en cruce controlado | BC | PPO sham | PPO blackout |
|---|---:|---:|---:|
| **H descodifica ambas identidades** | 160/160 | 160/160 | 156/160 |
| **C descodifica ambas identidades** | 160/160 | 160/160 | 160/160 |
| Primera acción docente correcta en ambas variantes, H/C intacto | 123/160 | 116/160 | 40/160 |
| **Ambos primeros brazos espaciales correctos, H/C intacto** | **112/160** | **112/160** | 40/160 |
| **Trasplantar H/C: ambas salidas siguen identidad del donante** | **112/160** | **112/160** | 40/160 |
| Ambos primeros brazos cambian al intercambiar H/C | 112/160 | 112/160 | 40/160 |
| H/C a cero: ambos primeros brazos correctos | 0/160 | 0/160 | 0/160 |
| Trasplantar **solo H**: ambos primeros brazos siguen donante | 31/160 | 15/160 | 38/160 |
| Trasplantar **solo C**: ambos primeros brazos siguen donante | 0/160 | 0/160 | 0/160 |

### Consistencia entre semillas

| Política | Intacto: ambas salidas correctas en semillas 0–3 | Transferencia causal completa 0–3 |
|---|---|---|
| BC | [30, 30, 19, 33] | **[30, 30, 19, 33]** |
| PPO sham | [11, 40, 36, 25] | **[11, 40, 36, 25]** |
| PPO blackout | [0, 0, 2, 38] | [0, 0, 2, 38] |

**Interpretación mecanicista:** si la historia original y la invertida llegan a la misma observación, F4 está completamente ausente y solo se cambia H/C, el cambio sistemático de salida hacia **la identidad del donante** es evidencia causal específica de utilización de un estado recurrente que depende de la pista. Es más fuerte que medir distancia entre estados o exactitud de un probe; sin embargo, **el estado H/C transporta también otros aspectos del controlador y un trasplante puede crear combinaciones fuera de distribución**, así que no se debe describir como manipulación molecular pura de un «recuerdo».

PPO con blackout largo muestra esa capacidad en **una sola semilla dominante (38/40)**, no en tres/cuatro; el protocolo no autoriza afirmar robustez replicada en este grupo. BC y PPO sham sí presentan el patrón en **4/4**.

**H/C por separado:** se obtienen transferencias parciales al intercambiar H, pero ninguna transferencia completa al cambiar solo C en este conjunto. La memoria causal puede depender de la configuración conjunta y de cómo la LSTM interpreta H con C, no de una unidad/localización inequívoca. No se debe inferir que C sea inútil: como se ve abajo, puede decodificar la identidad incluso cuando el actor falla.

## Régimen 2 — F4 ausente durante las 10 observaciones finales del prefijo

| Resultado S13, prefijo sin lectura F4 final | BC | PPO sham | PPO blackout |
|---|---:|---:|---:|
| **H decodifica ambas identidades** | 151/160 | 151/160 | 156/160 |
| **C decodifica ambas identidades** | 157/160 | 157/160 | 160/160 |
| Ambos primeros brazos correctos, H/C intacto | **0/160** | **0/160** | **0/160** |
| Trasplante H/C, ambos siguen salida del donante | 0/160 | 0/160 | 0/160 |
| Ambos primeros brazos cambian al intercambiar H/C | 0/160 | 0/160 | 0/160 |
| Primeras acciones docentes correctas en ambas historias | 0/160 | 0/160 | 0/160 |

**Resultado clave:** la intervención prolongada no destruye por completo el contenido decodificable de la pista: el probe de C conserva 98–100 % de pares correctos. Sin embargo, la política no utiliza esa diferencia para producir las dos salidas correctas, y un intercambio de H/C en el cruce ya **no** induce inversión de la salida. Por tanto, *existencia de señal interna* y *capacidad funcional de acceder a ella* son variables distintas en estas políticas.

La conclusión no puede extrapolarse sin más a otros calendarios, otros tamaños de problema, otras arquitecturas o a la «memoria» cognitiva en sentido general.

## Qué significa frente a F9-L/M/N/O/P

- **F9-K/L:** decodabilidad del LSTM tras distintos prefijos/interrupciones; por sí sola no probaba uso.
- **F9-M/N:** entrenamiento con cortes F4 reduce bloqueos y mejora llegada, sin una elección cue-dependiente S13 claramente estable bajo pérdida prolongada.
- **F9-O/P:** un agente puede llegar a todos los cruces y aun así elegir un lado fijo; PPO excesivamente privado de F4 no conserva la habilidad.
- **F9-Q:** ahora se observa una **transferencia causal del estado H/C a la salida correcta bajo geometría y observación idénticas, con F4 retirado en la decisión** cuando el historial fue normal, pero la **decisión colapsa tras un prefijo sin F4** pese a la señal decodificable.

Esta cadena experimental sigue el propósito de INFINITO: distinguir **representación histórica**, **disponibilidad causal para una acción** y **robustez frente a memoria externa ausente**, sin confundirlas con éxito general de navegación.

## Conclusión, límites y próximo problema bien formulado

**Conclusión:** F9-Q aporta evidencia de que ciertas políticas LSTM han desarrollado una señal histórica que **puede determinar causalmente la elección** sin lectura F4 en el instante de decidir, y también demuestra un límite: la política **no consigue utilizarla** cuando se interrumpe F4 durante el intervalo previo de 10 pasos, a pesar de ser decodificable.

- No hemos observado una memoria consciente, introspectiva ni un recuerdo deliberado.
- La **llegada al fork está impuesta por teacher forcing**; no se atribuye a F9-Q navegación autónoma completa.
- La ablación H/C puede perturbar estados no relacionados con identidad; la transferencia hacia la pista donante y comparación de cuatro semillas limitan, pero no eliminan, esta ambigüedad.
- Los probes son **lectores entrenados externamente**: demostrar linealidad de señal en H/C no implica que la política tenga un lector equivalente.
- Descartar seleccionar por retorno o por una sola variante. **Los 800000+ nuevos mundos están consumidos para confirmación.**

**Hipótesis siguiente (no ejecutada aquí):** estudiar el mecanismo por el cual la acción deja de depender del código cue tras varios pasos sin F4. Una prueba adecuada podría manipular una dimensión/representación de H/C cerca de la bifurcación bajo trayectorias idénticas, o comparar controles de actualización recurrente durante apagado de F4, siempre con **otra partición confirmatoria nueva**; no volver a ajustar sobre 800000+ ni 900000+.

La PR permanece en borrador para revisión y no se han fusionado ramas.
