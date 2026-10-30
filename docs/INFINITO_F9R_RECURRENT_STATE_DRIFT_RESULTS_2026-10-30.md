# INFINITO F9-R — Informe de confirmación: fallo de congelación H/C y rescate de última hora

**30 de octubre de 2026 — resultado primario NEGATIVO y evidencia secundaria de sensibilidad temporal.**

- [GitHub Actions #41045483435](https://github.com/webtilians/principiodelTodo/actions/runs/41045483435): **SUCCESS, 7/7 jobs**. Todos los tests y el gate previo, 4 réplicas confirmatorias y agregado completados.
- [PR #25](https://github.com/webtilians/principiodelTodo/pull/25): borrador apilado sobre F9-Q #24; no fusionado.
- [Protocolo preregistrado](./INFINITO_F9R_RECURRENT_STATE_DRIFT_PROTOCOL.md), publicado **antes** de abrir la partición nueva.
- [JSON con todas las métricas, 4 semillas y 10 intervenciones](./INFINITO_F9R_RESULTS_2026-10-30.json).
- [Ejecución y artifacts de casos originales](https://github.com/webtilians/principiodelTodo/actions/runs/41045483435); cada una de las cuatro evaluaciones conserva 40 pares y todas sus primeras acciones, compromisos y trasplantes.

## Por qué F9-R

F9-Q había hallado que con F4 disponible durante el historial, **el estado LSTM H/C podía causar una decisión cue-dependiente** aun sin lectura F4 justo en el cruce: intercambiar el H/C de dos pistas invertidas hizo que la salida siguiera la identidad del donante en 112/160 pares BC y 112/160 pares PPO sham. Pero si se suprimía F4 durante los diez pasos previos, **la pista seguía siendo decodificable desde C por un probe entrenado fuera de esa confirmación** y, sin embargo, 0/160 pares elegían las dos salidas correctas.

F9-R pregunta **qué efecto tienen las actualizaciones recurrentes y su momento temporal** sobre esa pérdida del vínculo entre señal histórica y acción. No entrena nuevas políticas, probes ni PPO; las observaciones públicas, ruta física y geometría del prefijo son exactamente iguales dentro de cada mundo.

## Integridad y separación de conjuntos

- 12 checkpoints congelados de F9-O (BC) y F9-P (PPO sham y PPO blackout) sobre cuatro semillas, **verificados por SHA-256** y sujetos a un gate independiente antes de abrir la nueva confirmación.
- Fixtures de desarrollo **840000 + 1000*index + j**, j=0..9 y cuatro índices, total 40, rango 840000..843009.
- Nueva confirmación **850000 + 10000*index + j**, j=0..39, total **160 pares S13 por checkpoint/intervención**, rango 850000..880039, completamente disjunto de fixtures y de F9-Q 800000..830039, de F9-P 700000+, de sus entrenamientos y del holdout histórico 900000+ **que permanece cerrado**. **850000+ ya está consumido** después de su primera evaluación.
- Cada par contiene dos historias con la pista original/invertida (key/ball), iguales en paredes, posición, orientación y acciones docentes hasta el primer punto de decisión divergente. El *actor* procesa las observaciones pero no mueve la simulación en ese prefijo docente.
- En **TODAS** las condiciones la lectura F4 está forzada a cero en la bifurcación y durante toda la continuación autónoma desde ese punto. En todas las condiciones blackout F4 está también ausente durante los mismos diez pasos antes del cruce; solo difiere el tratamiento de H y/o C.
- El registro de cada caso incluye **primera acción** y **primer brazo espacial** elegido. Nunca reemplazar el desenlace de dos primeras elecciones correctas por éxito individual o recompensa. Los pares sin compromiso se mantienen en el denominador.
- En cada condición se comprueban \`intact\` y \`swap_HC\` entre las historias opuestas. **Transferencia hacia donante** exige salida correcta para el destinatario bajo \`intact\` y salida opuesta correcta bajo el trasplante, para **ambas** pistas del mismo mundo.
- El control positivo \`blackout_patch_normal_fork_HC\` debe reproducir **primera acción y primer brazo exactos de \`normal_reference\` caso por caso**. Pasó en fixtures y confirmación. Se trata de validación del procedimiento, **NO evidencia secundaria independiente**.

## Resultado primario: congelar H/C durante apagón de F4

**No confirmado en ninguno de los tres tipos de checkpoint**. La congelación \`blackout_freeze_HC\` no mejora respecto a \`blackout_update\` ni los pares con ambas primeras salidas correctas ni la transferencia causal hacia el donante: deltas por semilla **[0,0,0,0]** en los tres grupos y ambas métricas; **0/4 réplicas positivas**, frente al umbral preregistrado de mejora en al menos 3/4.

| Condición — primeros brazos correctos en ambas pistas | BC (160 pares) | PPO sham (160) | PPO blackout (160) |
|---|---:|---:|---:|
| Historia con F4 normal, anulado en fork/continuación | **110** | **107** | **40** |
| Blackout desde post-cue, H/C actualizando | **0** | **0** | **0** |
| **Blackout, congelar H y C durante intervalo** | **0** | **0** | **0** |
| Blackout, congelar **solo H** | 0 | 0 | 0 |
| Blackout, congelar **solo C** | 0 | 0 | 0 |
| Blackout, restaurar H/C natural **en mitad del intervalo** | 0 | 0 | 0 |
| Blackout, restaurar H/C natural **en último paso antes del cruce** | **34** | **49** | **40** |
| Blackout, restaurar H/C natural **en el cruce** (control positivo) | 110 | 107 | 40 |
| Blackout, restaurar **solo H** natural en cruce | 2 | 14 | 21 |
| Blackout, restaurar **solo C** natural en cruce | 0 | 29 | 0 |

**Todos los conteos de transferencia completa del donante son exactamente iguales a los pares con ambas salidas correctas de la columna correspondiente.** El trasplante causal del estado completo cambia ambas salidas **solo cuando** el estado intervenido permite usar la identidad de la pista.

El resultado primario negativo significa que **preservar H/C temprano, tal como están justo después de ver la pista, no basta para que la política ejecute una decisión correcta diez pasos después**. Esto no demuestra que la representación de identidad esté ausente ni que las actualizaciones sean siempre perjudiciales; el estado recurrente codifica también orientación, fase temporal y otras variables del controlador.

## Hallazgo secundario: rescate cercano a la decisión

La intervención \`blackout_patch_normal_last_HC\` introduce el estado H/C que habría tenido la **misma política, mismo cue, mismo mundo y mismo instante** si F4 hubiera estado disponible, justo **antes de procesar el último frame del apagón**; el actor todavía procesa ese último frame con F4=0 y se actualiza una vez. Después toma su decisión desde el mismo fork, siempre sin F4.

- BC: **34/160 pares** con ambas salidas correctas y transferencia hacia donante.
- PPO sham: **49/160 pares**.
- PPO blackout: **40/160 pares**.
- Restaurar a la **mitad del apagón** no rescata ningún par (**0/160** en los tres grupos).
- Restaurar el estado natural **justo en fork** sí reproduce exactamente 110/160, 107/160, 40/160, pero esto era el **control de instrumentación**, no un tratamiento experimental que enseñe memoria nueva.

### Heterogeneidad por semilla

| Checkpoint | Referencia normal | Rescate último paso | Rescate mitad |
|---|---|---|---|
| BC | [31,31,18,30] | **[25,9,0,0]** | [0,0,0,0] |
| PPO sham | [8,40,38,21] | **[0,20,29,0]** | [0,0,0,0] |
| PPO blackout | [0,0,3,37] | **[0,0,3,37]** | [0,0,0,0] |

El rescate en el último paso es **heterogéneo**, con ganancia concentrada en dos semillas de cada familia (y una semilla dominante en PPO blackout). No es una demostración de robustez uniforme: esta prueba no preregistró superioridad del rescate último paso como resultado principal, y **no se debe seleccionar la mejor semilla por ello**.

### Diferencias H frente a C

Tras restaurar solo H natural en el fork, se recuperan [2,14,21] pares totales en BC/sham/blackout; restaurar solo C natural recupera [0,29,0]. Los patrones varían entre políticas. Por tanto, **no** podemos atribuir universalmente el mecanismo a H o a C con esta evidencia. La restauración del estado H/C **conjunto** es mucho más consistente.

## Interpretación mecanicista, sin sobreatribución

Estos datos limitan tres explicaciones:

1. **«Solo hace falta conservar el estado anterior a la desconexión»**: no, congelar H/C no repara el resultado. Tal conservación deja fuera la evolución recurrente y la alineación temporal que el controlador pueda necesitar.
2. **«Basta con devolver el estado natural en cualquier momento»**: no; restaurarlo a mitad del apagón no rescató las dos salidas, mientras hacerlo cerca del fork a veces sí.
3. **«Con F4=0 en el momento de decidir es imposible utilizar H/C para elegir»**: no; la historia normal y el control positivo de restauración en fork muestran decisiones cue-dependientes sin F4 en ese instante.

La explicación compatible es una **dependencia temporal del estado H/C completo y de cómo la política lo interpreta en la decisión**. Un código cue decodificable externamente puede coexistir con un controlador que no usa la señal al cabo de varias actualizaciones con F4 ausente. F9-R no demuestra que el LSTM «reescriba un recuerdo» con mecanismo localizado, ni separa por completo deriva de estado, fase del controlador, valores de otros elementos del vector, o efectos de injertar estados fuera de distribución.

La restauración exacta en fork funciona **por construcción**, de modo que no se utilizará para reivindicar por sí sola descubrimiento causal adicional. El rescate de última hora no era desenlace primario y requiere replicación específica en una **nueva partición** antes de afirmar mecanismo general.

## Siguiente pregunta, sin desviarse del objetivo de INFINITO

Lo más útil científicamente sería localizar **qué componentes funcionales de la actualización recurrente cambian la salida** durante los últimos pasos, separando codificación de la pista, dinámica del controlador y requisito de estado temporal. Se puede diseñar un ensayo de **clamp/patch causal de subespacios** H/C en un solo paso próximo al cruce, con un conjunto de entrenamiento de proyección distinto de la confirmación para evitar ajustar a resultados previos. Mantener de nuevo como criterio central pares que eligen correctamente ambas salidas, más transferencia hacia la pista donante.

No iniciar otro entrenamiento PPO para maximizar retorno y no reutilizar \`850000+\` ni \`900000+\`. F9-R es un experimento mecanicista local bajo prefijos docentes, no una prueba de autonomía completa, conciencia o memoria general.
