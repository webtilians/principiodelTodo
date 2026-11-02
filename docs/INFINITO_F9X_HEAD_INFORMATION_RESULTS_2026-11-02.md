# INFINITO F9-X / F9-X-B — Auditoría de la información usada por la cabeza READ

**2026-11-02.** Cuatro réplicas completas en GitHub Actions. El experimento F9-X se registró en el commit `6adf6283092d324e8e39afc00bf2e437497679de` antes de analizar los nuevos mundos. [Ejecución F9-X #41366421641](https://github.com/webtilians/principiodelTodo/actions/runs/41366421641). F9-X-B es un análisis por canales explícitamente **posthoc** de los mismos datos: [Actions #41366737686](https://github.com/webtilians/principiodelTodo/actions/runs/41366737686). No es una nueva confirmación.

## Pregunta
La cabeza neuronal congelada `lex_hc` usa los 148 valores públicos del cruce y 64 de H + 64 de C previos. Queríamos separar si el acceso F4 READ depende de una clave contextual de tipo llave/bola, de la historia de disponibilidad de F4 o de una estimación más general del valor causal de READ.

## Fuentes congeladas y mundos
- Exactamente los actores F9-P y las cabezas F9-V de DEV (#41345497955), con validación de SHA y sin reentrenamiento.
- Una sola ejecución de 32 mundos inéditos por actor `9_500_000+10_000*index+j`, j=0..31. Cada mundo tiene dos cues contrafactuales y dos regímenes F4: 512 estados/128 mundos en total.
- Los 640 estados ya consumidos `8300000+` NO se reutilizaron. `9300000+` (F9-W) y `9600000+` (F9-X futuro) permanecen sellados.
- Prefijo docente hasta primer cruce y después continuación autónoma sin F4. El oráculo `D=correcto_READ-correcto_SKIP` solo evalúa, nunca entra en la cabeza.

## F9-X — resultado prospectivo exploratorio

| Estrategia | Primera salida correcta /512 | READ | Útiles | Perjudiciales | Neutras |
|---|---:|---:|---:|---:|---:|
| lex_hc congelada | **365** | **22** | **22** | 0 | 0 |
| lex_public | 343 | 0 | 0 | 0 | 0 |
| rule key + historia normal (PRIVILEGIADA) | 368 | 128 | 25 | 0 | 103 |
| always | 379 | 512 | 53 | 17 | 442 |
| never | 343 | 0 | 0 | 0 | 0 |
| oráculo diagnóstico | 396 | 53 | 53 | 0 | 0 |

- La cabeza consiguió precisión de READ **22/22** y recall de oportunidades **22/53**, no 53/53.
- Los **22 READ proceden solo de cue=key, historia F4 normal, réplicas 0 y 3**. Por índices: **11, 0, 0, 11**. Las réplicas 1 y 2 conservan un sesgo SKIP total.
- En la condición key+normal hay 25 casos D+ de 128. La cabeza detecta 22 de los 25; la regla con acceso privilegiado al cue y al régimen lee los 128 (25 útiles, 103 neutros). Por tanto la cabeza no equivale literalmente a una regla key/normal, pero su cobertura está restringida a ese estrato; ignora **28** positivos D+ en otros estratos, además de los tres key/normal restantes.

### Trasplantes dentro de pares con idéntica observación pública

| Intervención solo en entrada de la cabeza | Decisiones READ/SKIP modificadas /512 |
|---|---:|
| H del cue opuesto, mismo mundo/régimen | 5 |
| **C del cue opuesto** | **39** |
| H+C del cue opuesto | 44 |
| H+C de la otra historia F4, mismo mundo/cue | 44 |

Los 256 pares con cues invertidos tuvieron solo 22 discrepancias de decisión (11 en réplica 0, 11 en 3), todas en la historia F4 normal. Debido a igualdad exacta de entrada pública, tales discrepancias exigen usar H/C. Los trasplantes C causan cambios de decisión mucho más frecuentemente que H; en réplicas 1 y 2 los logits cambian pero no cruzan el umbral.

Gradientes integrados desde la media de TRAIN original apoyan mayor sensibilidad del logit al grupo C en las cuatro réplicas, pero no permiten inferir conceptos específicos y dependen del punto base.

## F9-X-B — células concretas, exploración posterior sin nuevos mundos

Se parchearon una a una las **64 unidades de H y 64 de C** de cada actor usando como donante la pista invertida en mismo mundo/régimen o la otra historia del mismo mundo/cue. Además se midió cuán consistentemente cambia el valor de cada célula entre key/ball y normal/blackout. Estos rankings provienen de los mismos datos utilizados para descubrir el efecto global C y **no son resultados confirmatorios**.

### Réplica 0
- **C32**: cambiar solo esa célula entre pistas provocó **10 cambios de READ**. Su diferencia key−ball conserva signo en **64/64 pares**; también presenta fuerte separación normal/apagón.
- **C44**: también **10 cambios de READ** al parchear una sola célula, con dirección key−ball consistente en **61/64 pares** (95,3%). Está asimismo relacionada con el historial.
- **C19**: cambiarla por el valor del otro régimen provoca **9 cambios de READ**, aunque no discrimina de forma tan consistente el cue (42/64 pares del mismo signo). Muestra sensibilidad a la historia, no un detector semántico puro.
- Ninguna célula H individual alteró la decisión READ al parchearse sola. Que H en conjunto no cambiase decisiones de réplica 0 no implica que sea matemáticamente irrelevante para el logit.

### Réplica 3
- **C8**: el parche de esta célula cambia **3 decisiones READ** en pares de cue invertido; diferencia key−ball del mismo signo en 63/64 pares.
- **C35**: parchear esta célula entre regímenes cambia 1 decisión READ; su firma también está muy asociada al cue y al régimen.
- H en conjunto puede afectar 5 decisiones, pero ninguna unidad H individual alcanzó el umbral por sí sola.

### Réplicas 1 y 2
En ambas, las unidades H y C contienen diferencias cuantitativas entre pistas/regímenes, y algunas tienen firma cue o historia altamente consistente; sin embargo ningún parche de una unidad ni del grupo completo cambió ninguna decisión binaria. Esto es compatible con un lector que utiliza señales en el logit, pero mantiene una preferencia SKIP dominante.

**Precaución:** una célula C32 con señal estable no demuestra que exista una 'neurona de la llave'; puede mezclar identidad de pista, disponibilidad de F4, dinámica del actor y variables correlacionadas. El mismo índice de célula en otros checkpoints NO es comparable semánticamente. No se midió composición de decodificadores ni independencia respecto a variables temporales ocultas.

## Conclusión científica acotada

1. La cabeza sí depende de **estado recurrente**: se observan decisiones distintas con entrada pública idéntica y los trasplantes controlados del estado pueden alterar READ.
2. **C es más decisivo que H** para este comportamiento, bajo la intervención realizada. Existen canales concretos de C con influencia causal parcial y señales separadoras de cue/historia.
3. La política final NO equivale a leer cada vez que hay key+normal, pero está **restringida a ese estrato** y no alcanza el valor causal positivo fuera de él; por ello no podemos declararla un predictor general de la utilidad de la memoria.
4. La conclusión más parsimoniosa es que combina un conjunto restringido de patrones de cue+historia+estado del actor, distribuidos en C, con un umbral de lectura conservador. No sabemos aún si representa una variable causal general D que transfiera a nuevos regímenes.
5. F9-V confirmó un gate de robustez negativo. F9-X y F9-X-B no invalidan ese criterio ni rehabilitan el resultado original.

## Próximo experimento sugerido
Para separar cue y valor causal en condiciones no vistas, crear hipótesis F9-Y **prerregistrada con datos completamente nuevos**, manteniendo el mismo actor y controles pero contrabalanceando positivos D+ por cue/régimen y controlando la fuerza del umbral. La evaluación debe exigir precisión, recall, robustez en 4 actores y mejora frente a reglas simples, con particiones TRAIN/DEV/CONFIRM aisladas. Un separador H/C que solamente detecte cue o estado de blackout no bastará.
