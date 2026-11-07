# INFINITO E11-A — Confirmación de aprendizaje de relevancia FUTURA en claves nuevas

**1 de noviembre de 2026 · CI superado · criterio preregistrado superado EN UN ENTORNO SINTÉTICO ESTACIONARIO.** La inversión de la distribución produjo un fallo grave de transferencia. No confundir este resultado con la memoria semántica o autonomía general de INFINITO 3.0.

## Procedencia y verificaciones

- [**Protocolo preregistrado ANTES de código y confirmación**](./INFINITO_E11A_LEARNED_RELEVANCE_PROTOCOL.md): commit \`c174ed3adc2090294b7b791c007a3fe40fd5e4d0\`.
- [**GitHub Actions #41317309779 — SUCCESS**](https://github.com/webtilians/principiodelTodo/actions/runs/41317309779): tests, regresiones originales, TRAIN y DEV, congelación de cinco modelos, CONFIRM en job separado. Ningún error de CI.
- [**Datos confirmatorios: métricas, los cinco contrastes, intervalos y controles**](./INFINITO_E11A_CONFIRMED_RESULTS_2026-11-01.json).
- [**Artifact bruto confirmatorio**](https://github.com/webtilians/principiodelTodo/actions/runs/41317309779/artifacts/13088200715): 3×3.200 pares del mismo stream entre políticas, un JSON por régimen, los 9.600 resultados por política, report.json.
- [**Artifact de los cinco modelos congelados y 800 DEV**](https://github.com/webtilians/principiodelTodo/actions/runs/41317309779/artifacts/13088310237).
- [**PR #32 — borrador, desde master, sin cambios en \`src/infinito3/\`**](https://github.com/webtilians/principiodelTodo/pull/32).

### Qué NO repetimos de F4

F4 aprendía las consultas más probables entre ocho tipos/IDs que se repetían. E11-A exige generalización a **identidades completamente nuevas en cada episodio**, y a atributos combinados de las experiencias. No hay tabla de importancia por ID ni conocimiento anticipado de la clave futura.

### Método

- Cada episodio contiene **16 eventos nuevos** con ID único, payload y siete características observables: urgencia, estado pendiente, consulta previa, fuente cliente, fuente automatizada, recencia e interacción pendiente×consulta-previa.
- **Solo cuatro slots** reales; no compresión, ni memoria auxiliar de claves descartadas, ni acceso a eventos completos durante inferencia. \`consider(memory,event)\` se llama antes de que se consulte la futura clave.
- El entrenador observa la **clave solicitada DESPUÉS** del episodio y actualiza los siete coeficientes de un **predictor log-lineal** de demandas futuras con un gradiente online. Es **supervisión retrospectiva con feedback diferido**, no una política RL que aprenda directamente de recompensas de un LLM. Durante CONFIRM los pesos permanecen congelados.
- Cinco políticas aprendidas independientemente a partir de **2.500 historias TRAIN cada una**, identidades separadas de 800 DEV y de 3.200 CONFIRM por régimen (5 réplicas ×640). Las réplicas independientes de entrenamiento son las unidades principales para bootstrap.
- Baselines con misma memoria: FIFO, reservorio, prioridad fija fuerte, recencia+prioridad, **mismo modelo entrenado con consultas aleatorias barajadas**, y un oráculo de distribución (**solo techo**, conoce cómo se generan las consultas pero no recibe la consulta concreta antes de guardar).
- Tres tests confirmatorios sobre streams equivalentes: consultas predictivas estacionarias, consultas uniformes independientes, e inversión completa de la asociación entre características y demanda (OOD).
- No se ejecuta ningún LLM ni embeddings; prueba de disponibilidad de un recuerdo con valor exacto, no de comprensión semántica.

## Confirmación preregistrada: utilidad FUTURA predecible

**3.200 episodios nuevos, 4/16 slots de memoria.**

| Política | Recuerdos necesarios retenidos | Tasa |
|---|---:|---:|
| FIFO | 889/3200 | 27,78% |
| Reservorio | 780/3200 | 24,38% |
| Prioridad fija **(baseline práctico más fuerte)** | **2208/3200** | **69,00%** |
| Recencia+prioridad | 1364/3200 | 42,63% |
| Modelo aprendido con targets de TRAIN **barajados** | 1052/3200 | 32,88% |
| **Selector aprendido E11-A** | **2484/3200** | **77,63%** |
| Oráculo de distribución (techo explicativo) | 2494/3200 | 77,94% |

La ganancia de E11-A sobre el **mejor baseline práctico** es **+8,625 puntos porcentuales**. Las cinco réplicas de entrenamiento independientes favorecieron al aprendido frente a prioridad fija:
\`[+8,125,+9,375,+7,969,+9,063,+8,594]\` pp. Bootstrap por réplica (10.000 muestras, semilla 1711): **[+8,156,+9,094] pp**. El criterio preregistrado pedía **≥5 pp**, al menos **4/5 réplicas** positivas y límite inferior >0: **PASA**.

Este IC no demuestra generalización a otras familias de entornos, pues el generador utiliza la misma relación probabilística para TRAIN y CONFIRM; tampoco hay 3.200 políticas independientes.

## Control negativo indispensable: futuro impredecible

**La demanda es uniforme; las características no predicen nada.** El techo esperado para 4 de 16 es **25%**.

| Política | Consultas acertadas | Tasa |
|---|---:|---:|
| FIFO | 843/3200 | 26,34% |
| Reservorio | 759/3200 | 23,72% |
| Prioridad fija | 840/3200 | 26,25% |
| Recencia+prioridad | 826/3200 | 25,81% |
| Aprendido con feedback barajado | 806/3200 | 25,19% |
| **Aprendido E11-A** | **826/3200** | **25,81%** |
| Oráculo de distribución (política basada en los coeficientes originales; SIN ventaja en universo uniforme) | 830/3200 | 25,94% |

El aprendido no muestra evidencia de habilidad de anticipación cuando no hay ninguna señal predictiva. Está **+2,09 pp** sobre reservorio, por debajo de la tolerancia de 5 pp preregistrada y cerca del techo teórico; no debe interpretarse como capacidad predictiva real bajo demanda uniforme.

## Stress negativo, crucial: inversión de relevancia

Se invierte **completamente** la relación entre rasgos y utilidad futura, pero los modelos quedan congelados; *no* se entrenan/adaptan al nuevo régimen.

| Política | Retención tras invertir patrón | Tasa |
|---|---:|---:|
| FIFO | 691/3200 | 21,59% |
| Reservorio | 823/3200 | 25,72% |
| Prioridad fija | 157/3200 | 4,91% |
| Recencia+prioridad | 434/3200 | 13,56% |
| Aprendido con feedback barajado | 625/3200 | 19,53% |
| **Aprendido E11-A congelado** | **87/3200** | **2,72%** |
| Oráculo que conoce el nuevo régimen (techo EXTERNO) | 1824/3200 | 57,00% |

**Este es el límite más importante:** la selección aprendida puede ser mucho **peor que escoger recuerdos al azar** cuando la futura demanda cambia radicalmente. En un sistema real no deberíamos emplear ese selector como política irrevocable de escritura sin detección de cambios o reserva exploratoria. La reversión no invalida el éxito del protocolo estacionario, pero **sí impide declarar el mecanismo listo para producción**.

La prueba no incluyó adaptación durante el nuevo régimen; no se afirma que el sistema sea incapaz de reaprender, solo que **no generaliza por sí solo a una inversión desconocida**.

## Caveat de diseño y alcance

El generador de consultas fue escrito explícitamente con una distribución log-lineal de los siete atributos observables; el aprendiz usa **la misma familia funcional**, sin conocer los coeficientes, y aprende por feedback de consultas históricas. El resultado cercano al oráculo es por tanto esperable cuando el generador coincide con esa hipótesis. No es un descubrimiento de semántica emergente ni prueba de utilidad con usuarios reales. La aportación adicional respecto a F4 es aprender **coeficientes transferibles a claves nuevas** bajo capacidad estricta y feedback diferido, con baselines fuertes y una prueba de cambio radical.

La comparación con \`learned_shuffled\` controla el código y las estructuras del alumno sin información predictiva real: 32,88% en predictivo, notablemente inferior al 77,63%; el valor >25% se debe a una correlación casual/aleatoria de sus pesos con señales de este generador, y no lo convierte en oráculo.

## Decisión según la hoja de ruta

**E11-A SUPERA su gate local, pero NO está listo para integración a producción.** Recomendamos una única fase E11-B con **RAG semántico + LLM real** y retención aprendida sobre eventos verdaderamente expresados en lenguaje natural, frente a una memoria RAG de idéntico número de registros y tokens; registrar error por sobrealmacenamiento y cambios de distribución.

Para no desviarnos, E11-B debe declarar **antes** de escribir código su mecanismo de seguridad ante cambio de demanda (p. ej., mezcla de slots reservados para exploración, como control con pérdida de capacidad medible) y su criterio de parada. No rescatar rendimiento de E11-A ajustando el generador o las confirmaciones actuales. Si el aprendizaje no mejora la memoria semántica fuerte bajo presupuesto, **cerrar E11 y avanzar a E12 (recuperar y usar recuerdos autonomamente)**.

Las confirmaciones E10 y F9 permanecen intactas; el histórico E11-A \`3300000+\` ya está **consumido**. Código de producción \`src/infinito3/\` sin modificaciones.
