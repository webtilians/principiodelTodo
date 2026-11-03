# INFINITO F10-K — Primitivas recompensadas y composición cero-shot: fracaso científico con una primitiva aprendida

**3 de noviembre de 2026.** Resultado original: **F10K_TECHNICAL_COMPLETE** y **F10K_COMPOSITION_FAIL** (G0 0/4, G1 0/4, G2 0/4, G3 0/4). Ningún parámetro, mundo, gate ni duración se ha cambiado tras ver DEV/OOD. PR de investigación en borrador; confirmar 22300000+ continúa **sellado**.

- [Prerregistro F10-K](INFINITO_F10K_COMPOSITION_PREREG.md) commit **1cbddbd092b4c743f2b6914f36193bbb0db182b8**, publicado antes de escribir el primer archivo de experimento.
- [GitHub Actions F10-K #41653362579](https://github.com/webtilians/principiodelTodo/actions/runs/41653362579): 4/4 trabajos de training/evaluación y auditoría independiente SUCCESS técnico. Artefactos: f10k-original-0..3 (todos los 16 modelos SHA originales, curvas TRAIN y eval.json) y f10k-independent-fixed-scientific-audit (f10k-fixed-audit.json con bootstrap).
- TRAIN 22100000+; DEV 22200000+; OOD 22400000+. **CONFIRM 22300000+ NO GENERADA NI ABIERTA**. 16 checkpoints originales, 24.576 episodios de experiencia ejecutados por brazo/réplica (8.192 por cada P/I/S), 384 updates, igual tipo de tareas y mundos para todos. 20.480 predicciones DEV/OOD en 4 réplicas × 4 brazos × (768+512), 29.696 SHA de **identidad de caso** incluyendo seed/variante (no son 29.696 patrones binarios independientes). Cuatro variantes relacionadas por mundo: IC95 por 2.000 remuestreos de **mundo base**, no de fila.
- Cuatro brazos: A entrenamiento conjunto de P/I/S compartiendo procesador; B currículum P+I→S→repaso con procesador compartido; C el mismo currículo con procesadores públicos/F4 separados (más parámetros); D control **privilegiado** con índice ganador/respuesta como targets. A/B/C obtienen solo recompensa del resultado de sus propias acciones; la recompensa binaria determinista ofrece información correctiva indirecta, NO independencia de verdad.

## La pregunta y su respuesta acotada

Se intentó aprender por recompensa P (p XOR flip, bit público), I (leer/interpretar un único valor en F4, sin flip) y S (elegir última versión firmada e interpretarla, sin flip); después transferir **sin entrenamiento en composición** a M (elegir último firmado entre 9/11, leer, aplicar flip=1). **Todos los ejemplos TRAIN y DEV omiten completamente memoria+flip1**. Los valores F4 no entran en el selector, solo el de la posición elegida llega al procesador.

**Resultado**: hubo adquisición perfecta de una de las tres habilidades, pero adquisición incompleta de P y S. No hay evidencia de composición cero-shot funcional, y estrictamente NO se puede concluir que componer habilidades ya dominadas sea imposible: el requisito previo **G0 falló**.

## Primitivas DEV en el brazo B — currículum y pesos compartidos

| Réplica | P: procesar XOR | I: interpretar un solo recuerdo | S: seleccionar+interpretar | Selector S correcto |
|---|---:|---:|---:|---:|
| 0 | 75,00% | **100%** | 64,84% | 59,77% |
| 1 | 75,00% | **100%** | 51,56% | 43,75% |
| 2 | 75,00% | **100%** | 70,70% | 70,70% |
| 3 | 75,00% | **100%** | 69,53% | 68,36% |

**G0 exige P>=90%, I>=90%, S>=80% y selección>=80% en >=3/4. Resultado G0 0/4.** La lectura elemental I sí se adquirió por experiencia reward-only en las cuatro réplicas: su dificultad es mucho menor que S/M (un solo registro y sin transformación). Esto no prueba memoria temporal compleja.

## Composición completamente inédita — OOD M

| Réplica | A conjunto | B currículum | C sin compartir procesador | D supervisado privilegiado |
|---|---:|---:|---:|---:|
| 0 | 52,34% | 52,34% | 31,84% | **100%** |
| 1 | 46,88% | 46,88% | 17,19% | **100%** |
| 2 | 48,05% | 48,05% | 12,89% | **100%** |
| 3 | 50,00% | 50,00% | 14,06% | **100%** |
| **Promedio** | **49,32%** | **49,32%** | **19,00%** | **100%** |

B es indistinguible de A en toda la OOD por réplica: contraste B−A = 0, IC95 [0,0] en las cuatro. **G3 currículum: 0/4** (no demuestra ventaja del orden). Una puntuación C menor que B es descriptiva: C tiene más parámetros y su intérprete de memoria nunca vio flip=1, por lo que NO identifica un mecanismo causal único.

B composición OOD, IC95 por mundo:
- R0: 52,34%, IC95 **[47,66%; 56,64%]**;
- R1: 46,88%, IC95 **[42,58%; 51,17%]**;
- R2: 48,05%, IC95 **[43,75%; 52,34%]**;
- R3: 50,00%, IC95 **[45,70%; 54,30%]**.

**G1 exigía >=80% OOD, límite inferior IC95>65% y mínimos por estrato, >=3/4 réplicas. Resultado G1 0/4.**

El control D supervisado alcanzó 100% en todas las tareas, incluyendo la composición OOD, lo cual descarta una insuficiencia estructural obvia de capacidad en este entorno específico; NO demuestra que A/B/C puedan aprenderla mediante la señal de recompensa empleada.

## Inspección diagnóstica POST HOC — no nueva hipótesis confirmada

La auditoría preservó por fila la predicción original. Inspeccionando los resultados *después* de aplicar los gates, la primitiva XOR de B resolvió **exactamente tres de sus cuatro casos de tabla de verdad** en cada una de las cuatro réplicas:

| Entrada bit p | flip | Respuesta correcta XOR | Réplicas B que aciertan |
|---|---|---|---:|
| 0 | 0 | 0 | 4/4 |
| 0 | 1 | 1 | **1/4** |
| 1 | 0 | 1 | 4/4 |
| 1 | 1 | 0 | **3/4** |

Tres réplicas aprendieron una respuesta tipo «p AND no-flip» (fallan 0,1); otra aprendió una respuesta tipo OR (falla 1,1). Ambas son estrategias estables de **75%** en este conjunto equilibrado, pero **no implementan XOR**. En la composición M siempre flip=1: por eso el procesador puede producir una salida casi constante ignorando el bit observado.

La causalidad revela ese mecanismo con más fuerza:
- En **todas las réplicas**, cambiar por permutación los bits de F4 produce **cero cambios de precisión compuesta** en B. B−SHUFFLE = 0, IC95 [0,0], 4/4.
- Inyectar el registro ganador desde el evaluador en la cabeza de procesamiento (**oracle_slot solo diagnóstico**, sin pérdida privilegiada durante TRAIN) tampoco cambia la precisión compuesta de B. La selección por sí sola no rescata la interpretación con flip=1.
- Anular flip sí cambia resultados en algunas réplicas (R0 +20,51pp; R1 +0,78pp; R2 +35,16pp; R3 +32,42pp para B menos no_flip), pero esto no satisface G2, que requiere **dependencia simultánea de la evidencia F4 y del flag flip**.
- La selección ganadora de B en OOD M puede ser relativamente alta en R2/R3 (**87,11% y 81,64%**), pero la respuesta continúa en ~48–50%: **seleccionar el dato correcto no implica haber aprendido a utilizarlo**.

**G2 reutilización causal: 0/4.** Los mecanismos aprendidos no están combinándose funcionalmente en esta OOD.

## Interpretación y límites

El resultado elimina una explicación sencilla, pero NO responde definitivamente «no, no puede». Evidencia actual:

1. **Sí aprende una operación simple**: copiar/interpetar un valor leído correctamente cuando es el único disponible, usando solo recompensa terminal (I 100% 4/4).
2. **No aprende las otras dos con suficiente estabilidad**: XOR se atasca en 3/4 de su tabla, selección en 44–71% DEV.
3. **No compone las operaciones inacabadas**: sin haber visto M, la respuesta no utiliza causalmente los valores F4.
4. **Ordenar las mismas experiencias como currículo no aporta ventaja** en el protocolo fijado.
5. Una arquitectura **explícitamente factorizada y con cableado de operaciones dado** limita cualquier futura conclusión positiva a transferencia composicional en un espacio predefinido, no invención autónoma de una política cognitiva inédita.

La formación de un «mínimo local» de XOR es una hipótesis **diagnóstica post hoc** razonable, no un hecho causal probado: se necesitarían intervenciones de optimización preregistradas y mundos nuevos. El reward binario de acierto tiene alta varianza para REINFORCE en dos bits, aunque su promedio favorece la respuesta correcta; una tabla casi correcta puede persistir.

## Decisión

**STOP F10-K**. Conservar tanto la evidencia de I 100% como los FAIL de G0/G1/G2/G3. No retocar entropía, número de épocas, arquitectura ni selección de checkpoint después de OOD. No abrir la confirmación 22300000+, porque el prerequisito de adquisición G0 fracasó.

**Siguiente hipótesis para F10-L, NO ejecutada aquí**: investigar si la dificultad principal es el aprendizaje local de una operación no lineal de dos bits desde consecuencias. Comparar REINFORCE estándar con (i) entrenamiento contextual con dos acciones realmente ejecutadas bajo mismo estado y mismo presupuesto, y (ii) objetivos de predicción de experiencia sin indicar verdad/ganador, en nuevos mundos y expresiones binarias más diversas. **Exigir un control positivo de dominio completo de XOR y selección en DEV antes de hablar de composición;** después congelar y comprobar transferencias en combinaciones verdaderamente nuevas con intervención F4. Cualquier nueva señal que equivalga a revelar la respuesta debe declararse como tal, aunque se presente como recompensa de entorno.

No se tocó master, ni código del producto ni datasets Bemula/E10.
