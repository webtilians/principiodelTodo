# INFINITO F9-W — ¿la escasez de ejemplos positivos explica las cabezas que no leen?

**2026-11-01. RESULTADO DE DESARROLLO, nuevo split independiente.** Ejecución completa [GitHub Actions #41355239064](https://github.com/webtilians/principiodelTodo/actions/runs/41355239064): cuatro entrenamientos + agregador SUCCESS técnico; pre-registro previo `d1e2e44775491cd6f8a1910252df55a5d25b2920`. **El gate científico de escasez de positivos NO se cumplió**.

## Diseño ejecutado

Cuatro checkpoints congelados F9-P y memoria F4 real, F9-V intervenciones READ vs SKIP solo en el primer cruce con prefijo docente; cabeza observa exclusivamente 148 públicos + H/C previo, o públicos en el control. Semillas TRAIN nuevas `9_100_000+`: short = 32 mundos/índice (128 estados), long =128 mundos/índice (512 estados), short_weighted = mismo short con BCE ponderada, long_public = long sin H/C. Los cuatro tratamientos recibieron **200 actualizaciones Adam** con MLP de igual tamaño. DEV disjunto `9_200_000+`: 16 mundos ×2 cues ×2 historias ×4 réplicas = 256 decisiones. F9-V CONFIRM `8_300_000+` **no reutilizado**. F9-W CONFIRM `9_300_000+` **sellado**.

## Resultados agregados

| Cabeza | Primer brazo correcto | READ | READ positivas | READ dañinas | READ neutras |
|---|---:|---:|---:|---:|---:|
| short32_hc (32 mundos) | **191/256** | 29 | 25/27 | 3 | 1 |
| long128_hc (128 mundos) | 190/256 | 21 | 21/27 | **0** | 0 |
| short32_weighted_hc | **191/256** | 40 | 26/27 | 4 | 10 |
| long128_public (sin H/C) | 169/256 | 0 | 0/27 | 0 | 0 |

En este conjunto, **más ejemplos redujo lecturas equivocadas, pero también perdió algunas lecturas beneficiosas**. La cabeza con 128 mundos hizo 21/21 lecturas verdaderamente útiles (precisión 100%), pero detectó solo 21/27 oportunidades (77.8%), contra 25/27 (92.6%) en short32. La BCE ponderada recuperó 26/27 beneficiosas pero añadió 14 accesos que no favorecieron el resultado (4 perjudiciales y 10 neutros); no mejoró los 191 aciertos.

### Indices clave 1 y 2

| Índice | Positivos TRAIN 32 → 128 mundos | Beneficiosas READ DEV 32 →128 | Aciertos DEV 32 →128 | READ dañinas 32 →128 |
|---|---:|---:|---:|---:|
| 1 | 12 → **36** | 3 → **0** | 51 → 48 | 0 → 0 |
| 2 | 8 → **24** | 1 → **0** | 45 → 47 | 3 → 0 |

Los positivos aumentaron en ambos índices como se pre-registró, pero el método de 128 mundos acabó decidiendo **SKIP siempre** en ambos, pese a disponer ahora de 36 y 24 ejemplos positivos respectivamente.

**Gate preregistrado:** exige +3 positivos TRAIN por índice 1 y 2 (cumplido: +24,+16); ≥2 nuevas READ beneficiosas DEV agregadas en 1/2 (FALLÓ: −4); ≤2 READ dañinas adicionales (cumplido: −3); caída no peor que 2 aciertos agregados (cumplido: −1). **Resultado compuesto NEGATIVO.** F9-W no abre CONFIRM `9_300_000+`.

## Interpretación

La falta absoluta de ejemplos positivos **no explica por sí sola** el colapso a SKIP. Multiplicar los mundos y disponer de 3–4× más positivos no consiguió recuperar la selección de READ en réplicas 1/2. Más datos también modifican composición/dificultad/distribución de inputs; no podemos inferir por sí solos que exista una arquitectura insuficiente, un fallo de crédito temporal o un problema causal específico.

El contraste ponderado indica una tensión entre recall y precisión: subir la prioridad de los positivos recupera algunos, pero puede disparar consultas perjudiciales o neutras. Sería más útil investigar representación H/C, causalidad de ventanas de lectura (en lugar de un solo paso) y calibración/criterios de abstención, todos con nuevos splits y sin tunear el F9-V CONFIRM ya consumido.

## Limitaciones

Resultados de desarrollo; los 256 estados incluyen dos pistas y dos regímenes por mundo, no 256 mundos independientes. Prefijo hasta cruce guiado y cabeza entrenada con etiquetas del oráculo experimental: no generación autónoma de nuevas políticas ni navegación completa. Success en Actions implica integridad, NO victoria científica.
