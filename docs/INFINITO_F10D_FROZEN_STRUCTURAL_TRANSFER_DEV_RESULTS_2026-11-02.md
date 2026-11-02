# INFINITO F10-D — Transferencia estructural de actores congelados

**Resultado final DEV, 2026-11-02:** `F10D_STRUCTURAL_TRANSFER_FAIL` por los criterios preregistrados, pero **beneficio causal de memoria y ventaja general del actor relacional congelado sobre el unario en los mundos de transferencia**. No interpretar FAIL del conjunto como ausencia de transferencia, ni mejora global como prueba de una regla abstracta general.

- [Protocolo prerregistrado antes de implementar el generador](./INFINITO_F10D_FROZEN_STRUCTURAL_TRANSFER_PREREG.md), commit `f4888f557b7092660fca9c5a0e04520e21b26fd8`.
- [GitHub Actions #41449564639](https://github.com/webtilians/principiodelTodo/actions/runs/41449564639): **SUCCESS técnico**. Los cuatro trabajos validaron SHA de cinco checkpoints congelados; el agregado de gates corrió independientemente con stdlib. Artifact `f10d-frozen-actor-{0,1,2,3}` contiene todos los 256 casos DEV/actor y las comparaciones/celdas; artifact `f10d-independent-scientific-transfer-gates` contiene la matriz auditada.
- **No hay entrenamiento nuevo, cambio de arquitectura o apertura de confirmación**.

## Qué se midió realmente

Cada uno de los cuatro actores originales F10-C se mantuvo **completamente congelado** desde los checkpoints del run #41447171697: (1) GRU64 con ranking entrenable por pares y `topk5`, (2) unario con F4 y (3) GRU-only. También se compararon los dos checkpoints F10-B originales (run #41443064754), ambos congelados y verificados contra manifiestos por SHA. **Todos los modelos ejecutaron exactamente los mismos mundos F10-D nuevos**: `15_200_000+10000*index+j`, j=0..63, index=0..3. No se reutilizó un solo mundo F10-B/C. Reserva `15_300_000+` sin abrir.

Cada mundo nuevo contiene **60 informes públicos**, 24 claves y exactamente cuatro preguntas finales no anticipables, cada una con 3, 4, 6 u 8 informes para su clave; estas cardinalidades rotan entre las 4 celdas firmado/no firmado × estable/revisado. Las cuatro consultas comparten un historial idéntico; cuatro patrones de fuente (IID, fuente alineada con firma A/B y alternante) rotan equilibradamente. La respuesta se deriva SOLO de la última firma registrada públicamente, no de una etiqueta oculta para el actor. La credencial de firma es un bit sintético, no una verificación criptográfica. Hay **256 mundos independientes y 1024 consultas dependientes** a través de 4 réplicas.

El mismo actor congelado recibe READ auténtico, SKIP, F4 obsoleta o F4 con credencial falsificada, manteniendo constante historial público/estado recurrente/consulta y pesos. Se evalúan controles algorítmicos puramente públicos: último informe, mayoría, último firmado (=límite exacto de la definición de respuesta).

## Hallazgo 1 — Transferencia de rendimiento incompleta, pero mejora real frente al control unario

| Réplica | Relacional READ /256 | Unario READ /256 | Relacional SKIP /256 | GRU-only F10-C /256 | READ antiguo F10-B /256 | READ−SKIP | Relacional−unario |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | **229** | 197 | 129 | 159 | 198 | **+39,1 pp** | **+12,5 pp** |
| 1 | **252** | 205 | 112 | 138 | 209 | **+54,7 pp** | **+18,4 pp** |
| 2 | **223** | 201 | 132 | 156 | 199 | **+35,5 pp** | **+8,6 pp** |
| 3 | **247** | 206 | 128 | 146 | 203 | **+46,5 pp** | **+16,0 pp** |
| **TOTAL** | **951/1024 = 92,9%** | **809/1024 = 79,0%** | **501/1024 = 48,9%** | **599/1024 = 58,5%** | **809/1024 = 79,0%** | **+43,9 pp descriptivo** | **+13,9 pp descriptivo** |

Los **cuatro actores congelados relacionales** aventajan al mismo actor con SKIP y al **control unario F10-C congelado**, pese al cambio de tamaño de historial/credenciales/fuentes, pero los valores no son 1024 mundos independientes: son 64 mundos por réplica, cada uno con cuatro consultas relacionadas.

IC95% de **READ−SKIP**, bootstrap por mundo (64 unidades/actor): idx0 [32,8;45,3] pp, idx1 [48,8;60,5], idx2 [27,3;43,4], idx3 [39,8;53,1], todos positivos. IC95% de **relacional−unario**, bootstrap por mundo: idx0 [8,6;16,4], idx1 [13,7;23,0], idx2 [5,5;12,1], idx3 [12,1;19,9] pp. Hay **evidencia exploratoria de una ventaja de arquitectura en esta familia modificada** sin entrenamiento sobre esos mundos. Las redes difieren en parámetros y estructura, por lo que no se puede atribuir la ventaja **aisladamente** a su comparador pairwise.

## Hallazgo 2 — La falla no es la longitud 8 por sí sola, sino una condición de validez

| Registros por clave | READ relacional | Unario congelado | GRU-only |
|---|---:|---:|---:|
| 3 | 221/256 = 86,3% | 180/256 = 70,3% | 139/256 |
| 4 | 242/256 = 94,5% | 182/256 = 71,1% | 144/256 |
| 6 | 241/256 = 94,1% | 216/256 = 84,4% | 145/256 |
| 8 | 247/256 = 96,5% | 231/256 = 90,2% | 171/256 |

**El `topk5` fijo no hizo fracasar de forma general la condición de 8 informes**: de hecho fue el mejor bin del relacional. Eso no demuestra que el algoritmo tenga acceso a todas las ocho versiones; solo muestra que en este generador la información retenida a menudo resulta suficiente.

| Estrato | READ relacional | Unario congelado | Último informe sin verificar |
|---|---:|---:|---:|
| Firmado + estable | 253/256 = 98,8% | 225/256 | 256/256 |
| No firmado + estable | 254/256 = 99,2% | 195/256 | 0/256 |
| **Firmado + revisado** | **192/256 = 75,0%** | 140/256 | 256/256 |
| No firmado + revisado | 252/256 = 98,4% | 249/256 | 0/256 |

El problema más claro se concentra en **`signed|revised`**: el relacional solo alcanza **39/64** casos en la réplica 0 y **32/64** en la réplica 2 (pero 64/64 en la réplica 1 y 57/64 en la 3). En consecuencia, solo **2/4 réplicas** satisfacen el mínimo de 80% por todos los estratos y bins de cardinalidad. La mejora frente al control unario no equivale a transferencia uniforme de validez.

Por fuente, el relacional obtiene **240/256** IID, **237/256** firmado-A, **235/256** firmado-B y **239/256** fuente alternante: en agregado se mantiene alto en los cuatro regímenes, pero el gate exige éxito **por réplica y estrato**, no agregados favorables.

## Hallazgo 3 — La ventaja relacional no está donde se exigió

El preregistro especificó en >=3/4 réplicas:
1. mejora general relacional−unario >=5 pp, **Y**
2. mejora >=8 pp precisamente en `unsigned|revised`, el estrato históricamente problemático.

La condición general PASA **4/4**; la condición específica NO PASA **0/4**. Diferencias en `unsigned|revised` por réplica: idx0 **+2/64 = +3,1 pp**, idx1 **−3/64 = −4,7 pp**, idx2 **+3/64 = +4,7 pp**, idx3 **+1/64 = +1,6 pp**. El unario ya había alcanzado casi techo ahí, pero eso no autoriza cambiar un criterio preregistrado tras ver resultados. **G3 FAIL (0/4)**.

## Causalidad y ablaciones

READ verdadero relacional 951/1024 vs SKIP 501/1024: **+450 decisiones correctas**. D = +1 en **483** casos, D = 0 en **508**, D = −1 en **33**. Las 4 réplicas superan delta>=0,10, el límite inferior de bootstrap por mundo >0, y READ más de +5 pp respecto al GRU-only congelado.

READ_obsoleto: **822/1024 = 80,3%**. READ_credencial falsificada: **759/1024 = 74,1%**; la variación entre réplicas es fuerte. Son intervenciones sintéticas fuera de distribución; no permiten afirmar resistencia real a ataques ni prioridad autenticada robusta.

Controles basados solo en información pública: último informe **512/1024 = 50%**; mayoría **662/1024 = 64,6%**; último firmado **1024/1024 = 100% por definición de la tarea**. Este último es un baseline algorítmico con regla estructural conocida, no rendimiento de un modelo aprendido.

## Gates prospectivos y límites

- Integridad de originales, cinco SHA por actor, splits distintos y cuatro consultas con historia idéntica, cardinalidades 3/4/6/8, regímenes balanceados: **PASS técnico** (4/4).
- **G1** transferencia de validez por estrato, cardinalidad y régimen fuente: **FAIL 2/4** (se exigían >=3/4).
- **G2** utilidad causal READ−SKIP y vs GRU-only en mundos nuevos: **PASS 4/4**.
- **G3** superioridad relacional general **Y** en el caso específico difícil: **FAIL 0/4**.
- Estado global: **`F10D_STRUCTURAL_TRANSFER_FAIL`**, aun con evidencia exploratoria positiva de utilidad de F4 y de ventaja general relacional sobre unario.

No se entrenó un lector READ/SKIP; no se actualizaron pesos, no se usaron F10-B/C TRAIN/DEV, ni se abrió `15_300_000+` confirmatorio o confirmaciones históricas.

## Interpretación y decisión futura (sin ejecutarla en F10-D)

Los resultados rechazan tanto «ambos modelos solo memorizaron íntegramente los cinco registros y no transfieren nada» como «la política relacional aprendió una regla perfecta e invariante de último firmado». La evidencia apoya **transferencia parcial** y utilidad causal de un log externo en una tarea sintética determinada. La falla `signed|revised` y la heterogeneidad de efecto ante firma falsa revelan todavía dependencia de factores estructurales.

La siguiente hipótesis, si se decide continuar, debería evaluar con diseños nuevos y balanceados **la dependencia de la prioridad temporal frente a la credencial** con actualizaciones más diversas y casos adversariales donde el último firmado no coincida siempre con el último cronológico, sin adaptar F10-D tras ver su DEV. Sería preferible también incorporar **nuevas longitudes de historial/ruido** y un actor capaz de tomar decisiones de recuperación costosas; pero F10-D por sí solo **no** justifica abrir una cabeza READ, ni afirmar transferencia a F4 original MiniGrid, RAG, LLM o autonomía general.
