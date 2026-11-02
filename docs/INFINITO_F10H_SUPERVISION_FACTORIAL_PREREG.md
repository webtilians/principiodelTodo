# INFINITO F10-H — Factorial 2×2 de supervisión: selección versus interpretación

**Prerregistro prospectivo · 2026-11-02 · fijado ANTES de escribir generador, entrenamiento o leer mundos F10-H.** Motivación: F10-G obtuvo 100% OOD en dos brazos con dos pérdidas auxiliares conjuntas, pero no separó su contribución y solo admitía una o dos firmas relevantes. F10-H es un estudio nuevo exploratorio, **no confirmación independiente** de la hipótesis tras conocer F10-G.

## Pregunta primaria

¿Qué señal adicional de TRAIN favorece que el actor aprenda a usar la **última versión acreditada** en historias con *al menos tres actualizaciones firmadas*, y convierta su valor en una acción nueva?

**Una sola arquitectura unaria**, igual en los cuatro brazos (exactamente `Actor(kind="unary",width=32)` de F10-G, mismo número de parámetros; misma inicialización y orden de ejemplos dentro de réplica). No reentrenar ni modificar F10-G. El actor ve cada versión de la clave y no recibe etiquetas del evaluador durante inferencia.

| Brazo | BCE acción final | CE índice último firmado público TRAIN | BCE acción con lectura del ganador aislada TRAIN |
|---|---:|---:|---:|
| `joint` | 1 | 0 | 0 |
| `selection_only` | 1 | 1 | 0 |
| `interpretation_only` | 1 | 0 | 1 |
| `both` | 1 | 1 | 1 |

Pérdidas con pesos unitarios sin hiperparámetros por brazo; la etiqueta del ganador surge de la regla **pública**, pero es SUPERVISIÓN EXPLÍCITA ADICIONAL en TRAIN, no descubrimiento autónomo ni entrada a inferencia. Las cuatro cabezas/scores existen y se calculan en todos los brazos; el régimen de pérdidas es la única intervención de TRAIN entre ellos. Control adicional `gru_only` sin lectura externa de parámetros dentro de ±5% del actor unario, como referencia (NO miembro del factorial).

## Nuevas distribuciones cerradas

- TRAIN: `19_100_000 + 10000*r+j`, 512 mundos/réplica; DEV: `19_200_000+`, 128 mundos/réplica; OOD: `19_400_000+`, 128 mundos/réplica. r=0,1,2,3 y j=0..N−1; cada mundo produce exactamente 4 contrafactuales finales; **2048 mundos TRAIN, 512 DEV y 512 OOD distintos** (8192, 2048, 2048 estados respectivamente, agrupados por mundo). **`19_300_000+` reservado CONFIRM, no existe ejecutor**. Nada de mundos F10-A..G ni checkpoints previos para entrenar.
- Un historial fijo por caso de **48 registros públicos de 29 features**, 24 claves posibles, consulta de 30 features al final y operación binaria pública de inversión. Timestamps únicos públicos, permutados **independientemente del orden de almacenamiento**; clave de consulta no expuesta durante la historia. Fuentes A/B aleatorias TRAIN y DEV. Selector unario recibe solo firma, tiempo y fuente (además de funciones de esas variables), nunca valores de registros ni etiqueta winner.
- La clave consultada tiene `k∈{7,9}` informes TRAIN/DEV; OOD `k∈{12,16}`. Entre sus informes hay siempre `s=3/4` versiones **firmadas previas** TRAIN/DEV y `s=5/7` OOD, seguidas de la confirmación `C` firmada y actualización contradictoria `X` con firma manipulada: **4–6 informes firmados en TRAIN/DEV y 6–9 en OOD** según variante. Esto garantiza **>=3 actualizaciones firmadas** incluso sin X firmada.
- Las versiones firmadas anteriores a C alternan los valores b/1−b y finalizan en 1−b; C=b. Así hay **al menos tres cambios de valor en la secuencia firmada**, y X puede actualizar a 1−b. Los restantes informes de la clave son no firmados de valores aleatorios, incluidos informes **posteriores** a C/X; otras 23 claves reciben registros distractores aleatorios. No hay prioridad de fuente programada.
- C/X son los dos últimos *candidatos firmables* temporalmente: sus dos timestamps provienen de una región posterior a todas las versiones firmadas previas, pero otros informes NO firmados pueden ocupar timestamps posteriores. Entre `U_before`,`U_after`,`S_before`,`S_after` solo cambian firma pública de X y asignación de los dos timestamps C/X; no se cambia su clave, fuente, valor ni el orden de almacenamiento; las cuatro variantes comparten consulta, clave, valor b, distractores y otros timestamps. La verdad es `b xor inversión` salvo `S_after`, que requiere `(1−b) xor inversión`.
- En OOD, cuatro regímenes de procedencia balanceados por `j%4`: `iid`, `signed_source_0` (credenciales firmadas source0, otras source1), `signed_source_1` (reverso), `alternating_storage_source` (source A/B según posición de almacenamiento). El bit de firma, no la fuente, sigue determinando validez; los cuatro modos se informan separadamente. Fuente no es oráculo.
- La asignación de k por `j%2` y de modo fuente `(j//2)%4` mantiene **16 mundos por combinación k×modo por réplica**. Las cuatro variantes por mundo usan el mismo historial salvo los dos campos públicos intervenidos. Las etiquetas de confirmación se calculan solo de informes públicos.

## Entrenamiento y comparabilidad

Cuatro réplicas y cinco brazos. Cada modelo recibe la **misma población TRAIN** de su réplica: 512 mundos×4 estados=2048 ejemplos; inicialización manual `torch.manual_seed(195000+r)` idéntica para cuatro brazos del factorial; 20 épocas, batch64, Adam .002, clip-grad1, **640 pasos y 40.960 presentaciones por brazo**; orden de minibatch determinista `196000+100*r+epoch`. Guardar checkpoint final de todos los brazos y su SHA **antes** de crear DEV/OOD en cada réplica. No búsqueda, reintento, early stop ni selección tras DEV/OOD. `gru_only` usa ancho del GRU de parámetros más cercanos y mismo protocolo, pero no comparte todas las formas tensoriales internas.
Parámetros idénticos entre los cuatro brazos; contar y publicar parámetros, pasos, ejemplos, versiones de librerías, semillas, duraciones y estimación de FLOPs. Pérdidas auxiliares incluyen etiquetas y computación adicional: **mismo número de pesos y pasos NO equivale a mismo coste ni información supervisada**.

## Evaluaciones fijadas

DEV/OOD: para los cinco brazos reportar precisión READ, SKIP (mismo historial H/GRU y lectura/available a cero), SHUFFLE cíclico de valores de candidatos conservando metadata/score, ERASE_SIGN y ERASE_TIME solo en F4 (no en historial recurrente), precisión con `oracle_patch` del valor ganador *solo observador* y NO contabilizarlo como operacional. Registrar predicción por fila, selección `argmax score` correcta, peso medio ganador, tabla conjunta selección/acierto, acierto de **las 4 variantes por mundo**, por k, por cuatro variantes, por régimen fuente y por cantidad de versiones firmadas. Comparadores públicos: último puesto de almacenamiento, máximo timestamp sin firma, aleatorio reproducible y último firmado (algoritmo exacto, no aprendido). Toda inferencia operacional READ es autónoma y sin uso de winner del evaluador.

Contrastes de 2000 remuestreos bootstrap por **mundo base**, manteniendo juntas sus cuatro variantes, RNG `197000+r`; por réplica reportar delta+IC95 `selection_only−joint` para selección, `interpretation_only−joint` para lectura con ganador (patch diagnóstico), `both−joint`, `both−best(single)` sobre READ OOD, y READ−SKIP/SHUFFLE para `both`. No tratar 2048 decisiones OOD como mundos independientes.

## Gates prospectivos inmutables

- **TECHNICAL_COMPLETE**: 4 réplicas×5 checkpoints con SHA verificado, identidad de estado/hash/etiqueta entre los 5 brazos, cuatro estados por mundo, >=4 versiones firmadas por caso, separación de TRAIN/DEV/OOD, todos los budgets exactos, ∣Δparámetros GRU∣<=5%, test de factorial, ninguna confirmación ni pesos antiguos.
- **G1 CAPACITY_TRANSFER**: para `both` en **>=3/4 réplicas**, OOD selección >=90%, precisión READ >=90%, >=85% en **cada una de cuatro variantes** y en **cada régimen fuente y ambos k**, al menos 80% de mundos con 4/4 correctos, IC inferior READ−SKIP >+0.20 e IC inferior READ−SHUFFLE >+0.10.
- **G2 SUPERVISION_CONTRIBUTIONS**: en **>=3/4 réplicas**, `selection_only` mejora selección vs `joint` por >=+0.10 **y** `interpretation_only` mejora `oracle_patch` vs `joint` por >=+0.10; ambos con IC95 mundo inferior >0. Una mejora de patch no cuenta como rendimiento operativo.
- **G3 COMPLEMENTARITY**: en **>=3/4 réplicas**, `both−joint` READ >=+0.10 e IC inferior >0, **y** `both−max(selection_only, interpretation_only)` READ >=+0.05 e IC inferior >0. Máximo de las dos precisiones se calcula en cada réplica como comparación predefinida, no selección de checkpoint. No se ajustarán los gates por efectos techo; si un brazo conjunto llega al techo, reportar el FAIL como en F10-G.

Status `F10H_FACTORIAL_SUPERVISION_PASS` si integridad y G1/G2/G3 PASS. De lo contrario `F10H_FACTORIAL_SUPERVISION_FAIL` mostrando subgates, aun si se observa 100% OOD. Estos criterios contestan hipótesis diferentes; no confundir utilidad de etiquetas auxiliares con descubrimiento autónomo de políticas o autenticación real.

## STOP y límites

Solo entornos binarios y hasta 9 versiones acreditadas por caso, una consulta final, un bit público XOR, clave explícita, firmas sintéticas, timestamps exactos y horizonte fijo de 48 informes; se necesitan nuevas familias de mundos antes de afirmar generalización. OOD es disjunto por semillas y distribución de k/procedencia pero **no es confirmación independiente a la hipótesis**, derivada de F10-G. Ante fallo científico congelar resultados y no añadir épocas, retocar pérdidas ni generar otro OOD de reemplazo. Mantener `19_300_000+` y anteriores confirmatorios sellados. PR borrador hasta revisar resultados; no merge automático.
