# INFINITO F9-S — Confirmación de parches causales de subespacios H/C

**30 de octubre de 2026. Estado: completado. Hipótesis primarias A y B NO CONFIRMADAS.**

- [GitHub Actions #41052305804](https://github.com/webtilians/principiodelTodo/actions/runs/41052305804): **SUCCESS, 11/11 jobs** — tests, entrenamiento de ejes 890000+, gate de modelos/proyecciones, cuatro confirmaciones nuevas 895000+ y agregado.
- [Protocolo preregistrado](./INFINITO_F9S_CUE_SUBSPACE_PROTOCOL.md), publicado **antes de leer los resultados**.
- [JSON agregado y desglose por semilla](./INFINITO_F9S_RESULTS_2026-10-30.json); [artifact agregado #12976130441](https://github.com/webtilians/principiodelTodo/actions/runs/41052305804/artifacts/12976130441), más artifacts de cada mundo y tratamiento por semilla.
- [PR #27](https://github.com/webtilians/principiodelTodo/pull/27), en borrador apilada sobre F9-R #25; sin merge.

## Pregunta central para INFINITO

F9-Q demostró que, con la historia apropiada, **intercambiar H/C completo** puede cambiar causalmente las elecciones hacia la pista donante aun si F4 está anulada en el momento de decidir. F9-R comprobó que congelar H/C durante diez pasos sin F4 no basta para preservar esa capacidad y que restaurar H/C natural **al final** del apagón rescata algunas decisiones.

F9-S pregunta si podemos reducir el mecanismo a **un coeficiente cue por componente** (un eje H y un eje C), distinguiéndolo de alteraciones inespecíficas de la dinámica de control. El aprendizaje de una dirección lineal desde el contraste key/ball **no** implica que la política utilice esa dirección para decidir: tal es precisamente la hipótesis sometida a prueba.

## Integridad y particiones

- **Doce políticas congeladas:** BC F9-O \`continuous6_after_bc\`, PPO sham y PPO blackout F9-P, cuatro semillas; checkpoints originales verificados por SHA-256, sin actualizaciones a política, LSTM ni recompensa.
- Solo se ajustaron **dos direcciones 1D cue por política y semilla**, una para H (64) y otra para C (64), a partir del promedio \`ball−key\` de estados naturales al último paso pre-fork en **80 pares S13 TRAIN nuevos por semilla**: \`890000+index*1000+j\`. Total 320 pares TRAIN. Ejes ortogonales deterministas sin ajuste a acciones/recompensas, norma 1 y ortogonalidad verificada.
- La confirmación independiente empleó \`895000+index*1000+j\`, j=0..39, cuatro semillas → **160 pares S13 por política y condición**, distinta de TRAIN y de todas las particiones F9-M/N/O/P/Q/R y del holdout histórico 900000+ **SIN ABRIR**.
- Dos historias con la pista invertida sobre geometría idéntica, prefijo físicamente conducido por profesor hasta la bifurcación; durante este prefijo el LSTM procesa sus observaciones, pero sus acciones predichas no mueven el entorno. Tras el fork, continuación **autónoma**.
- Las 148 observaciones públicas de la decisión son idénticas entre key y ball, **F4=0 en el fork y durante toda continuación** para todos los brazos; durante los diez pasos previos los brazos blackout comparten máscara F4, misma observación pública y número de frames. El parche se introduce **una vez** justo antes de procesar el último frame sin F4, sin alterar geometría/reward/política.
- Cada condición registra primera acción del actor, primer brazo físico, aciertos con cue propio, traslado hacia cue donante bajo \`swap_HC\`, exclusiones cero, y comparaciones por cuatro semillas. Es una **prueba de decisión desde prefijo controlado**, no navegación end-to-end.

## Resultado preregistrado A — ¿el código cue es suficiente para rescatar una política tras diez pasos sin F4?

| Pares con ambas primeras salidas correctas /160 | BC | PPO sham | PPO blackout |
|---|---:|---:|---:|
| Historia F4 normal, fork F4=0 | **112** | **102** | 44 |
| Blackout, actualizaciones H/C sin parche | 0 | 0 | 0 |
| **Parche 1D cue propio en H+C sobre estado blackout** | **0** | **0** | **0** |
| Parche 1D cue **opuesto** sobre estado blackout | 0 | 2 | 0 |
| Perturbación ortogonal igual L2 sobre estado blackout | 0 | 0 | 0 |
| **Restauración H/C natural completa en último paso** | 29 | 47 | 44 |

**Criterio A: NO SUPERADO en los tres grupos.** Los deltas del parche cue propio frente a blackout sin parche son **[0,0,0,0]** para la métrica de dos primeras salidas y también para transferencia al donante con trasplante H/C; cero de cuatro semillas positivas y ninguna ventaja frente al control ortogonal. No se ha demostrado que el eje cue 1D baste para **rehabilitar** el controlador deteriorado.

La restauración **completa** recupera una parte: 29/160, 47/160 y 44/160; estos resultados de F9-S son sobre mundos **nuevos** y no deben confundirse con las cifras históricas de F9-R (34,49,40).

## Resultado preregistrado B — ¿cambiar solo el eje cue invierte la decisión en un controlador temporalmente funcional?

Partir del estado natural H/C restaurado en el último frame; sustituir solo los coeficientes cue de H y C por los de la **pista opuesta**, y comparar el resultado contra dirección ortogonal de **igual norma L2**. Una verdadera transferencia cue-específica completa requiere:
- estado basal toma las **dos salidas propias correctas**;
- eje invertido toma las **dos salidas de la pista donante**, es decir los dos brazos físicamente opuestos, no solamente una variación de acción;
- ventaja por semilla y agregado frente a control ortogonal.

| Pares con ambas primeras salidas correctas /160 | BC | PPO sham | PPO blackout |
|---|---:|---:|---:|
| Basal H/C natural completo en último paso | 29 | 47 | 44 |
| **Invertir solo eje cue H+C en estado natural** | 0 | 14 | 21 |
| Alterar eje **ortogonal** con igual L2 | 50 | 56 | 37 |
| Clamp de coeficiente cue al promedio key/ball | 11 | 47 | 29 |
| Invertir solo eje cue H | 41 | 61 | 38 |
| Invertir solo eje cue C | 0 | 23 | 30 |

**No confundir** la fila «ambas salidas correctas tras invertir el eje» con «transferencia contrafactual a la identidad opuesta»; son métricas diferentes. La métrica causal específica es la siguiente:

| Transferencia completa de la elección hacia pista donante, comparada con basal | BC | PPO sham | PPO blackout |
|---|---:|---:|---:|
| **Eje cue H+C, ambos brazos** | **0/160** | **0/160** | **6/160** |
| Eje ortogonal, misma norma, ambos brazos | 0/160 | 0/160 | 0/160 |
| **Semillas con inversión cue específica** | 0/4 | 0/4 | **1/4** |

**Criterio B: NO SUPERADO**. El único caso parcial —6/40 pares— aparece **solo en la semilla 3 de PPO blackout**. La comparación preregistrada exige efecto en ≥3/4 semillas y un total superior al placebo. Aunque ese control ortogonal no indujo transferencia específica al donante, **sí alteró sustancialmente los aciertos propios** (por ejemplo, 50/160 BC y 56/160 PPO sham); por tanto, algunas mejoras y pérdidas de decisión pueden deberse a perturbaciones inespecíficas del controlador. No interpretar una subida de aciertos como prueba de «código de memoria» si no transfiere la elección entre pistas.

### Hallazgos secundarios H vs C

La inversión del eje de H aislado produjo 41, 61, 38 pares con ambas elecciones propias correctas en BC/sham/blackout, mientras invertir solo C produjo 0, 23, 30. Pero **no hubo transferencia donante robusta por 4 semillas**; el único evento secundario de inversión C fue 6 pares en la misma semilla PPO blackout que muestra el eje conjunto. El resultado **no permite identificar** un único componente H o C como el «recuerdo causal».

## Consistencia entre semillas

| Política | Basal H/C natural completo al último paso (correctos por semilla /40) | Parche cue propio sobre blackout | Inversión causal hacia donante con eje cue H+C |
|---|---|---|---|
| BC | [21,8,0,0] | [0,0,0,0] | [0,0,0,0] |
| PPO sham | [0,20,27,0] | [0,0,0,0] | [0,0,0,0] |
| PPO blackout | [0,0,6,38] | [0,0,0,0] | [0,0,0,6] |

Los 160 mundos no son 160 réplicas de entrenamiento; hay **cuatro** políticas replicadas por grupo. Las conclusiones no deben basarse en escoger la semilla de mejor resultado.

## Interpretación mecanicista para INFINITO

1. F9-Q sigue proporcionando evidencia de que **el estado recurrente completo** transporta información histórica y, en condiciones concretas, puede causar la decisión adecuada sin F4 actual.
2. F9-R y F9-S muestran que **el estado temporal del controlador importa**: ni congelar H/C ni restaurar solo un par de coordenadas lineales cue 1D rescatan la elección tras diez pasos sin lectura F4, mientras restaurar el **estado completo natural** justo antes de la bifurcación puede hacerlo.
3. La señal de identidad puede estar **distribuida**, implicar interacciones no lineales H/C, o ser decodificable sin un lector accionable después del blackout. El diseño demuestra que este estimador lineal particular **no es suficiente**, no que todo subespacio sea inútil.
4. Perturbar una sola dirección ortogonal también altera el comportamiento: un cambio de acción no demuestra causalidad **específica de la pista**. El éxito exige cambio de salida hacia la pista donante y replicación entre semillas.
5. No hay evidencia de conciencia, acceso deliberado a recuerdos ni memoria general; la llegada al fork fue forzada por el prefijo del profesor.

**Conclusión:** ambos criterios principales F9-S **fallan**. Una dirección cue de rango 1 aprendida por diferencia promedio no recupera la memoria funcional ni transfiere la identidad del donante de manera consistente. El resultado **acota** el mecanismo causal hacia una representación/estado conjunto, temporalmente adecuado, con efectos de lectura y de control que todavía requieren separación.

**No seleccionar otro eje con \`895000+\` para repetir la confirmación**. Esa partición queda **consumida**. El holdout histórico \`900000+\` permanece sin abrir. Ante un siguiente experimento, será necesario diseñar un nuevo conjunto verdaderamente independiente; no ampliar unilateralmente rango del holdout ni recurrir a ajuste post-hoc. La PR permanece en borrador y sin fusionar.
