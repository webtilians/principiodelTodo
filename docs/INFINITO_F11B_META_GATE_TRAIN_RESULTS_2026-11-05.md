# INFINITO F11-B — resultado: selección aprendida entre política A3 y planificación A2

**Fecha 2026-11-05. Estado técnico SUCCESS, estado científico: TRAIN_HELDOUT_ONLY, NO se verifica la ventaja incremental de aprender el selector. No es PASS/FAIL confirmatorio de F11-A ni de F11-B.**

## Diseño preregistrado antes de código ni datos

- Documento prospectivo `docs/INFINITO_F11B_META_GATE_PREREG_2026-11-05.md`, commit **6255ef0518570f247f31a2877a503393935fba9d**, anterior al código F11-B.
- Se congelaron las comparaciones: meta-selector aprendido G vs A2 fijo, A3 fijo, regla simple H (cambiar brazo tras fallo), y el mismo selector sin historial G-nohist. Éxito prospectivo requería G superar H y ambos controles por >=5pp en >=3/4 réplicas y perder >=5pp al eliminar el historial.
- Nueva familia de semillas exclusivamente TRAIN: 62_100_000 + 10_000*r + i, por réplica r0..r3.
- i=0..255: ajustes de A2 y A3 en experiencias sintéticas aleatorias. i=256..383: ajuste del selector contextual con **un único brazo aleatorio realmente ejecutado por episodio**, feedback solamente observado, sin utilizar contrafactual, oráculo ni fase como entrada. i=384..511: mundos nuevos para evaluación TRAIN-heldout; todas las memorias de políticas, efectos y pesos del selector congelados antes de abrirlos.
- En mundos de meta-fit y heldout, una ruptura única entre los episodios 14 y 24, pseudorrandom por mundo, **invisible al selector**. Se retiene para auditar los resultados, nunca para decidir. 40 episodios por mundo. En heldout se actualizan únicamente las historias locales de resultados observados; no se actualizan pesos globales.
- Mismo entorno público de F11-A v3, las mismas acciones opacas A..F, horizonte máximo 12 y modelos A2/A3 guiados por la representación sintética de recursos. No LLM ni acceso a memoria de usuario.

## Reproducción y artefactos

GitHub Actions [run 42040426194](https://github.com/webtilians/principiodelTodo/actions/runs/42040426194), commit de lanzamiento **eab1ae3d62e558a1397ed06f482df42b2269ab0b**. Tests F11-B y contratos v3 SUCCESS; cuatro réplicas de entrenamiento+evaluación SUCCESS, cada una en su job. Artefactos `f11b-meta-gate-train-heldout-r0..r3`, retención 30 días, contienen eventos base, feedback META-FIT, pesos congelados, resultados por mundo y episodio de las cinco condiciones y resumen JSON.

Reproducir: `python -m unittest tests.test_f11b_meta_gate -v`; `python -m observatory.f11b_meta_gate --replica N --base-worlds 256 --meta-worlds 128 --heldout-worlds 128 --base-episodes 30 --sequence-episodes 40 --out PATH`.

En cada réplica: 7.680 episodios aleatorios para bases A2/A3, 5.120 decisiones observadas para entrenar el selector (un brazo por episodio), 5.120 evaluaciones por cada una de cinco condiciones; total evaluación por condición 20.480 episodios de 512 mundos, con episodios vinculados dentro de mundo y fases diferentes. Ningún mundo heldout intervino en ajuste de A2, A3 o pesos del gate.

## Resultado sobre 512 mundos nuevos dentro de TRAIN

| Control | Éxitos /20.480 | Tasa global | Reglas antiguas | Reglas cambiadas | Primeros cinco episodios tras cambio |
|---|---:|---:|---:|---:|---:|
| Siempre A2 | 10.825 | 52,86% | 0/9.653 (0%) | 10.825/10.827 (99,98%) | 2560/2560 (100%) |
| Siempre A3 | 12.307 | 60,09% | 1.896/9.653 (19,64%) | 10.411/10.827 (96,16%) | 2462/2560 (96,17%) |
| Regla H (cambio tras fallo) | **12.482** | **60,95%** | 1.668/9.653 (17,28%) | 10.814/10.827 (99,88%) | 2548/2560 (99,53%) |
| **Selector G aprendido** | **12.423** | **60,66%** | 1.815/9.653 (18,80%) | 10.608/10.827 (97,98%) | 2483/2560 (96,99%) |
| G sin historial de recompensas | 12.260 | 59,86% | 1.826/9.653 (18,92%) | 10.434/10.827 (96,37%) | 2465/2560 (96,29%) |

Diferenciales descriptivos G-H **−59/20480=−0,29pp**, G-A3 **+116/20480=+0,57pp** y G-G-nohist **+163/20480=+0,80pp**. Ninguno satisface los 5 puntos porcentuales prospectivos; el beneficio del historial es pequeño. No se calculó IC95 agrupado por mundo y no debe afirmarse significación estadística.

| Réplica | G aprendido | H simple | A3 fijo | A2 fijo | G sin historial |
|---|---:|---:|---:|---:|---:|
| r0 | 3173/5120 (61,97%) | 3158/5120 (61,68%) | 3101/5120 (60,57%) | 2722/5120 (53,16%) | 3083/5120 (60,21%) |
| r1 | 3334/5120 (65,12%) | 3300/5120 (64,45%) | 3351/5120 (65,45%) | 2659/5120 (51,93%) | 3327/5120 (64,98%) |
| r2 | 3140/5120 (61,33%) | 3210/5120 (62,70%) | 3076/5120 (60,08%) | 2722/5120 (53,16%) | 3073/5120 (60,02%) |
| r3 | 2776/5120 (54,22%) | 2814/5120 (54,96%) | 2779/5120 (54,28%) | 2722/5120 (53,16%) | 2777/5120 (54,24%) |

**Conclusión primaria preregistrada: CRITERIO NO SATISFECHO 0/4 réplicas.** Ni siquiera una réplica supera H, A2 y A3 por >=5pp. La ablación sin historial queda a menos de 5pp en todas.

## Qué se aprendió realmente y qué NO

La red del selector es una regresión logística contextual muy pequeña, con pesos actualizados por SGD en META-FIT usando el **éxito observado del brazo aleatorio elegido**, no resultados contrafactuales de un brazo no ejecutado. Las señales incluyen si hay una opción candidata, tasas y resultados recientes observados y uso por brazo. La fase y número de episodio no forman parte de la entrada; el breakpoint se varía entre mundos. Los pesos globales se congelaron antes de las pruebas. Se ejecutaron 5.120 actualizaciones por réplica; no fue entrenamiento de LLM.

G usa menos cambios de estrategia que H: **2.162 frente a 7.998**, pero sacrifica éxitos en la fase nueva. Eso puede ser un ahorro potencial de interrupciones, no una victoria de rendimiento ni coste demostrada.

Memoria serializada aproximada por réplica: A2 entre 1.708 y 1.806 bytes, opciones A3 entre 9.545 y 9.705 bytes, y gate entre 508 y 513 bytes. El conjunto A2+A3+gate ronda 11,8–12,0 kB. No se registró coste total de plataforma ni tokens; no comparar esta cifra directamente con el límite de coste total de F11-A. Pese a ello, el fuerte aumento de bytes merece cautela.

**LIMITACIONES:**
- El gate no fue evaluado con estructura de mundo distinta; TRAIN-heldout son nuevas semillas de la misma familia. No hay generalización composicional ni evidencia en entorno abierto.
- El selector aprende a decidir entre dos controladores preexistentes, no inventa acciones o políticas nuevas.
- Las historias de recompensa del episodio actual pueden revelar parcialmente efectos pero nunca fase interna. La comparación de G-nohist sirve como ablación, pero no aísla perfectamente todos los efectos de exploración estocástica.
- Se usaron episodios relacionados por mundo; no considerar 20.480 muestras estadísticamente independientes.
- Un control H sencillo iguala o supera al selector G en el agregado. No ajustar el gate retrospectivamente con este heldout; si se investiga otra hipótesis, usar protocolo y semillas nuevos.
- El modelo temporal A2 sigue usando esquema de recursos entregado por investigador, y A3 conserva políticas específicas de un sistema sintético.

## Decisión

**F11-B TRAIN_HELDOUT: NO se demuestra ventaja de aprender el gate. NO-GO para integrar este selector, abrir confirmación o entrenar un LLM para mejorar artificialmente esta comparación.** Mantener F11-A DEV/CONFIRM completamente sellados y conservar F11-B como resultado negativo reproducible.

Si el objetivo futuro es aprender cuándo replanificar, plantear experimento nuevo, no ajuste a estas semillas, donde cada decisión sea dentro de un mismo episodio ante evidencia real de anomalía, con control heurístico competitivo, costes de búsqueda y un evaluador ciego. No se debe reinterpretar este NO-GO como fracaso general de todas las formas de metacontrol.

No se modificó el producto, servicios, secretos, memoria personal ni se habilitaron escrituras externas.
