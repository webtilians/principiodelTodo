# INFINITO F11-C — Resultado: interrupción aprendida dentro de un episodio

**Fecha: 2026-11-05. Contratos y cuatro réplicas: SUCCESS. Conclusión prospectiva: NO se demostró aprendizaje útil de interrupción, NO-GO.** La verificación pertenece exclusivamente a TRAIN-heldout, no a DEV ni CONFIRM científico. El resultado negativo se publica íntegro.

## Procedencia y garantías experimentales

- Hipótesis y controles registrados **antes del código**: `docs/INFINITO_F11C_INTRAPOLICY_INTERRUPT_PREREG_2026-11-05.md`, commit **7e304224b5db74b514c547c1f406edad2ad61148**.
- Código y tests: `observatory/f11c_interrupt.py`, `tests/test_f11c_interrupt.py`; workflow `.github/workflows/infinito-f11c-interrupt.yml`.
- **[GitHub Actions 42049177686](https://github.com/webtilians/principiodelTodo/actions/runs/42049177686)**: contratos técnicos y 4/4 réplicas TRAIN SUCCESS, commit de lanzamiento **59de25189251425a5fc60c0acdbd1549418d4ccb**.
- Primer run **42049086044** FAIL técnico: `LearnedInterrupt.snapshot()` devolvía una referencia mutable a los pesos e invalidaba una prueba de inmutabilidad del checkpoint. Se corrigió copiando la lista **antes de comenzar cualquier ejecución TRAIN**; no se utilizó información del conjunto heldout para corregir.
- Cuatro artefactos `f11c-train-heldout-r0..r3`, con eventos base y detector, modelo congelado, `heldout_per_world.jsonl`, `summary.json` y hashes; retención temporal de GitHub Actions.
- Reproducir: `python -m unittest tests.test_f11c_interrupt -v`, luego `python -m observatory.f11c_interrupt --replica N --base-worlds 256 --gate-worlds 128 --heldout-worlds 128 --base-episodes 30 --gate-episodes 30 --out DIR`.

## Contrato de observación y entrenamiento

Cuatro réplicas, semillas nuevas F11-C `63_100_000 + 10_000*r + i`, nunca reutilizadas de F11-A ni F11-B:

- 256 mundos/r para A2 y A3, exploración aleatoria neutral, 30 episodios/mundo; **30.720 episodios base**.
- 128 mundos/r para entrenar el detector L mediante una elección **continuar/interrumpir en cada paso** y REINFORCE con recompensa terminal del episodio **realmente ejecutado**, nunca del brazo contrafactual: **15.360 episodios de gate**.
- 128 mundos/r excluidos de ambos entrenamientos, escenarios *sin cambio* y *cambio oculto de reglas entre pasos 3–7* durante el episodio, ambos con los mismos controles: 512 mundos nuevos, 1024 episodios/arm.
- El detector usa únicamente observación pública, desviación respecto al estado de una opción extraída de experiencias exitosas, recompensa observada anterior y progreso; no recibe la fase verdadera, identificador de semilla, `world.roles`, cola de efectos ni etiquetas de anomalía. Los resultados del simulador se registran por separado para auditar.
- Se comparan P=A2 siempre; C=continuar opción A3 y hacer fallback al acabarse; G=guarda exacta fija programada; H=heurística de diferencia sustancial de recursos/ruta; L=interrupción aprendida; L-no-sorpresa=ablación; A4=referencia privilegiada. Ninguna prueba abre las semillas reservadas DEV/CONFIRM de F11-A.

## Resultados heldout: 512 mundos por escenario

| Control | Sin cambio | Cambio de reglas intraepisodio |
|---|---:|---:|
| P: siempre planificar A2 | 0/512 (0%) | 194/512 (37,89%) |
| C: continuar opción A3 | 115/512 (22,46%) | **464/512 (90,63%)** |
| G: guarda exacta fija | 108/512 (21,09%) | 385/512 (75,20%) |
| H: regla simple recursos/ruta | 109/512 (21,29%) | 390/512 (76,17%) |
| **L: interrupción aprendida** | **115/512 (22,46%)** | **464/512 (90,63%)** |
| L sin sorpresa | 115/512 (22,46%) | 464/512 (90,63%) |
| A4 heurístico privilegiado | 512/512 (100%) | 512/512 (100%) |

Cada réplica obtuvo éxito sobre escenario *flip* para L y C idéntico: r0 105/128; r1 118/128; r2 120/128; r3 121/128. G tuvo 94,94,96,101; H tuvo 90,97,99,104, respectivamente.

**Evidencia adversa inequívoca:** el modelo REINFORCE aprendió pesos negativos para la acción de interrumpir, incluso en la característica de discrepancia. El detector L **interrumpió 0 veces en las 1024 evaluaciones**, y L-no-sorpresa también **0 veces**. Ambas variantes son exactamente iguales a C en éxitos, retornos, longitud y acciones ejecutadas. Se registraron 5066 consultas de L a su detector por condiciones (2535 sin cambio + 2531 con cambio), pero no se activó un solo replanteamiento. El control G interrumpió 403/512 veces sin cambio y 407/512 en flip; H 402/512 y 404/512.

### Interpretación central y sesgos estructurales

1. **Criterio prospectivo L > G y P por >=5pp en flip, >=3/4 réplicas:** L sí supera estos dos controles, pero NO se debe declarar ventaja de inteligencia porque **L y C son idénticos**, y L no interrumpe nunca; el mecanismo aprendido no añade valor causal a continuar.
2. **Criterio L > L-no-sorpresa >=5pp:** **0 puntos en todas las réplicas. NO SATISFECHO.** Las características de sorpresa no se utilizaron para cambiar la conducta final.
3. **Problema de diseño más profundo:** en el simulador usado, pasar a las reglas nuevas durante una secuencia suele **beneficiar** a la opción previamente almacenada; cambiar las reglas no equivale necesariamente a contradecir las expectativas *de manera perjudicial*. El C que nunca aborta obtiene 464/512 con flip pero solo 115/512 sin flip. El experimento quedó sesgado hacia la estrategia de no interrumpir y carece de un conjunto consistente de cambios que deterioren la política; así no puede separar capacidad de detección de una política de continuar óptima para este mundo.
4. A3 usa opciones exactas obtenidas con un colector neutral y guardas de estados pasados; no demuestra abstracción de política entre tareas. A2 se basa en un planificador guiado por el esquema y falla por completo en el escenario estable, por lo que **es un control limitado** y no demuestra ganancia de planificación general.
5. El control A4 es una heurística con acceso privado a roles/fase y un techo descriptivo, no una prueba de política óptima. Un cambio situado en el paso 3..7 puede ocurrir **después** de completar el episodio; en 440/512 probes C llegó efectivamente al paso de cambio (resto antes de completar), e incluso con cambio alcanzado el efecto no necesariamente exige abandonar una política.
6. **No hay una ventaja de coste:** la memoria serializada aproximada del conjunto A2+opciones+detector fue 11,5–11,9 kB/r frente a 1,7–1,8 kB/r del A2 aislado (≈6,3–6,7× de bytes); esto no equivale aún al coste total de ejecución. L añade consultas que el C sin detector no necesita. No se demostró cumplir 1,25× coste total.
7. Los episodios y pasos están correlacionados por mundo; no se calcularon IC95 de confirmación, ni se abrió ningún dataset CONFIRM, ni hay generalización de nuevas estructuras o LLM real.

## Decisión y dirección futura

**F11-C TRAIN-heldout: NO-GO. El mecanismo aprendió 'continuar siempre', NO aprendió cuándo debe interrumpir.** No integrar ni alegar aprendizaje autónomo de detección de contradicción. Conservar el resultado negativo, las cuatro réplicas y la corrección técnica anterior a TRAIN.

La siguiente pregunta causal, separada y con protocolo prospectivo nuevo si se desea, es construir cambios *benignos y perjudiciales* equilibrados que generen evidencia pública de error de predicción, con coste de continuación vs planificación y pares contrafactuales **solo en auditoría** para medir si el detector puede distinguirlos sin ver una etiqueta de verdad. Mantener un baseline simple de guarda, un baseline de continuar y A2 competente; no cambiar umbrales después de resultados.

No se tocó `src/infinito3`, producción, memoria de usuario, secretos, herramientas externas ni LLM. PR de investigación apilado sobre F11-A v3, nunca auto-merge.
