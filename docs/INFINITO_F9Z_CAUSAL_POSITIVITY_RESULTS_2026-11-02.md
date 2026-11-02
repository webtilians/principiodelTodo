# INFINITO F9-Z-A — Resultado del piloto de soporte causal factorial

**2026-11-02 — resultado definitivo del piloto, no predictor ni confirmación.**

Protocolo inmutable [`INFINITO_F9Z_CAUSAL_POSITIVITY_PREREG.md`](./INFINITO_F9Z_CAUSAL_POSITIVITY_PREREG.md), creado **antes** de implementar y evaluar las nuevas semillas, commit `bc19d5fe15fd0679a5dbcdf55c161c17f8b18f3e`.

- [Ejecución piloto original #41370406853](https://github.com/webtilians/principiodelTodo/actions/runs/41370406853): las cuatro réplicas de `frozen-factorial-pilot` y sus artifacts concluyeron **SUCCESS**; el job `scientific-support-report` falló por `ModuleNotFoundError: numpy` en un runner sin dependencias, sin afectar al piloto ni a sus archivos.
- [Recuperación SOLO DEL INFORME #41370623152](https://github.com/webtilians/principiodelTodo/actions/runs/41370623152): descarga los cuatro artifacts originales, ejecuta agregador independiente **Python stdlib** y audita de nuevo cada conteo. Terminó **SUCCESS** sin abrir mundos, sin descargar actores ni importar Torch. **No se trata de segunda replicación**.

## Diseño y controles comprobados

Cuatro actores F9-P de pesos originales congelados. **96 mundos nuevos `11_100_000+`** (24 por actor) × 2 cues (key/ball) × 2 historias (normal/postcue_off) × 3 capacidades reales SelectiveF4(`context`, 1/4/8) × 3 ventanas READ de 1/4/8 pasos = **3.456 casos** y **6.912 continuaciones causalmente emparejadas READ_W/SKIP_W**. No se eligió ningún mundo o configuración en función de D; todos los resultados neutrales y negativos permanecen. La referencia `N=8,W=1` reproduce exactamente las dos ramas originales F9-V sobre un mundo nuevo por réplica. Todos los SKIP son idénticos entre las tres ventanas; actor y F4 siguen sin entrenamiento ni cambios en el selector.

`D_{N,W}=correct(first_arm|READ_W)-correct(first_arm|SKIP)`; es una nueva intervención cuando W>1, no comparable sin matiz al READ único de F9-Y. Gate preregistrado: **un solo tratamiento común** `N,W` deberá proporcionar **>=3 D+ y >=3 D<=0 en cada una de las 16 celdas actor×cue×historia**, sobre los 24 mundos completos por celda. Si no, **STOP**.

## Resultado global (cada brazo tiene 384 decisiones = 24 mundos × 4 actores × 4 cues/historias)

| Capacidad | Pasos READ | D+ | D=0 | D− | Celdas con >=3 positivos y 3 no positivos (/16) | Celdas con al menos un D+ (/16) |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 1 | 41 | 322 | 21 | **4** | 6 |
| 8 | 4 | 41 | 321 | 22 | **4** | 6 |
| 8 | 8 | 41 | 321 | 22 | **4** | 6 |
| 4 | 1 | 51 | 299 | 34 | **6** | 8 |
| 4 | 4 | 60 | 275 | 49 | **7** | 8 |
| 4 | 8 | 60 | 275 | 49 | **7** | 8 |
| 1 | 1 | 5 | 379 | 0 | **1** | 1 |
| 1 | 4 | 5 | 379 | 0 | **1** | 1 |
| 1 | 8 | 5 | 379 | 0 | **1** | 1 |

**Preregistro: `SUPPORT_NOT_ACHIEVED`. Ningún brazo común obtiene positividad en 16/16 celdas.** Tampoco el brazo exploratoriamente mejor (C4,W4/W8), que llega a 7/16. No se selecciona un tratamiento distinto para cada actor o historia para fabricar un PASS.

### Comparación descriptiva de patrones de D+ por celda

Orden de cifras en cada réplica: `key-normal / ball-normal / key-blackout / ball-blackout`, de 24 mundos por condición:

| Réplica | N8, W1 (referencia) | N4, W4 (máxima cobertura exploratoria) |
|---|---|---|
| 0 | 11 / 9 / 0 / 0 | **0 / 13 / 0 / 0** |
| 1 | 0 / 0 / 0 / 8 | **6 / 12 / 3 / 11** |
| 2 | 0 / 2 / 1 / 0 | **0 / 2 / 10 / 0** |
| 3 | 10 / 0 / 0 / 0 | **0 / 3 / 0 / 0** |

La capacidad 4 y ventana 4 aumentaron el número de positivos en algunas celdas y también el número de lecturas perjudiciales; en otras, por ejemplo `key-normal` de las réplicas 0 y 3, eliminó los positivos observados con C8, W1. Por tanto la política del actor interactúa fuertemente con el contenido y duración de F4. Las diferencias W4→W8 no aportan cobertura adicional en este piloto, pero no prueban que ventanas largas sean inútiles en otros actores o tareas.

## Conclusión científica y STOP

1. El requisito de factibilidad **no se cumple** para ninguna de las nueve condiciones predefinidas. Es un resultado negativo específico a 4 actores congelados, la tarea S13, las 24 semillas por actor y los factores probados; no demuestra imposibilidad absoluta de aprender valor de memoria.
2. Variar capacidad y ventana altera las decisiones y puede crear D+ adicionales, pero **no garantiza positividad dentro de cada celda de forma compartida**. Por ello no procede reentrenar una nueva cabeza sobre estos mundos ni abrir TRAIN `11200000+`, DEV `11300000+` o CONFIRM `11400000+`.
3. La hipótesis de F9-Y —representar y transferir D+ más allá de pista/historia— sigue sin estar demostrada. El nuevo cuello de botella apunta al **comportamiento del actor y al diseño de la tarea**, no solo al tamaño de la cabeza.
4. Siguiente propuesta separada, NO ejecutada: tarea prospectiva con diversidad contrafactual y entrenamiento del actor para utilizar F4 en **cada** combinación, evaluación no condicionada a aciertos de entrenamiento, y pilotaje de positividad ANTES del entrenamiento de cabezas. Si se cambia el actor, se debe declarar nueva generación de checkpoints, nuevos controles, nuevas semillas y dominio; no presentarlo como confirmación de F9-V/Y.

**Todos los conjuntos históricos y confirmatorios permanecen sin reabrir**: F9-V `8300000+` (consumido y no reutilizable), F9-W `9300000+`, F9-X `9600000+`, F9-Y `10300000+`, F9-T `6300000+`, F9-U `7300000+`, F9 histórico `900000+`.

Los JSON completos de 3.456 casos y el informe agregado auditable se conservan en los artifacts de Actions vinculados.
