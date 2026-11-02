# INFINITO F10-I — recuperación técnica SIN reentrenar

**2026-11-02.** Primera ejecución [#41530837321](https://github.com/webtilians/principiodelTodo/actions/runs/41530837321): el TRAIN terminó en las cuatro réplicas, conservando **16 checkpoints originales** y sus `training.json` con SHA. DEV y OOD también llegaron a evaluarse con los modelos congelados; no pudo escribirse `rows.json` por un valor de metadatos `numpy.int64` no serializable. No fue fallo de entrenamiento o inferencia, sino de JSON después de terminar ambas evaluaciones. Por eso el workflow inicial marca FAILURE técnico y no hay auditoría final en ese run.

**ÚNICA corrección** sobre `observatory/f10i_experiment.py`: convertir escalares NumPy a escalares nativos **solamente al serializar las filas**. No cambia política, generador, recompensa, pesos, pérdidas, hiperparámetros, hashes de historia, métricas, ni entrenamiento. Arreglo commits `5d828e623ebe9d55089ea33479d4fd7f344346ae` y corrección sintáctica `fb990194d8e6567bb077d7434ecf3449c21043f7`. Ambos cambios son de salida JSON; ningún modelo se reentrena.

Este lanzamiento desencadena workflow **`.github/workflows/infinito-f10i-recover-frozen.yml`** que recupera directamente del primer run los **16 modelos originales SHA**, verifica que actor, generador, entrenamiento y recompensas son idénticos a commit `1350b6381616f3a6f9c9bc9e4a38927f31fd1497`, llama **solo** a `evaluate` (nunca a `train`) en exactamente los mismos splits DEV `20200000+` y OOD `20400000+` y guarda sus filas serializadas. Después, auditor externo aplica **sin ningún cambio** G1/G2/G3 del [prerregistro](./INFINITO_F10I_EXPERIENTIAL_LEARNING_PREREG.md).

El hecho de que los mundos DEV/OOD hayan sido abiertos durante la ejecución fallida se declara explícitamente; el arreglo no usa ni ha inspeccionado sus valores para cambiar el actor, tan solo el error de tipo observado en la salida. La confirmación `20300000+` no se abre. Tanto FAIL técnico original como el nuevo resultado científico se conservan permanentemente en la historia.

**STOP** después de esta recuperación: no otra ejecución de TRAIN, no tuning, no nueva confirmación.

## Segunda recuperación técnica, sin nuevas evaluaciones todavía

El primer intento de recuperación (run #41531538404) se detuvo **antes de descargar checkpoints y antes de ejecutar inferencia** porque el helper de JSON introdujo caracteres literales `\\n` en el código Python. Se corrigió exclusivamente esa sintaxis en commit `e6a77b1e2697fc3876fa8e8950d1a02971f95c75`. La comprobación sintáctica ahora tiene que pasar antes de descargar pesos; no cambia ningún dato, modelo ni función de entrenamiento. **Este commit del documento vuelve a ejecutar solo el recuperador congelado**, no el entrenamiento original.
