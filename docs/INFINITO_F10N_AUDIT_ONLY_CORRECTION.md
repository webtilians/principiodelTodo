# F10-N — incidencia técnica aislada de auditoría y corrección permitida

**2026-11-03.** La ejecución original [GitHub Actions #41714113859](https://github.com/webtilians/principiodelTodo/actions/runs/41714113859) terminó los **cuatro entrenamientos y evaluaciones** correctamente y preservó los cuatro paquetes originales de pruebas/entrenamiento por SHA. En el job estadístico independiente, después de auditar correctamente la réplica 0, hubo un **error de programación del auditor**:

La variable de entrada g0 contenía el manifiesto original de checkpoints F10-L. El código reutilizó accidentalmente el nombre g0 como booleano local del gate G0 de la primera réplica. Al entrar en la réplica 1, el auditor intentó indexar ese booleano como manifiesto, causando el TypeError "'bool' object is not subscriptable". **No se trató de un FAIL científico, ni se detectó modificación de checkpoints.**

**Corrección puramente técnica, sin cambio de criterio:** renombrar ese booleano local como g0_pass y usarlo en el campo G0 de salida, conservando el manifiesto g0 inalterado. Commit de corrección a0d1675891adac323d78c6a7825ad112acbad39d. Ninguna otra fórmula, bootstrap, umbral, semilla, modelo o dato se ha cambiado.

**Procedimiento de recuperación:** lanzar SOLO el auditor independiente desde un nuevo workflow que descarga exclusivamente los cuatro artefactos ORIGINALES de la ejecución #41714113859, más el manifiesto original F10-L del run #41685499939. No ejecutar NINGÚN entrenamiento, simulación, recolección, nueva partición DEV/OOD o confirmación. Verificar que código del mundo, memoria, experimento y prerregistro permanecen idénticos al commit de lanzamiento original 2ea33e63cab5af7b781c35a9c349224e30a7e7f9. Conservar el error original en el registro para trazabilidad. Cualquier criterio negativo de las cuatro réplicas seguirá siendo FAIL.

La partición de confirmación F10-N 26300000+ permanece SELLADA.