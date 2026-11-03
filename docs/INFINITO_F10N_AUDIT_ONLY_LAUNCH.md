# F10-N — recuperación técnica de auditoría, NO nueva ejecución experimental

Este archivo activa UNA VEZ el workflow independiente **infinito-f10n-frozen-audit-only.yml**. El prerregistro científico ORIGINAL no ha cambiado.

**RUN ORIGINAL INMUTABLE:** [GitHub Actions #41714113859](https://github.com/webtilians/principiodelTodo/actions/runs/41714113859). Los cuatro entrenamientos y evaluaciones finalizaron SUCCESS y produjeron los artefactos SHA congelados f10n-original-experience-0..3. El fallo técnico posterior del auditor fue una variable g0 del manifiesto que se sobrescribió con un booleano. Se corrigió ÚNICAMENTE el nombre local g0_pass, commit a0d1675891adac323d78c6a7825ad112acbad39d. El detalle figura en [INFINITO_F10N_AUDIT_ONLY_CORRECTION.md](INFINITO_F10N_AUDIT_ONLY_CORRECTION.md).

El workflow descarga los cuatro artefactos originales del run **41714113859**, los verifica por SHA y reconstruye el historial y efectos observados, y descarga G0 del run original F10-L **41685499939**. NO invoca módulos de entrenamiento, generador de mundos ni evaluador de políticas, NO usa PyTorch y NO cambia criterios G0–G4. El código original de mundos, experimento, memoria y prerregistro se compara contra el commit de lanzamiento F10-N **2ea33e63cab5af7b781c35a9c349224e30a7e7f9**.

Solo se debe reportar el resultado científico desde estos mismos registros congelados. F10-N CONFIRM **26300000+** no abierta. Mantener FAIL si los gates originales fallan.