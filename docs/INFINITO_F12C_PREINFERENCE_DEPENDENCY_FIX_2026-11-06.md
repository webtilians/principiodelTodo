# F12-C — Corrección técnica de dependencia anterior a toda inferencia

**Fecha 2026-11-06. Sin datos F12-C consumidos.** La primera ejecución del workflow, [run #42150788888](https://github.com/webtilians/principiodelTodo/actions/runs/42150788888), pasó los contratos técnicos pero el job de inferencia se detuvo durante `pip install`: el índice de ruedas CPU no ofrecía `llama-cpp-python==0.3.16` para la plataforma concreta (pip: `No matching distribution found`). El paso de descarga de modelo y el de pruebas L0-L5 fueron **SKIPPED**. Por tanto el espacio de casos sigue sin observarse ni ajustarse.

**Corrección previa a cualquier resultado:** actualizar la versión de bindings CPU a `llama-cpp-python==0.3.36`, publicada en el índice consultado y con wheel de CPU documentada. Mantener `--only-binary llama-cpp-python` para no compilar una versión no verificada. Todas las demás restricciones permanecen **idénticas**: Qwen2.5 3B GGUF Q5_K_M de mismo repositorio/archivo, hash de pesos, ChatML, temperatura 0, seis casos L0-L5, 6 acciones, criterios G0-G4, presupuestos y no memoria. Esta corrección de infraestructura **no** selecciona un comportamiento favorable.

**No reinterpretar el error de pip como NO-GO científico del modelo**; corresponde a BLOCKED antes de su ejecución. Si el nuevo wheel no existe, conservar el bloqueo sin cambiar semántica de pruebas.
