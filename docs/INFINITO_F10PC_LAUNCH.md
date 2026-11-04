# F10-P-C — LANZAMIENTO ÚNICO DEL ESTUDIO CONFIRMATORIO

**Fecha 2026-11-04.** El protocolo prospectivo de F10-P-C se publicó ANTES de implementar o generar datos nuevos: commit **2c1edcfcf8eac2bd5b8e5904216ddfe2c360b9f5**, documento [INFINITO_F10PC_PREREG_2026-11-04.md](INFINITO_F10PC_PREREG_2026-11-04.md). Fuentes originales: F10-P run **41888234405** commit **9de6adb6ff0299b361d72258b9d988f832256201**, F10-L run **41685499939**. Ambas fuentes y modelos originales quedan congelados.

Este archivo ejecuta UNA VEZ el workflow infinito-f10pc-original-frozen-confirm.yml en cuatro réplicas. Primero se verifica bit por bit el SHA y se reconstruyen 8960 filas originales por réplica × 4 intervenciones. **Si falla, no se abre ninguna semilla nueva**.

Solo tras ese PASS, se generan 128 mundos/r×4 variantes nuevos del área 28300000+, con el mismo generador sintético, usando EXACTAMENTE los siete modelos/contadores de memorias F10-P originales (sin reentrenamiento). El auditor independiente calcula los mismos G0..G4, 2k bootstrap por mundo, y los controles diagnósticos E50/I. Reportar PASS, FAIL o INCONCLUSIVE con sus causas. No ajustar, no repetir corrida, no tocar master, producción, bóveda, memoria personal ni MCP.


## Corrección técnica previa a cualquier mundo confirmatorio (sin cambiar criterio)

El primer run 41895819923 concluyó FAILURE técnico en las cuatro réplicas antes de generar ningún caso 28300000+: una igualdad de diccionarios Python consideraba distintas las tuplas XOR `(1,0)` y listas JSON `[1,0]`, aunque contengan los mismos valores. Se corrigió ÚNICAMENTE la comparación de metadatos de habilidades originales mediante normalización JSON en el commit 2bd8d5d6c8ac4538539f06a9419d698ab0d8b376, manteniendo exigencia de identidad exacta después de normalizar. La ejecución no pasó la frontera de preflight, no abrió confirmación ni produjo métricas científicas. Esta edición lanza la repetición técnica de la primera ejecución inconclusa, sin entrenar o cambiar memorias originales, generador, gates ni semillas. Si la preflight descubre una discrepancia REAL se abortará.
