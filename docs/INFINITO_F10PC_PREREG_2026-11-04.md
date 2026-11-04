# F10-P-C — confirmación independiente con memorias PÚBLICAS congeladas

**Prerregistro prospectivo 2026-11-04: publicar ANTES de implementar confirmación, generar casos o consultar resultados de 28300000+.**

## Hipótesis y congelación

Confirmar F10-P con sus siete memorias originales SIN reentrenar: EventEffects, alimentado por las salidas públicas de las herramientas P/S/R/X, debe proponer SRX jamás ejecutada durante TRAIN, en contextos MEMORY+FLIP1 nuevos. El contrato de eventos, abstracción de trabajo, búsqueda de 85 programas, umbral de 10 observaciones y consistencia 90%, costes, primitivas F10-L, todas las memorias y controles se mantienen iguales. Este estudio es una confirmación en nuevas semillas de la MISMA familia de mundos sintéticos, no validación de herramientas reales.

Fuente original EXACTA: commit **9de6adb6ff0299b361d72258b9d988f832256201**, PR #71, run GitHub Actions **41888234405**. Descargar artefactos ORIGINALES f10p-original-public-events-0/1/2/3 y f10p-independent-public-event-science. Verificar las 7 memorias JSON por réplica y log TRAIN contra SHA de frozen.json. Los 20 checkpoints de habilidades F10-L del run ORIGINAL **41685499939** deben coincidir con su G0 original y frozen.json. Código F10-P y del ejecutor, selector y src/infinito3, INMUTABLE respecto a commit 9de6adb6ff0299b361d72258b9d988f832256201.

**PRE-FLIGHT MANDATORIO ANTES DE TOCAR 28300000+:** reconstruir decisiones y las CUATRO intervenciones de las 8960 filas DEV/OOD ORIGINALES por réplica (7 brazos por 768 DEV+512 OOD), exactamente las mismas respuestas, programas, SHAs, scores, contexto y primitivas. Recalcular G0: al menos 3/4 réplicas E100>=90% DEV en cada PUBLIC0/PUBLIC1/MEMORY0, contrastándolo con la auditoría original. Si falla algún SHA, fila o G0: INCONCLUSIVE, prohibido abrir nuevos mundos.

## Confirmación nueva (sellada hasta preflight)

Cuatro réplicas r=0..3, **128 mundos base por réplica** con 4 variantes U_before, U_after, S_before, S_after, total 512/r y 2048 por brazo. Semillas **28_300_000 + 10_000*r + j**, j=0..127, exclusivamente MEMORY+FLIP1, 9/11 informes, cuatro perfiles de fuente igual que OOD original. Obtener casos del mismo generador F10-P ORIGINAL sin cambios de código ni semántica: cambiar SOLAMENTE OOD_BASE en memoria del módulo de 28_400_000 a 28_300_000, llamar a dataset("ood",r), devolver constante al valor original. Emitir filas bajo split "confirm". Verificar SHAs de caso nuevos sin intersección con los de TRAIN/DEV/OOD originales, otras réplicas ni confirmaciones previas. No abrir otro conjunto de semillas, no repetir ensayo ni modificar modelos tras ver resultados.

Cargar exactamente snapshots originales E100/E50/E0/E_noX/E_shuffleX, B_reward e I_privileged y los cinco checkpoints F10-L originales de cada réplica. Cada brazo planifica con SOLO source, flip y public_bit, nunca verdad, F4 sin leer, índice de slot, contenido, traza ni recompensa de prueba. Ejecutar cada programa elegido con normal, shuffle, x_disabled, public_randomized sin replanificar; conservar SHAs, filas y original fitness. No mutar ficheros originales; no procesar datos de usuario, secretos, bóveda ni MCP de INFINITO.

## Gates CIENTÍFICOS sin alterar

Auditor independiente sin importar Torch, mundo ni learner, con **2000 remuestreos percentiles por mundo base** (128 grupos de 4 variantes por réplica), RNG **294000+100*r+offset**: 0 exactitud E100, 1 F4_shuffle, 2 X_disabled, 3 E100-B, 4 E100-E0, 5 E100-E_noX, 6 E100-E_shuffleX, 7 E100-E50. Todas las comparaciones pareadas sobre EL MISMO mundo y variante. Cada gate pasa globalmente si >=3/4 réplicas.

- **G0** previo: E100 original >=90% en CADA uno de los 3 contextos DEV, 3/4; reproducción exacta de 8960 filas y todos los SHA por réplica.
- **G1** E100 CONFIRM >=80%; IC95 inferior >70%; todos los estratos n, modo fuente, variante >=70%; SRX elegida >=80% (con cero ejecuciones TRAIN).
- **G2** E100 normal - shuffle_F4 y normal - X_disabled cada uno >=20pp y cada CI95 inferior >0.
- **G3** E100 - B_reward >=15pp, CI95 inferior >0.
- **G4** E100 - E0, E100 - E_noX y E100 - E_shuffleX cada uno >=15pp y CI95 inferior >0 (misma arquitectura de learner; entrenamiento original compartido, solo cambia feedback).
- **G5 diagnóstico no requerido** E50 CONFIRM >=80%.
- **G6 diagnóstico no requerido** I_privileged CONFIRM >=80%.

**PASS: F10PC_CONFIRM_PASS** si integridad/replay PASS y G0,G1,G2,G3,G4 >=3/4. **FAIL científico íntegro: F10PC_CONFIRM_FAIL** sin rescatar réplicas o ajustar hiperparámetros. Errores técnicos: F10PC_INCONCLUSIVE, jamás confundir el éxito CI con PASS.

## Evidencia y decisión de producto

Guardar manifiesto SHA de memorias y skill originales, reproducción de predicciones anteriores, casos y resultados confirmatorios, métricas de los siete brazos, estratos, programas, intervalos por mundo, gates, y auditoría de otro proceso (no auditor externo humano ni implementación distinta de física).

Un PASS confirma solo que una interfaz artificial de respuestas P/S/R/X permite una planificación composicional con tablas de efectos y abstracción/acciones dadas; NO demuestra inventar herramientas, aprender estados abiertos desde texto, transferir a otro generador o explotación segura en producción. Independientemente de PASS, **NO integrar en el núcleo ni fusionar automáticamente PR**. Próximo ensayo útil: operaciones reales contra entornos sintéticos en modo shadow/read-only; sin memoria personal, escrituras, olvido ni bóveda.