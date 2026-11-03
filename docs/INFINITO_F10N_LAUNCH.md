# F10-N — lanzamiento único de memorias explícitas y transferencia de efectos

**2026-11-03**. El prerregistro prospectivo original, publicado **ANTES de escribir ningún código F10-N o abrir datos**, es [INFINITO_F10N_EXPERIENCE_EFFECT_MEMORY_PREREG_2026-11-03.md](INFINITO_F10N_EXPERIENCE_EFFECT_MEMORY_PREREG_2026-11-03.md), commit **2992972edaa8c955b6cc4e623acab6d39d1cd0d0**.

Este archivo desencadena una sola ejecución de **.github/workflows/infinito-f10n-explicit-experience-memory.yml**. Se mantendrán los gates originales G0–G4 sin modificación tras resultados.

- Cuatro réplicas y cuatro condiciones: A recompensa por programa/contexto; B extrapolación de recompensas por contexto; C modelo de efectos aprendido de transiciones observadas usando exactamente los mismos 98.304 intentos que A/B; D modelo de efectos con intentos dirigidos por incertidumbre, mismo presupuesto pero decisiones experimentales distintas.
- Los **84 programas TRAIN** excluyen deliberadamente **SRX**; la síntesis inédita de SRX solo puede ocurrir con las memorias de efectos en evaluación congelada. Se ensayan dos programas reales por mundo en cada una de 16 épocas. El log de cada ensayo registra recompensa, pasos, fallo y estado observado, nunca el programa correcto, truth ni winner.
- Se descargarán los pesos **ORIGINALES** F10-L del run **41685499939**, verificando sus SHA frente al manifiesto G0 pre-OOD. Ni PyTorch ni los pesos de los modelos de habilidades se modifican.
- TRAIN 26100000+, DEV 26200000+, OOD 26400000+, CONFIRM **26300000+ sellada**. MEMORIA+FLIP1 está ausente de TRAIN/DEV; OOD tiene 9/11 informes, cuatro fuentes y cuatro variantes causales por mundo.
- Los ficheros de experiencias, resúmenes de recompensa y modelos de transición se congelan por SHA ANTES de abrir DEV/OOD. Un auditor en job separado y sin PyTorch reconstruye 786.432 experiencias reales de las cuatro réplicas, valida nombres de campos, bit F4 solo después de READ y ausencia de SRX en TRAIN, aplica 2.000 bootstrap agrupados por mundo y G0/G1/G2/G3/G4.
- Un fallo científico se informa aunque la ejecución técnica termine SUCCESS; **no** se retocan parámetros para rescatar OOD, no se abre CONFIRM si el resultado falla y no se fusiona en master ni producto.

**PR #66** de investigación, en borrador. Un resultado positivo solo demostraría reutilización de efectos observados para programar secuencias nuevas dentro de una estructura diseñada por nosotros, no invención general de operaciones.
