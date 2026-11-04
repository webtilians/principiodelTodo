# F10-P — lanzamiento único del ensayo preregistrado de resultados públicos

**Fecha: 2026-11-04.** Prerregistro científicamente fijado ANTES de implementar o generar resultados F10-P: [INFINITO_F10P_PREREG_2026-11-04.md](INFINITO_F10P_PREREG_2026-11-04.md), commits **14a44d4ce3c6fd4314a1c15c68318c401175f99c** y precisión técnica preregistrada sobre soporte/consistencia **f30af41d7776887090d63cd90b26f3e0bcdfe5b3**, ambos anteriores a código del algoritmo.

Este archivo activa UNA VEZ el workflow **infinito-f10p-public-tool-outputs.yml**. Cuatro réplicas independientes; cada una ejecuta EXACTAMENTE 98304 programas reales en TRAIN (nuevas semillas 28100000+) y entrega los mismos resultados a los siete brazos. SRX excluido TRAIN. En E100/E50/E0/E_noX/E_shuffleX el algoritmo es idéntico y solo cambia qué salidas públicas ve de P/S/R/X; B_reward solo registra puntuaciones; I_privileged recibe trazas de estado por control diagnóstico. Sin estado privado en aprendiz E. 

Verificar cinco checkpoints originales F10-L por réplica contra manifiesto G0 del run ORIGINAL **41685499939** antes/después. Las siete memorias explícitas se congelan por SHA **ANTES de generar DEV 28200000+ y OOD 28400000+**. Auditor independiente reconstruye 393216 registros y contadores desde el log público, valida campos no privilegiados, costos, SHA, estratos y 2000 bootstrap por mundo base. CONFIRM **28300000+ SELLADA**.

**Criterios G0..G4 definidos antes del lanzamiento**. No cambiar algoritmo, máscaras, umbrales, semillas ni hiperparámetros tras ver DEV/OOD. Reportar PASS o FAIL científico con total claridad; el éxito técnico de Actions NO implica éxito científico. **No tocar master, producto, MCP, datos de usuario ni secretos**; PR #71 DRAFT y archivado científico.
