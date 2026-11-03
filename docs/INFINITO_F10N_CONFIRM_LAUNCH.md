# INFINITO F10-N-C — lanzamiento único de confirmación congelada, sin aprendizaje

**3 de noviembre de 2026.** Este archivo lanza UNA vez el workflow **infinito-f10n-original-memory-confirm.yml**. Su protocolo científico está preregistrado en [INFINITO_F10N_CONFIRM_PREREG_2026-11-03.md](INFINITO_F10N_CONFIRM_PREREG_2026-11-03.md), commit **da441ba1d2f6d02a02dad896ac2574d9a8fdbfc1**, publicado ANTES de implementar cualquier generador o evaluador de nuevos casos CONFIRM.

Se prohíbe modificar los 12 originales SHA de memorias de experiencias F10-N y las cinco políticas F10-L por réplica. Los originales se obtienen de los runs **41714113859** y **41685499939**, verificados contra la auditoría original ya terminada **41714672031**. El código F10-N original congelado es **6d3e7bd8753f3ad5d1d4a6e44208866bac036298**.

Antes de generar un solo dato 26300000+, las cuatro réplicas deben completar pruebas con casos OOD ANTIGUOS 26400000+, validación de 20 SHA F10-L y 12 SHA F10-N, y repetir bit a bit todas las predicciones antiguas DEV/OOD con el evaluador F10-N original e inalterado. Si falla algo, **no** se abre la partición de confirmación.

Después, solamente evaluación original sin entrenar sobre las nuevas semillas **26300000+10000*r+j**, 128 mundos y cuatro variantes por réplica, con modelos A/B/C/D congelados. Un auditor independiente, sin PyTorch, aplicará C-G1, C-G2, C-G3 y C-G4 usando **los criterios originales inalterados** y bootstrap por mundo base.

Un positivo solo significa replicación en semillas nuevas de la misma familia sintética. Mantener posibles fallos y excepciones visibles, no retocar pesos ni repetir buscando mejor resultado. CONFIRM solo una vez. PR #69 DRAFT; no fusionar producto.
