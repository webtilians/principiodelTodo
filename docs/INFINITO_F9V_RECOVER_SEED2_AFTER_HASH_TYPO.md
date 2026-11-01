# Auditoría F9-V — reparación única del checksum de la réplica 2

**Fecha 2026-11-01.** Primera ejecución de confirmación [#41354569501](https://github.com/webtilians/principiodelTodo/actions/runs/41354569501):

- Réplica 0: éxito; se consumieron sus 40 mundos independientes de CONFIRM 8300000+.
- Réplica 1: éxito; se consumieron sus 40 mundos independientes.
- Réplica 3: éxito; se consumieron sus 40 mundos independientes.
- Réplica **2: abortó ANTES de generar o evaluar cualquier mundo 8300000+** en el paso de validación de checksums de pesos. Motivo literal: `AssertionError: Frozen checkpoint mismatch: lex_public index 2`. No hay cambios de modelo ni resultados de CONFIRM de índice 2.

**Causa comprobada:** al transcribir el SHA-256 de `lex_public.pt` del artifact de desarrollo original `13099027297`, faltaba el último dígito `d`. Se había introducido (erróneamente) la cadena de 63 caracteres `8028987431da04c405b3d1e25e394c03681a1e5369b807a9837a72af167d248`. El hash **original verificado independientemente ANTES del nuevo intento** tiene 64 dígitos: `8028987431da04c405b3d1e25e394c03681a1e5369b807a9837a72af167d248d`. Los otros once SHA-256 habían pasado longitud 64; los pesos exactos se mantienen.

**Corrección permitida exclusivamente de integridad:** commit `22ed577f191e74af7390cc27e4c26a171c70dcb6`, añade la última `d` en la copia del hash de índice 2. CERO cambios al archivo original `observatory/f9v_lexicographic_causal_f4.py`, al protocolo, arquitectura, weights, pérdidas, umbral, entrada, semillas o métrica.

**Plan de recuperación de reproducibilidad:** [workflow](../.github/workflows/infinito-f9v-confirm-seed2-recovery.yml) ejecuta **solamente índice 2** en los mundos confirmatorios aún sellados `8320000+0..39`, NO repite índices 0,1,3. Agrega los artifacts inmutables de la primera ejecución 41354569501 con el único nuevo artifact seed2, aplicando el criterio de confirmación y bootstrap previamente congelados. No se reajustan cabezas, seleccionan resultados o reabren otros holdouts.

Este registro precede a la ejecución de recuperación. El error y su solución se publican para no ocultar una interrupción en la confirmación.
