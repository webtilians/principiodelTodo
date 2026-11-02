# F10-F — Único lanzamiento de auditoría mecanicista congelada

**2026-11-02.** El [protocolo F10-F](./INFINITO_F10F_FROZEN_MECHANISM_PREREG.md) fue registrado de forma prospectiva en commit **`34637e5e3c745d921a28127bb2e6169eb7181d69`**, antes de implementar el generador o cargar nuevas semillas. Las mediciones y clases de error se mantienen invariables.

Ejecutar **solo** mundos nuevos `17_200_000+10000*replica+j`, replica0..3 y j0..107, 108 mundos base/replica y cuatro variantes 2×2 de evento público por mundo: 432 mundos y 1728 evaluaciones dependientes. Reproduce el diseño público F10-E pero no sus semillas ni resultados ni actores reentrenados. F10-F es una *auditoría descriptiva mecanicista en datos nuevos*, NO confirmación independiente ni nueva prueba de generalización de arquitectura.

Los actores congelados originales F10-C `relational_read` y `unary_read_control` proceden del run **#41447171697**, cuyos pesos se validan por SHA. Se instrumenta y reproduce exactamente la ruta `topk5 → unary+pairwise → softmax → lectura → GRU+head`; exigir desviación <=1e-5 respecto a READ y SKIP reales en todos los casos.

Se anotan candidatos seleccionados, peso y score por evidencia, inclusión o ausencia del último registro firmado, y **solo con fin diagnóstico** se parchea la atención a registros firmados, se incluye el ganador o se utiliza onehot del ganador. Los patches seleccionan mediante regla pública conocida por el observador: **NO son decisiones aprendidas ni resultados operativos**. Los errores quedan clasificados sin solaparse como `MISSING_WINNER`, `HEAD_OR_HIDDEN_LIMIT` (sin atribución exclusiva), o `MIXING_OR_WEIGHTING_SENSITIVE`.

El agregado independiente certifica ejecución, fidelidad matemática y presencia de todos los casos, no éxito del agente. **No entrenar ni alterar checkpoints, no abrir confirmación `17_300_000+` ni anteriores, no cambiar thresholds después de inspeccionar errores.**

Este commit dispara `.github/workflows/infinito-f10f-frozen-mechanism.yml`.
