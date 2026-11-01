# INFINITO F9-V — Única apertura de CONFIRM prospectiva, pesos congelados

**Fecha 2026-11-01. Esta declaración se publica ANTES de abrir mundos de confirmación.**

Protocolo preregistrado antes del código: e1347f4603f781d7730222b13ae5e9bc09591291.

DEV [Actions #41345497955](https://github.com/webtilians/principiodelTodo/actions/runs/41345497955), cuatro réplicas y 256 estados. lex_hc 183/256 correctos con 15 READ; always 182/256 con 256 READ; lex_public 168/256 y 0 READ; oráculo 193/256 con 25 READ. Efectos READ−SKIP: +25, 0=220, −11. **Gate de DEV superado** pero todavía no hay confirmación independiente.

**Integridad antes del CONFIRM:** [manifest con los 4 SHA F9-P, 12 SHA cabezas, 4 SHA reportes, SHA código/protocolo](./INFINITO_F9V_CONFIRMATION_LOCK.json), commit 9466e30e8a7a4e2311a6b8d76e68980f989f2451. El workflow no reentrena ni selecciona; primer job verifica todos los SHA antes de abrir la partición.

**Partición confirmatoria virgen:** cuatro semillas × 40 mundos (8300000 + 10000*index + j), cada uno original+cue invertido×dos regímenes de F4 = **640 estados de primera bifurcación**. Dos continuaciones (READ/SKIP) por estado; la cabeza predice una sin conocer las consecuencias. Prefijo profesor hasta el fork, NO end-to-end.

Baselines always, never, lex_public, scalar_hc, oráculo diagnóstico. Reportar las 640 primeras elecciones, lecturas, efectos D y distribución por réplica/régimen/cue, pares con ambas elecciones correctas.

Criterios sin reajustar: lex_hc acierta al menos 90% de always, consume <=80% READ, al menos tres réplicas pierden no más de dos primeras elecciones respecto a always, supera lex_public >=3 puntos porcentuales y hay casos D positivos medibles. Reportar tanto positivos como negativos.

En DEV las 15 lecturas ocurrieron en 2 réplicas y solo para una identidad de pista: no se interpreta como autoconsciencia o una estimación general de incertidumbre. Esta confirmación evalúa una habilidad ACOTADA de anticipar el valor causal de una lectura.

Después de ejecutar, 8300000+ queda CONSUMIDO cualquiera que sea el resultado. F9-T 6300000+, F9-U 7300000+ e histórico 900000+ quedan cerrados. No usar confirmación para cambiar pesos. PR #37 seguirá en borrador.
