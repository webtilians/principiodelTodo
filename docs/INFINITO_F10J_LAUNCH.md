# INFINITO F10-J — único lanzamiento prospectivo (2026-11-03)

Original prereg: [`docs/INFINITO_F10J_STRUCTURED_EXPLORATION_PREREG.md`](./INFINITO_F10J_STRUCTURED_EXPLORATION_PREREG.md) commit `9a90692bf0b3f1af69f10df174f82da29a1023e6` **antes del generador, código, datos y primeras mediciones F10-J**. Ajuste prospectivo antes de ejecutar TRAIN `ed679200b1916ee643ec68095f2c03742b362c3b`: ortogonalizar 2 tipos × 2 longitudes × 4 fuentes, no permitir shortcut por longitud. No se modifican gates G1-G4 ni parámetros después de ver resultados.

Este archivo es el **disparador único** de `.github/workflows/infinito-f10j-structured-exploration.yml`. Cuatro replicas, cinco brazos con el mismo modelo y **81.920 interacciones observadas por brazo/réplica**:
A control on-policy, B una trayectoria READ y otra SKIP ejecutadas, C B más mezcla de exploración 50% uniforme de slots, D C más crédito por retorno emparejado, E control privilegiado etiquetado.

La observación pública recurrente NO contiene valores F4; únicamente una lectura externa elegida proporciona el valor. La mitad de los mundos necesitan memoria, la otra mitad contiene respuesta pública y desaconseja READ. Nuevos TRAIN 21100000+, DEV 21200000+, OOD 21400000+; **confirmación 21300000+ sellada**. Tests PRE-TRAIN solo en TRAIN. Los 20 checkpoints finales SHA se guardan ANTES de generar DEV/OOD; un segundo job independiente audita integridad, muestras, subgrupos y G1-G4 con bootstrap por mundo. La PR #56 permanece borrador; jamás fusionar con producto por defecto.

**No interpretar éxito CI como éxito científico. Si un gate fracasa se reporta FAIL y STOP.**
