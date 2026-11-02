# F9-Z-A — Informe recuperado sin reejecutar mundos

2026-11-02. La ejecución piloto original [Actions #41370406853](https://github.com/webtilians/principiodelTodo/actions/runs/41370406853) completó correctamente las **cuatro réplicas** y los archivos `f9z-pilot-seed-0..3`; los tests y anclaje original F9-V pasaron. El trabajo agregado posterior falló exclusivamente en la importación de `numpy` en un runner independiente sin dependencias, **antes de calcular el reporte**, sin implicar cambio científico ni de pesos.

Este commit activa **únicamente** el workflow `infinito-f9z-report-recovery.yml` que descarga los **mismos artefactos del run #41370406853**, calcula con Python stdlib el informe completo y audita cada conteo de los 3456 casos. **No descarga actores, no importa PyTorch, no construye mundos ni reproduce intervenciones**. No se alteran las nueve condiciones experimentales, las semillas, los denominadores, el preregistro `bc19d5fe15fd0679a5dbcdf55c161c17f8b18f3e` ni su umbral de soporte.

Las particiones futuras `11200000+`, `11300000+`, `11400000+` siguen cerradas. El run original no debe contarse como un experimento independiente del reporte corregido.
