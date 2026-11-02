# F10-H — lanzamiento único preregistrado del factorial de cuatro supervisiones

**2026-11-02.** Protocolo prospectivo fijado en commits **`02e9f0d056e312016ade0fa43a31a69e5c38d705`** y aclaración preimplementación de fuentes estables **`41238d77cfdeb6eac990ccfaaf1dce18c0063bfa`**. Estos commits preceden a cualquier ejecución de mundos F10-H. [Prerregistro](./INFINITO_F10H_SUPERVISION_FACTORIAL_PREREG.md).

Este archivo desencadena una sola vez `.github/workflows/infinito-f10h-supervision-factorial.yml` para entrenar cuatro réplicas originales de CINCO brazos bajo presupuesto igualado: `joint`, `selection_only`, `interpretation_only`, `both` y `gru_only`. Los cuatro primeros conservan exactamente la misma arquitectura unaria, inicialización por réplica, 7458 parámetros, 20 épocas y 640 pasos sobre **las mismas muestras**; las pérdidas auxiliares aportan supervisión explícita adicional solo durante TRAIN.

**Únicos mundos abiertos por este ejecutor:** TRAIN `19100000+`, DEV `19200000+`, OOD `19400000+`. Cuatro variantes por mundo, 48 registros públicos, al menos 4 versiones firmadas y 3+ actualizaciones acreditadas de valor; OOD transfiere de 7/9 a 12/16 informes y de 4–6 a 6–9 firmas, con fuentes de procedencia alteradas y tiempos independientes del almacenamiento. **CONFIRM `19300000+` sellada**, al igual que todas las anteriores. Ningún peso F10-G se usa ni se modifica.

La ejecución guarda los checkpoints **ANTES de generar DEV/OOD**, y el job de auditoría independiente verifica hashes, misma historia/truth para todos los brazos, presupuesto y puertas G1/G2/G3 preregistradas, incluyendo IC bootstrap por mundo. El workflow puede terminar técnicamente con éxito y científicamente con FAIL: **ambos estados se documentan sin retocar el entrenamiento**.

Se autoriza el lanzamiento con este commit, sin reutilizar DEV/OOD históricos ni reajustar a posteriori.
