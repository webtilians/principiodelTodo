# F9-V — Apertura única de confirmación congelada

**2026-11-01.** El protocolo [F9-V preregistrado](./INFINITO_F9V_LEXICOGRAPHIC_CAUSAL_F4_PROTOCOL.md) cumplió el criterio DEV `passes=true` en [run #41345497955](https://github.com/webtilians/principiodelTodo/actions/runs/41345497955). Antes de abrir un solo caso de `8300000+`, se han añadido:

1. [Plan de confirmación con criterio congelado](./INFINITO_F9V_INDEPENDENT_CONFIRMATION_PLAN.md).
2. [Verificador de los pesos originales de cada réplica](../observatory/f9v_frozen_confirm.py) incluyendo SHA-256 de 12 archivos `.pt`.
3. [Workflow de ejecución única](../.github/workflows/infinito-f9v-independent-confirm-once.yml), que aborta ante cualquier cambio de código/protocolo/pesos.

**Activa deliberadamente UNA ejecución** de confirmación de los cuatro modelos congelados, sin reentrenamiento, en 40 pares de mundos sellados originales/invertidos por réplica y ambos regímenes de memoria. Mide 640 estados y 1280 continuaciones contrafactuales de lectura, exclusivamente hasta la primera elección de brazo tras el recorrido docente común.

El resultado será reportado **positivo o negativo sin ajustes posteriores**. Tras esta ejecución la partición `8300000+` se considera CONSUMIDA y no puede reutilizarse como validación de variantes F9-W.
