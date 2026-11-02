# INFINITO F9-X — Inicio de atribución neuronal

2026-11-02. Ejecución única exploratoria con el protocolo previamente registrado en commit `6adf6283092d324e8e39afc00bf2e437497679de`.

## Objetivo
Determinar si las decisiones READ/SKIP originales de F9-V dependen de H, C, la identidad de la pista, el régimen histórico de F4 o información adicional. Solo usa las cuatro cabezas `lex_hc` congeladas en F9-V DEV #41345497955 y sus actores F9-P intactos; no hay entrenamiento.

## Datos y controles
- 32 mundos nuevos por actor sobre `9500000 + 10000*index + j`, cuatro estados por mundo, 512 estados de atribución.
- Comparación matched-cue (observación pública igual), trasplante H, C y H/C; trasplante de régimen F4 si observación pública es igual; gradientes e integrated gradients de la cabeza, estrategia cue-key-normal del evaluador y controles lex_public / always / never.
- Los datos `8300000+` ya consumidos NO se usan; `9300000+` y `9600000+` permanecen sellados.
- Datos a reportar por réplica y globales. No cambiar parámetros ni escoger réplicas según el resultado.

Esta declaración activa `.github/workflows/infinito-f9x-frozen-head-attribution.yml`, y no modifica los experimentos antiguos.
