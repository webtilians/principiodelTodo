# F10-B — Lanzamiento único TRAIN+DEV prospectivo

2026-11-02. La versión vigente del [prerregistro F10-B](./INFINITO_F10B_PROVENANCE_TEMPORAL_PREREG.md) quedó congelada ANTES de implementar generador/actores/evaluación en commit `daa7b621c933d7bf8970dfa13b628d3baf34594e`; el historial de Git conserva la primera propuesta `45296e1c5b5b20d1918412c51c656f5c73d70d64` y su mejora pre-datos anti-fuga de query.

Se ejecuta SOLO `13_100_000+10_000*i+j` (TRAIN, 256 mundos por índice 0..3) y `13_200_000+10_000*i+j` (DEV, 64 mundos por índice). Cada mundo incluye 24 claves, 60 informes públicos con procedencia verificable y secuencias de revisión, y cuatro consultas distintas **con exactamente el mismo historial**. Dos actores de igual arquitectura GRU64: con F4 observacional auténtica vs histórico GRU sin F4. Misma arquitectura, inicio, dosis (12 épocas, batch64, Adam 0.002), train labels, semillas y orden.

El test DEV evalúa para el actor de memoria: READ auténtico, SKIP, F4 obsoleta y credencial F4 falsificada sin tocar estado recurrente; además el GRU-only entrenado independientemente. Incluir toda la matriz de efectos D+,D0,D−, contrastes de firma/revisión, intervalos por mundo y gate prospectivo, sin hacer filtros con etiquetas de DEV.

No se entrena cabeza de READ, no se usan MiniGrid viejos, ni actores F9-P/F10-A. `13_300_000+` queda reservado/sellado. La tarea es sintética: no afirmar transferencia a LLM ni al sistema integrado por conseguir acierto en esta tarea.

Este commit dispara `.github/workflows/infinito-f10b-temporal-evidence.yml`.
