# INFINITO F10-K — lanzamiento único de composición a partir de primitivas (2026-11-03)

**Prerregistro original irrevocable**: [INFINITO_F10K_COMPOSITION_PREREG.md](INFINITO_F10K_COMPOSITION_PREREG.md), commit 1cbddbd092b4c743f2b6914f36193bbb0db182b8, publicado **antes del generador, el modelo y cualquier dato F10-K**.

Este archivo activa **una sola ejecución** del workflow .github/workflows/infinito-f10k-composition.yml. La PR #57 permanece en borrador, basada en la rama F10-J, sin modificar master/producto.

Cuatro replicas × cuatro brazos: A mismo actor entrenamiento conjunto P/I/S, B mismo actor y ejemplos pero orden curricular P+I, S, repaso, C currículo con intérpretes público/memoria separados (más parámetros), D supervisión privilegiada de ganador/acción solo como techo. A/B/C reciben recompensa terminal de sus propias acciones, nunca etiquetas como pérdidas. Exactamente 24.576 episodios ejecutados por modelo y 8.192 episodios de cada primitiva, 384 updates.

Preflight abre solo TRAIN 22100000+. **Después de que los cuatro actores de la réplica estén entrenados, guardados y verificados por SHA**, genera DEV primitivas 22200000+ y OOD composición 22400000+, que **nunca fue entrenada** (source=memory, flip=1). Los 16 checkpoints quedan fijos antes de auditoría independiente con 2.000 bootstrap por mundo; confirma G0 primitivas, G1 composición sin ajuste y G2 uso causal de memoria+flip. G3 compara currículo vs training mezclado.

CONFIRM 22300000+ SELLADO, no abrir. Los hashes incluyen semilla/variante para identificar casos; no afirmar que 29.696 hashes representan patrones binarios diferentes. No tuning después de DEV/OOD, publicar resultado aunque FAIL, no integrar código directamente al producto.
