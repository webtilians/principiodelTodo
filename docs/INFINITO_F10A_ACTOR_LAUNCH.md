# F10-A — Lanzamiento TRAIN/DEV de nueva generación de actores

**2026-11-02.** Protocolo prospectivo congelado **antes de construir ejecutores y abrir mundos**: [`INFINITO_F10A_ACTOR_RELIABLE_UNRELIABLE_MEMORY_PREREG.md`](./INFINITO_F10A_ACTOR_RELIABLE_UNRELIABLE_MEMORY_PREREG.md), commit **`e44c30704ec9af347da7a3dfe7cbc8d8c64a5065`**.

Este lanzamiento ejecuta exactamente dos entrenamientos de actores del mismo tamaño en cada una de las cuatro réplicas desde F9-O `continuous6_after_bc.zip` validado. Nuevos mundos `12100000+10000*index+j (j=0..127)`, 128 por réplica; cada mundo ofrece ambos cues, ambos regímenes de historial y F4 auténtica/engañosa/vacía, **sin excluir ninguna muestra**. Un segundo brazo de control recibe todas las mismas acciones/frames pero con F4 auténtica. Un total de 1536 secuencias por actor y brazo, 10 épocas de BC, sin PPO posterior.

DEV, sin aprendizaje, sobre `12200000+10000*index+j (j=0..31)`: comparación causal `READ_clean` / `READ_adversarial` / `SKIP` desde la misma vista pública y H/C en el primer cruce, más ejecución independiente autónoma sin docente desde reset. Informar los 16 estratos completos actor×cue×historia, éxitos, errores y oportunidades READ buenas/perjudiciales/neutras; separar gate de robustez y gate de positividad. Las métricas no se usan para ajustar entrenamiento, cabeza ni umbrales.

**El universo confirmatorio `12300000+` permanece sellado**, al igual que F9-V `8300000+`, F9-W `9300000+`, F9-X `9600000+`, F9-Y `10300000+` y F9-Z `11100000+`. Nada de estas semillas se vuelve a evaluar. `F9-P` no se modifica: se crean checkpoints F10-A nuevos.

Esta publicación **dispara una sola ejecución** de `.github/workflows/infinito-f10a-new-actors.yml`. Si aparece un error técnico de preflight se corrigirá el ejecutor preservando el preregistro, sin reinterpretar como nueva confirmación los mismos mundos ya abiertos.
