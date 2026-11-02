# INFINITO F10-I — Lanzamiento único del estudio prospectivo de experiencia y RL

**2026-11-02, 22:xx Europa/Madrid.** La pregunta, comparación A/B/C/D, presupuestos, semillas, función de recompensa, costes y gates se congelaron en [prerregistro F10-I](./INFINITO_F10I_EXPERIENTIAL_LEARNING_PREREG.md) commit `1e146c47b5ac8c76019484368c448b197f374802`, con las enmiendas **pre-DATOS** `780444c826f31263409b3177b77edc47b6ce0985` (distractor no firmado posterior en las cuatro variantes) y `e72f6b423bb898e5e0e29ee28e628f9c70365e82` (retraso de 2–4 pasos es solo descuento, sin decisiones intermedias). Las enmiendas se hicieron antes de abrir cualquier F10-I TRAIN/DEV/OOD.

**Una sola arquitectura en 4 brazos**, a igualdad de pesos iniciales por réplica, datos, pasos, 81.920 rollouts por brazo/réplica:
- `rl_standard`: REINFORCE ordinario con baseline EMA de retornos previos.
- `rl_pair`: gradiente estimado con dos trayectorias ejecutadas del mismo mundo y diferencia de retornos.
- `rl_pair_predict`: mismo RL emparejado + predicción de observación verificadora *ruidosa POST acción* como señal experiencial adicional (no etiqueta del índice ganador).
- `supervised_ceiling`: mismo actor con índice y acción correctos etiquetados por el evaluador durante TRAIN; solo techo privilegiado.

Cada caso exige elegir READ/SKIP, índice de F4 si READ y acción binaria. **No hay etiqueta del ganador en el entrenamiento de los tres RL**; el propio reward de acierto ofrece información binaria indirecta sobre verdad y la observación postacción de C es un objetivo predictivo ruidoso. Se compara frente a controles por conjuntos de mundos, no por variantes independientes.

Nuevos namespaces: TRAIN `20100000+`, DEV `20200000+`, OOD `20400000+`; **CONFIRM `20300000+` SELLADA**, sin ruta para generar ni abrir. TRAIN/DEV k6/9, OOD k12/16, múltiples informes firmados y actualizaciones contradictorias, distractor posterior no firmado incluso en `S_late`; fuentes IID de TRAIN y cuatro regímenes OOD. Los pesos finales de las cuatro políticas se guardan **ANTES de generar cualquier DEV/OOD**.

La ejecución dispara una única vez `.github/workflows/infinito-f10i-experiential-rl.yml`. Pruebas PREVIO a TRAIN y auditoría independiente de checkpoints, 20.480 filas evaluadas, 12.288 hashes disjuntos, mismo presupuesto de experiencia por brazo y bootstrap por mundo. Formal `F10I_EXPERIENTIAL_LEARNING_PASS/FAIL` según gates prospectivos G1–G3, sin retuning posterior ni sustitución de OOD.

STOP sin abrir confirmación, PR borrador hasta revisar resultados.
