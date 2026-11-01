# INFINITO F9-U — Recompensa contrafactual por pares para la política neuronal de lectura F4

**Fecha: 2026-11-01. PREINSCRIPCIÓN ANTES DE IMPLEMENTAR, ENTRENAR O ABRIR DEV/CONFIRM.** Rama apilada sobre F9-T; ningún LLM.

## Motivación verificable

F9-T DEV (#41327365559) terminó, con 4 checkpoints PPO sham congelados, 600 mundos de entrenamiento/semilla y 20 pares DEV/semilla:
- `learned_hc`: 36/80 pares de primeras elecciones correctas, 50,3% READ/frame. Réplica 0: 0/20 con 7,6% READ; 1: 16/20 con 100% READ; 2: 0/20 con 0% READ; 3: 20/20 con 93,5% READ.
- `learned_public_only`: 56/80, 90,6% READ; `always`: 55/80, 100% READ; `never`: 0/80. No se cumplió gate F9-T. El F9-T original hacía **una actualización REINFORCE por episodio** usando `R_ind = R_nativo - 0.15 × fraction_READ`, sin tratar el cue invertido como una unidad de aprendizaje.
- El agente puede alcanzar recompensas individuales aproximadas de 50% seleccionando siempre la misma salida. Necesitamos demostrar que se condiciona por la pista, no solo que evita lecturas.

## Pregunta y cambios permitidos

**¿Cambia el aprendizaje de la cabeza READ/SKIP si se añade una bonificación por resolver las DOS variantes de un mismo mundo, frente a un control con el mismo número de muestras y de pasos de optimización?**

El actor es SIEMPRE el RecurrentPPO/LSTM congelado `ppo_sham_after_rl.zip` de F9-P, por índice 0..3, con validación SHA de F9-O/F9-P antes de entrenamiento. La memoria F4 real es `SelectiveF4("context",8)`, mismo escritor y sus reglas, vector 148+512, H/C del actor como features exclusivas de la cabeza cuando está habilitado. No entrenar el actor; no afectar su estado H/C salvo el readout enmascarado por SKIP; no modificar producción.

Dos factores factoriales **2×2**, cabezas de igual arquitectura MLP 276→32→1, mismas inicializaciones por índice y modo informacional, mismos cuatro checkpoints y las mismas dos variantes de cada uno de 600 mundos TRAIN:

- `individual_hc`: H/C disponible; recompensa individual `r_i = reward_nativo_i − 0.15*read_fraction_i`.
- `paired_hc`: H/C disponible; recompensa conjunta
  `R_pair = 0.5*(native_A+native_B) + 0.5*min(native_A,native_B) − 0.15*0.5*(read_fraction_A+read_fraction_B)`.
- `individual_public`: H/C cero solo en la cabeza, misma regla individual.
- `paired_public`: H/C cero solo en la cabeza, misma recompensa conjunta.

**Escala de la bonificación preregistrada `beta=0.5`**. No se entrega `cue`, `correct_side`, `first_choice_correct`, el lado docente ni información privilegiada a la cabeza, al actor ni a los términos de recompensa. El entrenamiento usa exclusivamente las **dos recompensas nativas de los episodios** y sus tasas de READ para actualizar después de finalizar el par. Entrenar sobre dos variantes originales/invertidas es parte del currículo, pero **no** aportar objetivo o evento oracle al ejecutor.

## Comparabilidad y estimador del gradiente

- **600 iteraciones de entrenamiento/semilla/cabeza, cada iteración = dos episodios** originales/invertidos del mismo mundo, una sola actualización Adam de cabeza después de ambos. Total **1.200 episodios y 600 actualizaciones por cabeza**, frente a F9-T original de 1.200 actualizaciones; esta diferencia obliga a evaluar el NUEVO control individual de minibatch de dos, no a presentar la cifra histórica F9-T como control causal.
- En ambos tratamientos usar **un solo optimizador Adam con `lr=0.001`** (como F9-T), entropía 0.001, clip_norm 2, inicialización de cabeza igual dentro de cada índice+modo informacional, `torch.manual_seed` controlado; malla exacta de mundos y orden sin selección por éxito.
- `logp_i` es la **media temporal de log-probabilidades de READ/SKIP** de todos los frames del episodio `i`, incluidos frames con F4 vacío (todo READ cuesta), diferenciable solo respecto a cabeza.
- Gradiente de control individual: `L_ind = −0.5*[logp_A*(r_A−b_ind)+logp_B*(r_B−b_ind)]−0.001*mean(entropy_A,entropy_B)`.
- Gradiente de recompensa por pares: `L_pair = −0.5*(logp_A+logp_B)*(R_pair−b_pair)−0.001*mean(entropy_A,entropy_B)`.
- `b_ind` y `b_pair` son baselines escalares EMA sin gradiente, actualizados **solo después** de optimizar el par, con tasa 0.02, separados por brazo. No confundir esta recompensa conjunta con atribución causal de crédito READ por frame: sigue siendo un REINFORCE de episodio de alta varianza.
- Entrenamiento ejecuta los pares con política Bernoulli estocástica; evaluación reporta **tanto determinista con umbral 0.5 como estocástica** con semilla independiente preregistrada (no elegir el mejor modo a posteriori). Ningún checkpoint se escoge por su resultado DEV.

## Particiones y recursos

Todas las semillas son nuevas, separadas por réplica `index∈[0,3]`:
- TRAIN `7100000 + index*10000 + j`, j=0..599, ambos cues.
- DEV `7200000 + index*10000 + j`, j=0..19, ambos cues, cuatro cabezas y controles sin entrenamiento. DEV puede servir para diagnosticar errores técnicos pero no retocar pérdidas/criterios tras ver CONFIRM.
- CONFIRM **sellado, sin ejecución en la fase actual**: `7300000 + index*10000 + j`, j=0..39. Abrir solo tras gate de integridad y preregistro; nunca reusar 6300000+ (reserva F9-T), 900000+ histórico, ni semillas F9-S/Q/R.
- Igual número de training worlds, cues y pasos de optimización en cada brazo. El número de **pasos del entorno**, variables por duración de navegación, se reporta como costo adicional, no se promete igualdad de frames.
- No instalar SDK de OpenAI; las dependencias son Torch, RecurrentPPO, MiniGrid/observatory, no modificar `src/` ni las políticas fuente.

## Baselines, métricas y criterio

Referencias por cada checkpoint en los **mismos** DEV mundos: `always`, `never`, `periodic_4`, `public_event_clock`, más control nuevo `individual_hc`/ `individual_public`. Contrastes 2×2 individual↔paired solo comparables por mismo input y mismo minibatch.

**Primaria**: conteo de PARES con dos primeras elecciones de brazo correctas, de 20 por réplica y 80 DEV total; reportar también dos éxitos terminales por par, la fracción de READ por frame (agregada como lecturas totales / total frames y por réplica), episodios que llegan al cruce y tasa de READ antes/después del cue. **Gate de avance exploratorio a CONFIRM:** `paired_hc` gana ≥**5 pares /80** a `individual_hc`, con ganancia en ≥**3/4 réplicas** y ninguna caída de más de 2/20; rendimiento de `paired_hc` no inferior a `always` por más de **4/80 pares** (5pp) y reduce lecturas ≥**25%** frente a always; `paired_hc` supera `paired_public` por al menos **4 pares/80** para poder atribuir ventaja funcional a H/C. Estos criterios son exigentes: si no se cumplen, **F9-U como hipótesis fuerte es NEGATIVO**, aunque existan mejoras parciales. No abrir confirmación en caso de fallo.

**Control de atajo fijo**: reportar `chosen_side_A`, `chosen_side_B` y pares con decisión idéntica en las dos variantes aunque la pista cambie. En casos sin compromiso (none), no se atribuye falsamente una decisión correcta. Reportar las evaluaciones determinista y estocástica **separadamente**, sin elegir la mayor.

**Control H/C**: con cabeza `paired_hc` congelada, anular H/C **solo en entrada de la cabeza** y mantener el estado H/C completo del actor y escritura F4; medir acierto y lecturas. Diferencias no prueban por sí solas uso de recuerdo específico; requieren interpretación de dependencia causal y posible shift fuera de distribución.

**Informe científico:** dar resultados por réplica, agregado, tasa de lectura, número de pasos, presencia de política constante/degenerada, contraste par/individual y H/C/public, hash de checkpoint, pérdida/EMA. Nunca convertir un `SUCCESS` de CI técnico en una victoria de capacidad.

## Reglas de parada

Una confirmación requiere cumplir gate **antes de abrir 7300000+**. Si falla, documentar fracaso y conservar CONFIRM virgen. No cambiar a posteriori `beta=0.5`, LR, penalización `0.15`, forma del crédito o fuentes de actor para elegir mejor resultado. Cualquier experimento adicional deberá preinscribir su hipótesis, nueva partición y coste.
