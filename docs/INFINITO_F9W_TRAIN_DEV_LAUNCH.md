# F9-W — lanzamiento de desarrollo con nuevos ejemplos positivos

**2026-11-01.** Una sola ejecución TRAIN/DEV del protocolo F9-W registrado antes del código: [protocolo](./INFINITO_F9W_POSITIVE_EVIDENCE_SCARCITY_PROTOCOL.md), commit `d1e2e44775491cd6f8a1910252df55a5d25b2920`.

Nueva partición TRAIN `9100000+` con 128 mundos por réplica, conjunto de contraste de 32 mundos incluidos en esos 128; todos con las dos pistas key/ball y ambos regímenes normal/blackout. Tres políticas H/C: baseline 32, exposición 128, baseline 32 con BCE ponderada. Control 128 sin H/C. Exactamente 200 actualizaciones Adam, arquitectura idéntica, congelación de F9-P y F4. Sin LLM.

DEV completamente disjunto `9200000+` de 16 mundos por índice. Nunca acceder a CONFIRM F9-W `9300000+`, ni al conjunto ya consumido de confirmación F9-V `8300000+`, ni a los holdouts de F9-T/U o histórico.

Criterio preregistrado: comprobar positivos adicionales por réplica, diferencia en detección de lecturas beneficiosas en índices 1/2, posible aumento de lecturas perjudiciales y acierto del primer brazo. Sin ajuste retrospectivo.
