# INFINITO — Punto de reanudación tras F7/F7b

**Fecha:** 23 de octubre de 2026  
**Estado:** acuerdo de siguiente paso, todavía **no ejecutado**.  
**Rama de partida:** `experiment/f7-active-information-value`.  
**Protección:** no modificar `master` ni el V5.1 histórico.

## Siguiente paso propuesto: experimento F8, antes de modificar V5.1

**Hipótesis central:** un agente que predice las consecuencias de sus acciones debe poder advertir que el mundo ha cambiado, reconocer incertidumbre relevante, decidir cuándo comprar información y actualizar su modelo para actuar mejor. No se medirá consciencia.

### Secuencia concreta

1. **Reparar el banco de pruebas F6.** Hacer comparaciones con parámetros de dinámica verdaderamente idénticos en entrenamiento y evaluación para el grupo estacionario. Separar claramente generalización entre parámetros de cambio inesperado. Igualar presupuestos de información entre baselines; volver a estimar intervalos entre semillas independientes.
2. **Replicar F7b fuera de muestra.** Fijar detector y umbrales actuales, probar nuevas semillas, costes, ruido, cambios graduales y cambios que no se corresponden con la inversión simple de dos sensores. Exigir que siga habiendo un control negativo puramente aleatorio y documentar falsas alarmas. Tratar F7b actual solo como resultado exploratorio.
3. **Construir F8 integrado en una rama nueva**, combinando estado homeostático y dinámica aprendida de F6 con opciones de observar/investigar de F7. Las acciones alteran el mundo; las observaciones consumen tiempo o recursos; el modelo de transición se adapta en línea solo con experiencias adquiridas.
4. **Comparar políticas con idéntico mundo e información inicial:** control reactivo; MPC sin exploración; MPC con exploración aleatoria; MPC con valor de información; controlador con reglas/oráculo de cambio; variante sin memoria de experiencias. Registrar decisiones, predicciones antes de actuar, errores a posteriori y motivos operativos de investigación.
5. **Protocolo congelado** y pruebas confirmatorias: semillas independientes y cambios inéditos. Métrica principal: retorno real neto, no error del predictor ni «porcentaje de conciencia». Secundarias: recuperación tras cambio, desperdicio en ruido, calibración de incertidumbre, precisión causal y dependencia de memoria por ablación.

### Regla de decisión y parada

- **Continuar hacia integración INFINITO Brain / V5.1** solo si el valor de información mejora el retorno frente a baselines simples en nuevos contextos, sin curiosidad espuria sobre ruido, y sobrevive a ablaciones de memoria/aprendizaje.
- Si resultados equivalentes se explican por MPC/Bayes/heurísticas sencillas, documentarlos sin postular propiedades emergentes: optimización funcional ≠ conciencia.
- No perseguir un valor Φ alto. No fingir experiencia subjetiva. Explorar fenómenos inesperados solo con controles ciegos, réplicas e intervenciones causales.

## Por qué este es el siguiente paso y no simplemente F5 o una V6

El objetivo original es descubrir **qué cambia el comportamiento cuando un sistema aprende a considerar consecuencias que todavía no han ocurrido**, no engordar la arquitectura. F7 enseñó que una inspección puede valer su coste; F7b, con un detector programado a posteriori, reveló cómo investigar tras un cambio. F8 debe averiguar si ambas cosas mejoran una tarea secuencial real y si la decisión de investigar se puede aprender sin nuestros umbrales explícitos.

**Estado actual:** F7 y F7b documentados y ejecutados; esta hoja de ruta queda guardada como punto de reanudación. No afirmar que F8 ha sido ejecutado.
