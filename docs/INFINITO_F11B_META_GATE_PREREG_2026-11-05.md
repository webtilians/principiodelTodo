# INFINITO F11-B — aprender cuándo reutilizar una política y cuándo replanificar
**Prerregistro prospectivo; 5 de noviembre de 2026. Solo diseño, anterior a código F11-B y a la primera generación de datos F11-B.**

## Motivación y pregunta
En F11-A v3, con exploración neutral y 128 mundos TRAIN retenidos por réplica, A2 alcanzó 512/512 con reglas nuevas pero 0/512 al recuperar reglas antiguas. A3 ganó 490/512 nuevas y 122/512 antiguas; su ventaja general NO se confirmó, y su memoria ocupaba 6,08–6,68× más que el A2 estimado. ¿Puede un meta-controlador **aprender cuándo elegir** A3 (opción aprendida) o A2 (planificación por efectos) a partir de las recompensas que recibe, con una ventaja en entornos cambiantes y sin recibir la fase real de las reglas?

Este es un estudio **nuevo, F11-B**, no la confirmación de F11-A. Se reutiliza exactamente el entorno sintético y los controles A2/A3 desarrollados en F11-A v3, sin mirar secretos del usuario, herramientas reales ni modelos de lenguaje; las limitaciones de los controles se mantienen.

## Diseño predefinido
- Cuatro réplicas r0..r3. Espacio nuevo TRAIN F11-B: `62_100_000 + 10_000*r + i`. Se comprobó previamente que la cadena `62_100_000` no aparece en búsquedas del repositorio; antes de la ejecución verificar límites.
- Cada réplica y sus 512 mundos base reservados dentro de TRAIN F11-B se dividen de manera INMUTABLE: **base-model fit** índices 0..255, **meta-gate fit** índices 256..383, **TRAIN heldout** índices 384..511.
- Base-model fit: 256 mundos/r, 30 episodios de exploración **totalmente aleatoria**, mitad bajo las reglas antiguas y mitad bajo las nuevas; mismo registro público de observación/acción/resultado para modelo de efectos A2 y opciones A3. No hay exploración guiada por A3.
- Meta-gate fit: 128 mundos/r nuevos, 40 episodios/mundo. Un solo brazo A2 o A3 actúa por episodio, seleccionado aleatoriamente, sin recabar el contrafactual. Solo el resultado del brazo ejecutado actualiza la recompensa y la historia observable de ese mundo. Entrenar una política de selección contextual mediante SGD logístico de éxito observable; variables: disponibilidad de opción, éxitos/fallos recientes por brazo y cuántas veces se usó cada uno. **Sin fase**, seed, identidad del mundo ni posición del episodio en las entradas.
- TRAIN heldout: 128 mundos/r totalmente nuevos; 40 episodios/mundo; en cada mundo las reglas cambian **una vez sin aviso al selector**, en un punto pseudorrandom entre los episodios 14..24, fijo por mundo e igual para todas las condiciones. **Congelar los pesos del selector** antes de consultar heldout. Durante heldout la historia local de recompensas observadas de cada brazo puede actualizarse, pero los pesos globales NO.
- Comparar meta-selector aprendido (G) contra siempre A2, siempre A3, **regla simple de cambio tras fracaso** (H), y meta-selector con características de historial apagadas (G-nohist). Todos comparten bases congeladas y mundos/ruido emparejados. Medir éxito, retorno, éxito 5 episodios tras el cambio, adaptación a 20 episodios, decisiones, cambios de estrategia y costes. A4 oráculo no utilizado como fuente de enseñanza.
- Implementación respetará hasta 12 acciones por episodio, mismos límites y catálogo A..F. No se introduce ajuste de pesos de LLM ni producto. Los resultados del usuario no se tocan.

## Criterios prospectivos de interpretación
Los resultados TRAIN heldout son **solo diagnóstico** y no pueden considerarse confirmación, generalización estructural o inteligencia general. Antes de ver datos se fija:
- **Evidencia prometedora** (no PASS científico): G supera a **H** y a **ambos A2/A3** por >=5 puntos porcentuales de éxito global en al menos 3/4 réplicas, y se recupera más rápido tras el cambio. G-nohist debe perder >=5 puntos respecto a G para atribuir valor al historial; no superar H equivale a falta de ventaja incremental de aprender el selector.
- **Evidencia insuficiente**: cualquier condición anterior incumplida. Publicar tasas y comparaciones por réplica sin repetir heldout tras ajustar. Si los controles saturan o G nunca explora el otro brazo, se documenta como problema metodológico, no rescate.
- Control de costes: guardar bytes de modelos base y meta-selector, llamadas, acciones y tiempo; no afirmar ventaja de eficiencia sin medir coste del árbol completo A2+A3 y sin cumplir presupuesto comparable.
- **NO-GO para DEPLOY/DEV/CONFIRM de F11-A**, independientemente del resultado. Una eventual confirmación de F11-B exigiría particiones nuevas, protocolo congelado, grupos por mundo e IC95 pareados, con los controles de memoria y costos ya resueltos.

## Riesgos causales a documentar
La fase no es visible y los episodios muestran indicios incompletos de cambio. La recompensa de la selección puede ser muy ruidosa. El A2 sigue guiado por la representación de recursos, el A3 almacena trazas específicas; la mejora podría depender de la física sintética y no transferir a familias nuevas. Historiales y número de operaciones deben contabilizarse, y una prueba en seed nuevo de la misma familia no es OOD estructural.

**Estado al publicar: no hay código F11-B ni resultados F11-B.**