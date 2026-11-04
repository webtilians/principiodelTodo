# INFINITO F10-P: aprender efectos de resultados públicos de herramientas

**PRERREGISTRO PROSPECTIVO, 2026-11-04.** Publicado antes de código, entrenamiento o apertura de casos F10-P. F10-N/C siguen congelados PASS interno, F10-O es FAIL de observación exclusivamente terminal. F10-O cambió información y tipo de modelo; por tanto no pudo atribuir causalmente su diferencia con F10-N a observabilidad pura.

## Hipótesis

¿Una **misma memoria de efectos tabular, factorizada** puede sintetizar S→R→X (cero ejecuciones en TRAIN) observando solo los resultados públicos que devuelve cada herramienta, sin recibir trazas privilegiadas de su registro interno? Comparar distintas cantidades de información usando **exactamente el mismo learner, planificador, catálogo y experiencias**. No confundir con aprendizaje de una representación totalmente latente/no estructurada.

## Frontera pública de observación fijada por el investigador

P devuelve un bit (0/1) después de cargarlo; S confirma selección con ACK sin revelar slot/valor/winner; R devuelve bit leído después de ejecutar READ, o ERROR; X devuelve bit transformado después de ejecutarlo, o ERROR; ANSWER final devuelve respuesta y éxito. Los aprendizajes de eventos solo ven la secuencia de op y output, bit público, contexto SOURCE/FLIP, recompensa final. No deben ver before, after, selected, work register, invalid_delta, read_observed, truth, winner, contents, features ni mask. El mundo original puede usarlos internamente para ejecutar y para control I, pero el serializador público no los entrega al aprendiz.

La arquitectura EventEffects infiere de **experiencias reales** las tablas P(public_bit→output), X(valor observable anterior→output), frecuencia de S ACK y R RETURN tras S. La semántica de que un valor devuelto sirve como valor de trabajo observable, que ACK indica selección disponible, la biblioteca de operaciones y el objetivo SOURCE XOR FLIP son **supuestos del investigador**, no descubrimientos del algoritmo. Para planificar, considerar las dos hipótesis del posible resultado de R (0/1), sin ver F4 antes de READ, y enumerar los mismos 85 programas de hasta 3 operaciones con coste .01/operación. No suministrar la regla de inversión de X; aprenderla solo de los pares de valor observable antes/después de ejecuciones reales.

## Brazos sobre el MISMO historial de 98.304 programas ejecutados por réplica

- **E100**: 100% resultados públicos visibles.
- **E50**: cada resultado intermedio se omite de forma independiente con p=0.5, RNG 288000+rep; ausente se representa como ausente, nunca como cero.
- **E0**: sin resultados intermedios, solo terminal answer/success; mismo algoritmo EventEffects sin transiciones observables.
- **E_noX**: todos los resultados P/S/R, pero X siempre oculto.
- **E_shuffleX**: todos los resultados disponibles, pero el bit de output X válido se reemplaza por una muestra independiente Bernoulli(0.5), RNG 289000+rep. Conserva la frecuencia de mensajes pero rompe el mapeo causal de transformación.
- **B_reward**: tabla aditiva de recompensas original F10-N sobre EXACTAMENTE los mismos programas y recompensas.
- **I_privileged (diagnóstico)**: EffectMemory original con trazas before/after privilegiadas. E100 vs I NO es una comparación que aísle solo información: son algoritmos/representaciones distintos. E100 vs E50/E0/E_noX/E_shuffleX sí aísla el nivel o calidad del feedback en el mismo learner.

4 réplicas r=0..3. TRAIN 3 contextos PUBLIC0/PUBLIC1/MEMORY0 ×256 mundos×4 variantes=3072 casos/r. 16 épocas ×3072×2=**98.304 episodios reales/r**, compartidos entre TODOS los brazos, no contar siete veces. Uniforme sobre 84 programas; **nunca ejecutar SRX durante TRAIN**. RNG orden 285000+100*r+epoch, programas 286000+100*r+epoch; máscaras 288000+r, 289000+r. Confirmación 28300000+ SELLADA.

- TRAIN 28100000+10000*r+1000*t+j, t=0..2, j=0..255.
- DEV 28200000+10000*r+1000*t+j, j=0..63.
- OOD 28400000+10000*r+j, j=0..127, solo MEMORY1, 9/11 informes, cuatro modos de fuente y variantes U_before,U_after,S_before,S_after.

Los **20 checkpoint SHA originales F10-L** (run 41685499939) se verifican antes y después. Las siete memorias se congelan por SHA ANTES de generar DEV/OOD. No actualizar, seleccionar ni ajustar después de ver DEV/OOD. Cero modificaciones al producto.

## Criterios científicos

Auditoría independiente revisa SHA, logs públicos sin campos internos, 98.304 programas/r, 0 ejecuciones SRX TRAIN, distribución y disjunción de casos, resultados DEV/OOD y CIs. Bootstrap 2000 remuestreos por cada mundo base (128 mundos/r, agrupar sus cuatro variantes), RNG 291000+100*r+offset. Offsets: 0 precisión E100, 1 E100−F4_shuffle, 2 E100−X_disabled, 3 E100−B, 4 E100−E0, 5 E100−E_noX, 6 E100−E_shuffleX, 7 E100−E50.

- **G0** E100 DEV >=90% en cada contexto visto, >=3/4 réplicas.
- **G1** E100 OOD >=80%, IC95 inferior>70%, cada estrato de n/fuente/variante >=70%, SRX elegido >=80% y cero SRX en TRAIN; >=3/4.
- **G2** E100 normal − F4_shuffle >=20pp y E100 normal − X_disabled >=20pp, IC95 inferiores>0 para ambos; >=3/4.
- **G3** E100−B_reward >=15pp e IC95 inferior>0; >=3/4.
- **G4** E100−E0, E100−E_noX y E100−E_shuffleX cada una >=15pp y cada IC95 inferior>0; >=3/4. Evalúa causalmente la necesidad de feedback por operación y de observar el efecto de X con el MISMO modelo.
- **G5 diagnóstico** E50 OOD >=80% en >=3/4, no requisito.
- **G6 diagnóstico** I_privileged OOD >=80% en >=3/4, no requisito.
- **SCIENCE PASS = G0 AND G1 AND G2 AND G3 AND G4**, cada criterio >=3/4. Si íntegro pero falla: F10P_OBSERVABLE_EFFECTS_FAIL. Si error técnico: INCONCLUSIVE, nunca PASS.

**No significa** inventar operaciones, comprender texto libre, inferir estados totalmente latentes, acciones de agente real ni seguridad de integración. El contrato de salida y búsqueda están diseñados. Un PASS solo permite proponer prueba shadow aislada con herramientas reales y sin accesos a datos personales. Un FAIL no autoriza rescatar el estudio ajustando parámetros. No tocar master, memoria MCP, preferencias/secretos ni pesos anteriores.
