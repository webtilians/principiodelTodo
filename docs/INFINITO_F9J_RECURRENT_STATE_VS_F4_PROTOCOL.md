# F9-J — Dónde se guarda la pista cuando F4 tiene ocho eventos

Fecha 29/10/2026. Desarrollo exploratorio con **checkpoints originales congelados F9-I**. No nuevos entrenamientos, no apertura del conjunto reservado.

## Hipótesis

En F9-I, ocho eventos con selección contextual obtuvieron 72,5 % de acierto S13 y 21/40 pares resueltos, aunque ninguno conservaba el **primer evento** de pista al llegar al cruce. Esto no demuestra que no haya otras representaciones del recuerdo. Queremos distinguir el estado oculto/celda LSTM, el readout externo F4 y su posible interacción.

## Prueba A: navegación autónoma con modelos congelados

Cuatro semillas originales, diez pares de desarrollo para cada tamaño S7/S13. Repetir el comportamiento normal exactamente y comparar:
1. NORMAL: checkpoint original sin intervenciones.
2. RESET_HIDDEN_AT_FORK: poner a cero solo la LSTM en el primer instante de bifurcación; mantener F4 intacto.
3. ZERO_F4_AT_FORK: borrar solo los 512 canales del readout externo al llegar al cruce; mantener intacta la LSTM.
4. RESET_HIDDEN_AND_ZERO_F4_AT_FORK: ambas intervenciones en el mismo instante.
5. ZERO_F4_AFTER_CUE: quitar los canales externos en cada observación donde la pista ya haya sido vista y no esté actualmente visible, preservando la información recurrente creada previamente.
6. ZERO/RESET_HIDDEN_AT_FORK: repetir reset en el agente con canales de memoria siempre vacíos, para detectar perturbación general de navegación.

Los modelos reciben 660 canales originales, siete acciones nativas, mismos episodios/splits y reward. La geometría que identifica la primera bifurcación sirve SOLO para disparar la intervención evaluadora; nunca se comunica a la política. Registrar cobertura real: episodios sin bifurcación no han recibido reset efectivo. El reset de LSTM es fuera de distribución y no prueba por sí solo memoria semántica.

## Prueba B: bifurcaciones emparejadas con un prefijo guiado

En cada semilla generar las dos trayectorias nativas del profesor (pista original/invertida). Mantener el mismo prefijo de acciones hasta la primera acción de profesor diferente y exigir que la imagen parcial actual en ese instante sea idéntica.

Procesar ambas historias con el mismo checkpoint F9-I. Congelar pesos y probar las cuatro combinaciones de estado recurrente H y readout F4 M:
- H_original/M_original
- H_original/M_invertida
- H_invertida/M_original
- H_invertida/M_invertida

Comparar acciones y comprobar separadamente identidad de readout, diferencia de estado recurrente y coincidencia con la acción correcta de la demostración. El profesor únicamente suministra el prefijo de la prueba, jamás etiquetas a la política. **Este experimento guiado no es éxito autónomo** y podría estar fuera de distribución.

Si el readout es idéntico y cambiar solo H modifica la acción, eso constituye evidencia de influencia causal del estado recurrente en ese punto. Un hidden distinto sin cambio en acción **no** demuestra decodificación útil. Si M también influye, no atribuir todo a LSTM.

## Condiciones de integridad

Reproducir por semilla exactamente el éxito y pares del reporte original F9-I; verificar SHA-256 de checkpoints context8 y zero antes de empezar. Guardar todos los éxitos y fallos. No modificar PC/VPS. Las cuatro semillas dev NO son holdout confirmatorio. No probar LearnedRelevance hasta disponer de una query retrospectiva observacional y crédito válido.

PR #16 sigue como base experimental, PR #15 es una auditoría diferente que no se debe sobreescribir. Original: https://github.com/webtilians/principiodelTodo/actions/runs/40928547873
