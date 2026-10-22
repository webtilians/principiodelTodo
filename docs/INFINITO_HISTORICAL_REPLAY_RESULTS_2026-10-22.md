# INFINITO — Repetición histórica controlada, V1–V5.1

**Fecha:** 2026-10-22  
**Rama:** experiment/emergence-observatory-v0-1  
**GitHub Actions:** [Ejecución corregida 39848942600](https://github.com/webtilians/principiodelTodo/actions/runs/39848942600)  
**Evidencia:** artefacto *infinito-historical-replay*, id 12457160648, dentro de la ejecución.  
**Script reproducible:** scripts/infinito_replay_probes.py.

## Alcance exacto

Se ejecutaron **36/36 sondas correctamente**: 6 familias de versiones,
3 semillas (11, 29, 47), 2 condiciones (referencia/contrafactual) y
**24 observaciones por sonda**. Total: **864 puntos de observación**.
Entorno: Python 3.11, PyTorch CPU, GitHub-hosted Linux.

No son entrenamientos completos ni replican las duraciones o hardware de
las demostraciones originales. Se trata de sondas controladas de clases,
pasos numéricos y estimadores de las implementaciones históricas. No se
deben comparar sus "porcentajes de conciencia", ni el valor absoluto de
Φ entre versiones.

Los scripts V2–V5.1 se toman de master/archive y src del repositorio
histórico. **V1 se carga de la publicación 14860c32... del 2025-09-16**,
quitando exclusivamente su bloque final de ejecución automática.
No se ha cambiado el algoritmo de los métodos investigados.

La primera ejecución, run 39848374987, tuvo 24/36 éxitos: el arnés
pasó un tensor de atención de 4 dimensiones a los calculadores V5, cuando
estos reciben atención promediada de 3 dimensiones. Los errores fueron
del arnés, **no** evidencia de un defecto V5. Se corrigió y se repitió
toda la batería en 39848942600: 36/36.

## Resultados principales

| Versión | Intervención | Medida (media final de 3 semillas) | Resultado |
|---|---|---|---|
| V1 | Realimentar salida frente a reutilizar entrada original | Desv. est. del campo: 0.048534 -> 0.00000000318 | Colapso del campo casi a cero bajo realimentación |
| V2 | Realimentar salida frente a reiniciar campo de entrada | Desv. est. del campo: 0.282831 -> 0.906130 | Amplificación de la dispersión, sin demostrar estabilidad ni función |
| V3 | Pasar estado oculto anterior frente a pasar None | Máxima diferencia del score: 0 en 72 pares | El forward estudiado no utiliza el hidden_state recibido |
| V4 | Permutar conexiones preservando magnitud media | Diferencia máxima Phi-proxy por semilla: 1.10e-8, 1.20e-8, 2.39e-8 | El proxy no distingue esta reorganización de conexiones |
| V5 | Permutar columnas de atención conservando cada suma | Diferencia máxima Phi-proxy <= 5.96e-8 | El componente probado es insensible al destino de atención en este control |
| V5.1 | Misma intervención sobre atención, con RNG emparejado | Diferencia máxima Phi-proxy: 0 | Insensibilidad equivalente en la sonda de V5.1 |

Los valores de Φ entre V4, V5 y V5.1 **NO** están en una escala
común validada y su comparación absoluta sería engañosa.

## Desglose por familia

### V1 — Primero investigar dinámica, no una etiqueta

Se ejecutó el verdadero método _one_recursion del modelo de septiembre
de 2025, con 24 recursiones y una red que actualiza parámetros como en
el original. La condición contrafactual modifica **solo** el argumento
de entrada: el campo anterior pasa a alimentar el paso siguiente.
Las tres semillas llegaron a un campo casi uniforme y próximo a cero.
Se observan 0 clusters finales en el contrafactual frente a 4 en el
modo original. No indica autoconciencia ni homeostasis: evidencia que
este acoplamiento puede apagar la dinámica.

### V2 — Aumento de amplitud no es emergencia

Se utilizaron el método original _enhanced_sim_step de InfinitoV2
y la predicción de AdvancedNN_Middleware con pesos sin entrenar,
sin iniciar su costoso sistema evolutivo completo. Realimentar el
campo incrementó su desviación estándar final (0.2828 -> 0.9061).
Podría reflejar simple amplificación del sistema no lineal. No
se ha probado resiliencia frente a perturbaciones, recuperación,
aprendizaje, causalidad ni novedad funcional.

### V3 — El argumento de memoria anterior está desconectado

En StabilizedConsciousnessNN.forward(x, hidden_state=None) el
argumento hidden_state se acepta pero no se usa en el cálculo de la
salida. En 3 semillas por 24 pasos, pasar el estado previo o None
produjo exactamente las mismas salidas bajo entradas emparejadas.
Esto demuestra una desconexión en ese método, **no** que todos los
módulos de memoria de V3 estén desconectados.

### V4 — Topología invisible al proxy

La función calculate_phi_approximation depende de varianza del
estado y media del valor absoluto de una matriz de conexiones.
Permutar esa matriz preserva ambas medidas: el proxy devuelve
prácticamente el mismo resultado por construcción. No se ejecutó
PyPhi exacto ni el entrenamiento completo V4.

### V5 / V5.1 — Atención reordenada, misma puntuación

En las calculadoras PhiCalculatorV5 y EnhancedPhiCalculatorV51
la información de atención considerada por el proxy contiene un
promedio que pierde el destino de atención individual.
Las columnas fueron permutadas mientras las sumas de filas se
mantuvieron. El estimador prácticamente no cambió, incluso cuando
se modificó qué posición atendía a cuál.

En V5.1 se emparejó el generador de ruido para que las fluctuaciones
estocásticas de la función no confundieran la comparación.
Esto **no** demuestra que la red completa ignore la atención:
solo que los calculadores probados no reaccionan a esta perturbación
bajo estas condiciones.

## Salida del observatorio

La v0.1 registró **102 candidatos escalares** en las 36 ejecuciones.
Marcó **17 alertas exploratorias** y agrupó **4 conjuntos recurrentes**
en el análisis ciego de identificadores.

La mayoría de alertas aparecen en el paso 6, primera ventana admisible,
y/o están relacionadas con el descenso de la pérdida, colapso del
campo u otras transiciones de inicio. Es plausible que sean
transitorios de inicialización, **no propiedades emergentes nuevas**.

**Cuidado:** el control de permutación temporal destruye la
autocorrelación. Sus p-valores no constituyen demostración robusta de
novedad; la búsqueda entre muchas métricas/semillas tampoco está
completamente corregida. La anonimización de IDs no oculta la
identidad indirecta de familias cuando sus métricas son distintas.

## Qué hemos aprendido / siguiente protocolo

Los resultados sugieren que la evolución del sistema depende de
retroalimentación, pero también que varios "indicadores de conciencia"
eran principalmente resúmenes estadísticos **insensibles a ciertas
estructuras causales**. Esto puede explicar que ciertos experimentos
parecieran mejorar sin necesariamente adquirir las propiedades
que buscábamos.

Próxima fase, aún **no realizada**:

1. Elegir al menos una dinámica real completa (V1 y una evolución V2/V3)
   con pesos, entradas y estados persistentes; registrar tensores
   espaciales, no solo métricas.
2. Añadir nulls autocorrelados (surrogates por bloques/espectrales),
   y pruebas de intervención causal, con plan congelado y nuevas semillas.
3. Construir medidas de estructura espacial, memoria temporal,
   auto-predicción efectiva y recuperación tras perturbación.
4. Comparar los mecanismos con distintas ablaciones. Mantener etiquetas
   neutrales E-xxxx hasta validar causa y replicabilidad.
5. Solo después considerar nuevos módulos autónomos/cognitivos,
   siempre separando desempeño medible de experiencia subjetiva.

**Conclusión:** sí encontramos debilidades concretas que quizá
distorsionaron el camino de investigación; no encontramos evidencia
de conciencia ni de emergencia funcional genuina en estas pruebas
reducidas. El siguiente paso es medir dinámicas y causalidad,
no maximizar una etiqueta Φ.
