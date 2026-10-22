# INFINITO V5.1 — Primera prueba de memoria temprana (F1/F2)

**Fecha:** 22 octubre 2026  
**Rama:** \`experiment/emergence-observatory-v0-1\`  
**Ejecución válida y corregida:** https://github.com/webtilians/principiodelTodo/actions/runs/39855496546  
**Artefacto:** \`infinito-v51-memory-feedback-with-interference\`  
**Código:** \`scripts/infinito_v51_memory_feedback.py\`  
**Protocolo:** [INFINITO_V51_EARLY_MEMORY_FEEDBACK_PROTOCOL.md](INFINITO_V51_EARLY_MEMORY_FEEDBACK_PROTOCOL.md)

## Resultado resumido

La técnica **sí cambia los estados internos** de la red, pero **no
aumentó la precisión de recuerdo** de una señal antigua con el protocolo
probado. No hay evidencia de curiosidad espontánea ni conciencia.

Se ejecutaron tres inicializaciones (11, 29, 47) del modelo original
\`ConsciousnessBoostNet\`, con configuración compacta
(\`hidden_dim=64\`, 16 slots), cuatro módulos cognitivos,
memoria externa y un reinyector experimental de ganancia fija 0,5.

Se alimentaron secuencias estructuradas con pistas binarias en el
primer paso y consulta neutra al final. El modelo original **no se
entrenó** para recordar ni predecir: únicamente se entrenó un
decodificador *ridge* lineal de las representaciones congeladas.
Por tanto hablamos de información **decodificable**, no de
capacidad autónoma aprendida del sistema.

## Resultados de precisión en datos no vistos

| Condición | Prueba corta: 5 pasos, ruido 0,12 | Prueba larga: 17 pasos, ruido 0,55 |
|---|---:|---:|
| Original V5.1 | 100 % | 50,0 % |
| Feedback previo a módulos | 100 % | 50,0 % |
| Original + memoria borrada tras cue | 61,1 % | 52,1 % |
| Feedback + memoria borrada tras cue | 61,1 % | 52,1 % |

Las medias son de tres semillas. Cada semilla evaluó 24 casos nuevos en
el test corto y 16 en el largo.

**Control sin señal:** con amplitud de cue exactamente cero, el
clasificador tuvo medias de entre **41,7 % y 52,8 %** (según condición),
lo que no indica una decodificación consistente de etiqueta.

## Qué sí se demuestra

- V5.1 original retiene información decodificable de una pista reciente,
  incluso sin entrenamiento supervisado de la arquitectura.
- Borrar su almacenamiento externo después de presentar la pista
  reduce de 100 % a 61,1 % el probe de recuerdo corto en las
  configuraciones probadas; es evidencia de dependencia funcional
  de esa memoria en este protocolo.
- Reinyectar la memoria en módulos previos produce cambios
  medibles de activación, pero **no supera al original**.
- Con una demora mayor y distractores fuertes, las cuatro condiciones
  están próximas al azar: no se observa retención robusta.
- Se verificó que dos trayectorias comparadas utilizan estímulos
  idénticos y la misma aleatoriedad **en cada paso**, incluso si
  los mecanismos internos consumen diferentes números de muestras.

## Qué NO se demuestra

- **No** hemos enseñado al modelo a recordar; hemos entrenado un
  decodificador externo para ver si la información está presente.
- **No** se demuestra memoria autobiográfica, curiosidad, agencia,
  autoconciencia ni experiencia subjetiva.
- El test corto satura al 100 %: no puede medir mejoras adicionales.
- El test largo con solo 3 semillas y 16 muestras de test por semilla
  tiene intervalos muy amplios; diferencias de unos pocos puntos
  porcentuales no serían fiables.
- Los parámetros no son los del histórico V5.1 a escala completa.
- El observatorio aún no ha medido calidad de predicción autónoma
  mediante un objetivo aprendido por el sistema.

## Interpretación y siguiente intervención

El sistema puede contener una representación utilizable de una señal
reciente, pero ésta no se conserva con distractores intensos durante
muchos ciclos en la arquitectura no entrenada.

Hipótesis F3: **sobreescritura** de memoria.
Si bloqueamos las escrituras de distractores (dejando lectura y estados
LSTM activos), quizá se conserve mejor la pista inicial.
El experimento se ha implementado de forma independiente en
\`scripts/infinito_v51_memory_overwrite.py\`, para distinguir
sobreescritura de pérdida temporal de información.

Antes de construir un "objetivo propio de curiosidad" conviene
determinar qué falla realmente en la persistencia de información.
