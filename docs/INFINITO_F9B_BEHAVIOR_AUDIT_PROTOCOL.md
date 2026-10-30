# INFINITO F9-B: auditoría de comportamiento congelado (S7 → S13)

Fecha: 29 octubre 2026. Fase: **diagnóstico exploratorio de desarrollo**, NO prueba confirmatoria de INFINITO.

## Pregunta

Los baselines PPO y RecurrentPPO LSTM entrenados con 100.000 y 300.000 pasos en
\`MiniGrid-MemoryS7-v0\` obtuvieron los mismos 27/50 aciertos, el mismo retorno
y longitud media (4,24), pese a tener arquitecturas y checkpoints distintos.
¿Han aprendido una estrategia fija que no utiliza la señal inicial, o responden
realmente a la pista y su memoria?

**No ajustar los cuatro modelos, ni las semillas, ni hiperparámetros a partir de la respuesta.**

## Material congelado

- [100k, Actions run #40870053190](https://github.com/webtilians/principiodelTodo/actions/runs/40870053190): modelos PPO/LSTM, manifest con SHA256; una semilla de entrenamiento.
- [300k, Actions run #40871286349](https://github.com/webtilians/principiodelTodo/actions/runs/40871286349): modelos PPO/LSTM, manifest con SHA256; una semilla de entrenamiento.
- Python 3.11, \`observatory/requirements_f9.txt\`, \`f9b_causal_memory.py\` hash
  inalterado. El evaluador \`f9b_behavior_audit.py\` sólo añade instrumentación;
  no altera las políticas y verifica SHA de checkpoints, código y dependencias.

## Tres pruebas en este orden

**1. Registro conductual S7:** reproducir las mismas 50 semillas de *desarrollo*
(\`seed_for("development", 0, 0..49)\`), guardar cada acción, posición previa y
posterior, recompensa y terminación. El auditor lee de \`env.unwrapped\` la
identidad de la señal de partida (key/ball), la ubicación de ambas salidas y
cuál corresponde al objetivo. Estas variables se utilizan exclusivamente
para **etiquetar la trayectoria a posteriori**; el agente sólo recibe la
observación parcial original de 148 dimensiones.

**2. Intervención pareada en S7:** para cada semilla, repetir *el mismo mundo*
pero sustituir \`Key("green") ↔ Ball("green")\` en el cuarto inicial, y
cambiar entre sí las posiciones verdaderas de éxito y fracaso. Permanecen
idénticos la geometría, posición inicial, dirección inicial, objetos de las
salidas, semilla, pesos del agente y código de inferencia. No se modifica
directamente el estado oculto de la LSTM. El objeto inicial puede no resultar
visible desde todas las posiciones iniciales: registrar cuántas observaciones
iniciales cambian realmente y nunca interpretar como ignorancia una pista
que el agente nunca tuvo oportunidad de observar.

Métricas: salida escogida (superior/inferior/ninguna), tasas condicionales por
tipo de señal, número de pares donde cambia la salida, diferencia de éxito,
cambio de primera acción, identidad de secuencias y conteo por acción. Un
cambio demuestra *sensibilidad conductual*, no por sí solo uso efectivo de
memoria: hace falta estudiar qué información estuvo disponible y cuándo.

**3. Transferencia S13:** cargar los **mismos modelos sin ningún entrenamiento
adicional** e introducirlos en \`MiniGrid-MemoryS13-v0\`, manteniendo
codificación simbólica parcial 7×7×3 y dirección. Evaluar 30 semillas de
desarrollo por política; repetir el intercambio pareado de pista en S13.
Esto explora transferencia de tamaño de pasillo; **no sustituye** un estudio
pre-registrado de generalización con múltiples semillas ni el holdout
congelado de F9-B.

## Controles y cautelas

- Los seis tratamientos causales con F4 **no están** en esta auditoría. Sólo
  hay cuatro checkpoints de los *baselines* externos. No afirmar ganancias de
  INFINITO.
- Un acierto del 54% no implica que el agente recuerde; una política
  prácticamente fija puede acertar aproximadamente la mitad de los mundos.
- La pista inicial puede estar detrás del campo visual. Observación actual
  y posición inicial no equivalen a haber observado la señal.
- S13 puede cambiar geometría, distancia y distribución de arranque; caída
  de rendimiento no prueba por sí sola incapacidad de memoria.
- Las estadísticas de las 50 semillas de episodio **no** constituyen 50
  entrenamientos independientes. No confundir medición pareada con
  significación entre modelos.
- No se utilizan semillas 900.000+, ni \`--phase heldout\`; tampoco se
  entrena sobre S13.

## Outputs

\`report.json\`: hashes, versión de tarea, métricas agregadas por variante
(100k/300k PPO/LSTM). \`trajectories.jsonl\`: acciones y posiciones por
episodio, etiqueta de señal, salida y condicionamiento pareado. Ambos
archivos quedan accesibles como artifacts de GitHub Actions.

El siguiente paso dependerá de las acciones observadas; no se programa
otro incremento de pasos sin antes entender la estrategia aprendida.
