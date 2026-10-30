# F9-A: Resultados técnicos de los primeros entornos externos

**Fecha:** 2026-10-29. **Estado:** Stage A implementado, CI aprobado; no existe aún comparación INFINITO vs baselines.

**Código:** [observatory/f9_external_minigrid.py](../observatory/f9_external_minigrid.py)  
**Protocolo registrado:** [INFINITO_F9_EXTERNAL_EVALUATION_PROTOCOL.md](INFINITO_F9_EXTERNAL_EVALUATION_PROTOCOL.md)  
**CI F9:** https://github.com/webtilians/principiodelTodo/actions/runs/40859890807  
**CI portal:** https://github.com/webtilians/principiodelTodo/actions/runs/40860012556

## Qué hemos ejecutado y qué no

Ejecutados sobre MiniGrid 3.2.0, Stable-Baselines3 2.9.2 y SB3-Contrib 2.9.2, Python 3.11:

- **Memory S7:** PPO MLP, RecurrentPPO LSTM y random. 1 semilla, 128 transiciones por algoritmo, 2 episodios de validación de desarrollo por algoritmo.
- **DoorKey 5x5:** los mismos tres algoritmos, misma duración.
- **DynamicObstacles 5x5:** los mismos tres algoritmos, misma duración; tiene **3 acciones**, no 7. El test inicial asumía 7 y falló; se corrigió el test, no se alteró el entorno externo.
- Las observaciones son **parciales**, idénticas para todos: imagen simbólica egocéntrica 7×7×3 y dirección. No se habilita vista completa.
- No se realizó ningún test retenido; tampoco se han conectado módulos propietarios de INFINITO al RL externo.
- El portal pasó 22 tests con su nuevo catálogo, incluyendo autorización para F9 **deshabilitada por defecto**.
- En CI las dos variantes entrenadas son políticas reales, con pesos PyTorch guardados como checkpoints, no imitaciones ni reglas construidas a mano.

## Resultados de la prueba técnica (NO significancia científica)

| Entorno | Random | PPO MLP | RecurrentPPO LSTM |
|---|---:|---:|---:|
| Memory | 0 / 2 | 0 / 2 | 0 / 2 |
| DoorKey | 0 / 2 | 0 / 2 | 0 / 2 |
| DynamicObstacles | 0 / 2 | 0 / 2 | 0 / 2 |

**No atribuir estas tasas a un fallo cognitivo**: el presupuesto de 128 transiciones solo sirve para verificar que los algoritmos aprenden realmente un gradiente, se ejecutan y producen resultados reproducibles. DoorKey y Memory tienen recompensa escasa y necesitarán entrenamiento mucho más largo. Los intervalos por semillas no se calculan con una sola semilla.

El proceso de CI para DynamicObstacles detectó que la política aleatoria chocaba con obstáculos (retorno -1) mientras que las redes sin entrenar agotaban el horizonte (retorno 0). Es un comportamiento esperado que muestra por qué **recompensa y tasa de éxito se deben reportar por separado**.

## Diseño de lo siguiente

1. Mantener las tres tareas independientes; preparar modelos PPO y LSTM con 5 semillas y presupuesto creciente **solo en desarrollo**.
2. Medir retorno, éxito, longitud de episodio, parámetros de red y tiempo de entrenamiento. No seleccionar el mejor checkpoint por episodios retenidos.
3. Solo después, introducir una **integración efectiva** del módulo de memoria INFINITO y su ablación frente a LSTM, bajo igual régimen de datos/recursos. Si no podemos igualar capacidad/cómputo, reportarlo.
4. Congelar checkpoints, dependencias y hashes antes de una única evaluación retenida. La CLI incluye modo \`--phase heldout\` que carga modelos congelados; NO reentrena.
5. Si PPO y LSTM no resuelven la tarea en desarrollo, revisar recursos/entrenamiento antes de evaluar INFINITO; no atribuir a INFINITO una mejora falsa frente a una referencia débil.

## Dependencias, recursos y servidor

F9 importa MiniGrid, Torch y Stable-Baselines3 **solo en el proceso dedicado**. No es seguro activarlo sin comprobar la RAM de la máquina compartida (systemd del laboratorio tiene MemoryMax=600M). Por eso las tareas del portal se muestran pero **rechazan la ejecución hasta que el administrador configure** \`LAB_F9_ENABLED=1\` y \`LAB_F9_PYTHON\` hacia un intérprete Python F9 instalado por separado.

La rama F9 **no se ha fusionado ni desplegado automáticamente en el VPS**: el servidor sigue actualmente la rama configurada por el administrador. Es deliberado.

### Ejecución local/servidor de cómputo dedicado

~~~bash
python3.11 -m venv .venv-f9
. .venv-f9/bin/activate
pip install -r observatory/requirements_f9.txt
python -m observatory.f9_external_minigrid --task memory --phase development --seeds 1 --timesteps 128 --eval-episodes 3 --output observatory_results/f9_memory_smoke
~~~

La salida conserva \`report.json\`, \`frozen_manifest.json\` (huellas de modelos/dependencias/código) y \`checkpoints/*.zip\`. Para la evaluación retenida —solo después de congelar la configuración, no para afinar el algoritmo—:

~~~bash
python -m observatory.f9_external_minigrid --phase heldout --checkpoints observatory_results/f9_memory_smoke --eval-episodes 500 --output observatory_results/f9_memory_heldout
~~~

La ejecución \`--phase heldout\` falla si se modifican pesos/checkpoints, dependencias o el código del benchmark desde su congelación.

## Limitaciones aún abiertas

- No hay integración de módulos INFINITO todavía. Stage A son baselines externos.
- La arquitectura LSTM tiene distinto número de parámetros; se registran cantidades y tiempos, no igualdad exacta.
- Las semillas de evaluación reservadas son nuevas dentro de las mismas familias MiniGrid. Para probar transferencia **entre tareas** harán falta otras variantes y un protocolo separado.
- La configuración corta de CI no es evidencia de superioridad de ningún agente.
- No hay acceso a experiencia subjetiva ni medida de conciencia.
