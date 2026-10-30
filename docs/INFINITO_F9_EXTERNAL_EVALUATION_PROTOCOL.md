# INFINITO F9 — Evaluación externa, sin ajustar al simulador propio

Fecha: 2026-10-29. Estado: protocolo de F9-A registrado **antes de ver resultados**.

## Por qué F9 reemplaza temporalmente F8b

F8 mostró que adquirir información cuando compensa aumenta la recompensa en un mundo diseñado con dos regímenes conocidos. 48 semillas no descartan sobreajuste a **la familia de entornos**. F9 no modifica F8 para subir la puntuación. Usa entornos MiniGrid independientes del proyecto.

## División del trabajo y la evidencia

**F9-A (esta rama):** habilitar los tres mundos externos, establecer el adaptador idéntico de observación, ejecutar comparativas reales PPO y PPO recurrente LSTM (SB3-Contrib) y conservar modelos para evaluación posterior. Estos son **baselines externos**, no «módulos INFINITO». No se anunciarán mejoras de INFINITO hasta tener integraciones verdaderas que se sometan a sus mismas pruebas.

**F9-B (futuro, solo si A funciona):** integrar memoria de INFINITO como mecanismo concreto que sustituya/añada una capacidad de retención, con prueba de ablación y control de parámetros/cómputo. No intentar injertar un almacén simbólico F4/F5 sin interfaz causal ni afirmar que existe una memoria neuronal ya integrada. Después, y por separado, testar modelo predictivo y exploración.

## Entornos externos (paquete fijado)

- memory: MiniGrid-MemoryS7-v0 (la pista solo aparece al principio).
- doorkey: MiniGrid-DoorKey-5x5-v0 (planificar llave → puerta → meta).
- dynamic: MiniGrid-Dynamic-Obstacles-5x5-v0 (evitar obstáculos móviles).

Paquetes de F9-A: \`minigrid==3.2.0\`, \`stable-baselines3==2.9.2\`, \`sb3-contrib==2.9.2\`. Python 3.11+.

**No usar FullObsWrapper.** La entrada idéntica para todos los agentes es la imagen egocéntrica simbólica 7×7×3 y dirección, normalizada a valores [0,1]. La misión textual se excluye para todos los algoritmos; las tres tareas tienen objetivo fijo. Ninguna variante recibe estado verdadero ni conocimiento del mapa completo.

## Comparación justa en F9-A

- Random como control inferior, sin entrenamiento.
- PPO MLP: referencia clásica feed-forward.
- RecurrentPPO LSTM: referencia con memoria recurrente **de una implementación externa conocida**. Tiene distinto número de parámetros; registrar tamaño y tiempo real. No se le atribuyen ganancias de INFINITO.
- Igual número de transiciones **de entrenamiento** por modelo y semilla, misma representación, semillas de evaluación pareadas, misma arquitectura MLP base en lo posible, hiperparámetros fijos (sin exploración manual de test retenido).
- Durante entrenamiento, evaluación en desarrollo **sin usarla para seleccionar checkpoints**; se evalúa únicamente el checkpoint final.
- Variación principal: distintas semillas de entrenamiento; episodios dentro de una misma semilla NO se tratan como independientes para un intervalo entre arquitecturas.
- Reportar éxito, retorno medio, longitud de episodio, duración real, parámetros de política y fallos. Cero éxito significa cero éxito; las tareas con recompensa escasa pueden necesitar cientos de miles de pasos para aprender.

## Separación de semillas y evaluación retenida

- Entrenamiento (por índice): 1.000 + i*1.009; generador del entorno posee sus propios niveles aleatorios.
- Evaluación de desarrollo: 100.000 + i*10.000 + episodio.
- Evaluación **retenida**: 900.000 + i*10.000 + episodio. **No ejecutar** durante el desarrollo, no usar para elegir variantes/hiperparámetros.
- Para lanzar el retenido, primero congelar configuración, revisiones del código, dependencias y hashes de checkpoints; la ejecución separada debe cargar los checkpoints ya guardados sin entrenamiento y sin elegir el mejor post hoc.
- Configuración inicial propuesta por tarea: 5 semillas de entrenamiento, 500 episodios retenidos por semilla, 500.000 pasos PPO por semilla y variante. **Es un objetivo condicionado a cómputo**, no debe iniciarse automáticamente en el VPS de 1 CPU/2 GB.
- Smoke técnico: 1 semilla, 256 pasos por agente, 3 episodios de desarrollo. Solo valida API/ejecutabilidad, no resultados científicos.

## Política de recursos

No añadir paquetes pesados globalmente al servicio web sin aislamiento ni control de memoria. El servidor compartido tiene límite systemd de memoria ~600 MB y ejecuta otros servicios. Instalar PyTorch y RL en un entorno F9 independiente; verificar su RSS antes de habilitar ejecuciones largas, o usar un runner dedicado/GitHub Actions de corta duración. La presencia del nuevo código en GitHub no significa que el VPS haya instalado sus dependencias.

## Regla de decisión

- Si PPO y LSTM no aprenden bajo el presupuesto de prueba, no adjudicar ese fracaso a INFINITO: mejorar SOLO el presupuesto/conjunto de desarrollo y documentar la revisión antes de abrir el test retenido.
- Si un módulo integrado no supera a LSTM bien calibrado, registrar «sin mejora demostrada»; no modificar los parámetros en respuesta al test retenido.
- Antes de afirmar generalización, evaluar mapas/variantes de dificultad verdaderamente diferentes, además de semillas nuevas dentro de cada mapa.
- Éxito del laboratorio no equivale a conciencia. F9 investiga rendimiento, generalización y atribución de beneficios por ablación.

## Indicadores de calidad y próximos hitos

F9-A se considera infraestructura probada solo cuando un CI instala MiniGrid y SB3-Contrib, crea los tres entornos, ejecuta al menos una optimización corta real y emite JSON con éxito/retorno de un conjunto de episodios cuyos estados no se filtran al agente.

F9-B NO comienza hasta documentar esa evidencia. El portal solo debe anunciar agentes/modelos realmente ejecutados, con advertencia de que baselines externos no son integraciones de INFINITO.
