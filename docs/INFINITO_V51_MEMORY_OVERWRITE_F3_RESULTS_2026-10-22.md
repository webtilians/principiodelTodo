# INFINITO V5.1 — F3: diagnóstico de sobrescritura de memoria

**Fecha:** 22 de octubre de 2026  
**Ejecución verificada:** https://github.com/webtilians/principiodelTodo/actions/runs/39856046936  
**Artefacto:** \`infinito-v51-memory-write-freeze\`  
**Script:** \`scripts/infinito_v51_memory_overwrite.py\`  
**Rama:** \`experiment/emergence-observatory-v0-1\`

## Pregunta

¿La caída de información recuperable tras 17 pasos de distractores
se explica en parte porque la política de escritura de la memoria
sustituye los recuerdos relevantes por nuevos contenidos?

## Diseño e intervención

Se reutilizan EXACTAMENTE los mismos casos sintéticos, semillas
11/29/47, magnitud de distractores 0,55, clasificación lineal externa
y arquitectura original V5.1 reducida que en F2.

Seis condiciones emparejadas:

1. **historical** — implementación V5.1 original;
2. **historical_memory_cut** — borrar memoria después de presentar cue;
3. **historical_write_frozen** — permitir primera escritura, bloquear
   todas las escrituras posteriores de distractores;
4. **early_feedback** — lectura externa reinyectada antes de módulos;
5. **early_feedback_memory_cut** — reinyector pero borrado de memoria;
6. **early_feedback_write_frozen** — reinyector con primera escritura conservada.

El arnés intercepta temporalmente \`EnhancedExternalMemory.write\`
para las condiciones 3 y 6 y la restaura al terminar cada episodio.
La lectura, las LSTM, la atención y los otros módulos permanecen
activos. **No se modificó el archivo original de V5.1.**

Esto es una intervención deliberada que *sabe cuándo es la primera
entrada del episodio*. No se trata de un mecanismo espontáneo de
selección de relevancia aprendido por la IA.

## Resultados de decodificación diferida (17 pasos)

| Condición | Semilla 11 | Semilla 29 | Semilla 47 | Media |
|---|---:|---:|---:|---:|
| Original | 50 % | 56,25 % | 43,75 % | 50 % |
| Original, memoria borrada | 43,75 % | 62,5 % | 50 % | 52,08 % |
| Original, escrituras posteriores bloqueadas | 81,25 % | 87,5 % | 75 % | **81,25 %** |
| Feedback temprano | 50 % | 62,5 % | 37,5 % | 50 % |
| Feedback, memoria borrada | 43,75 % | 62,5 % | 50 % | 52,08 % |
| Feedback, escrituras posteriores bloqueadas | 81,25 % | 87,5 % | 75 % | **81,25 %** |

Cada semilla tiene 16 ejemplos de prueba nuevos por condición
(total de 48 por condición). **Modelo sin aprendizaje de tarea**:
se entrena únicamente un decodificador lineal sobre representaciones
producidas con pesos de V5.1 sin entrenar. La salida decodificable
no equivale a una respuesta autónoma generada por la red.

Los resultados anteriores son una comparación **dentro** de la misma
arquitectura, no una comparación entre puntuaciones de conciencia.

## Interpretación

1. **Dependencia causal funcional:** cuando el mismo almacenamiento
   conserva una primera escritura y evita nuevas escrituras
   irrelevantes, la señal vuelve a ser más decodificable. Ello es
   compatible con que las sobrescrituras expliquen parte de la
   pérdida de información útil de la política original.
2. **La señal existía parcialmente:** en F2, con distractores fuertes,
   el probe de la lectura directa de memoria obtuvo 70,83 % y el
   probe del workspace final solo 50 %. Parte de la dificultad
   parece estar en la conservación *y/o integración* de recuerdos,
   según el lugar desde el cual se observa.
3. **La reinyectación temprana aislada no dio mejora:** F2 y F3
   sitúan las dos rutas \`historical\` y \`early_feedback\` en 50 %,
   y ambas rutas con escritura congelada en 81,25 %.

Esto no demuestra que el sistema sepa seleccionar recuerdos útiles:
el momento de escritura fue suministrado **externamente por el
experimento**. Lo que sí hace es aislar un mecanismo plausible
que afecta a la información disponible posteriormente.

## Limitaciones y siguientes pruebas

- Tres semillas y 16 ejemplos de evaluación por semilla son
  insuficientes para declaraciones generales con incertidumbre
  estadística reducida; hacen falta replicaciones independientes.
- La tarea artificial codifica una pista al inicio y distractores
  fuertes después. Un gate "guardar la primera y nunca más"
  conoce la estructura del experimento y no generaliza necesariamente.
- No se ha demostrado aprendizaje de anticipación, ni generación
  autónoma de preguntas. La arquitectura permanece sin un objetivo
  supervisado que enseñe cuándo el recuerdo será relevante.
- El próximo experimento debe **intercalar** pistas relevantes y
  eventos irrelevantes en posiciones aleatorias, desconocidas para
  el mecanismo, y exigir predicción de información futura.
- Un mecanismo adaptativo de escritura debería tomar decisiones
  basadas en utilidad futura predicha, no conocer los pasos
  ni premiarse por maximizar Φ.
- Es necesario mantener un control de igual capacidad y carga de
  escritura, estudiar errores de falso recuerdo y extender las
  semillas de evaluación sin reutilizar el conjunto de desarrollo.

**Conclusión operacional:** el siguiente problema medible ya no es
cómo *añadir* otra conexión de memoria, sino cómo decidir **qué
guardar, qué sobrescribir y cuándo consultar un recuerdo**.

Eso está estrechamente relacionado con los objetivos de memoria
selectiva de INFINITO Brain, pero no equivale a experiencia subjetiva
ni a curiosidad intrínseca.
