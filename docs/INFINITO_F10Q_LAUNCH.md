# F10-Q — lanzamiento único del estudio preregistrado

**2026-11-04.** El protocolo F10-Q fue publicado en GitHub en el commit **311455d03d81bf01d79273b235766a6ec12a98d3** antes de implementar el código y generar cualquier mundo del experimento. PR #73 en borrador.

Este archivo activa una sola ejecución GitHub Actions del estudio. Entrena 4 memorias independientes con observaciones derivadas de **consultas SQL realmente ejecutadas por un servidor MCP stdio** contra SQLite sintético efímero, separado de la memoria del usuario y del producto. Incluye original frozen zero-shot, controles de observabilidad, abstención por evidencia ausente y perturbaciones del filtro de acreditación y de XOR, y auditor sin modelos.

Cada replica ejecuta 1024 programas TRAIN. La memoria se congela por SHA antes de generar los OOD nuevos; no hay ejecución de SRX en TRAIN. Los mundos CONFIRM 29300000+ siguen sellados. No cambiar código, umbrales ni memoria tras OOD, preservar todos los resultados y fallos. Ninguna fusión ni despliegue en producto.