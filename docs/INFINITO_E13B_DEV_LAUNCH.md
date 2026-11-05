# E13-B DEV — primera ejecución

2026-11-05. Prerregistro prospectivo ya publicado en commit 657b9815c3749f54f4411db32b2d4dc4deed1b67, antes del código experimental. Ejecutar exclusivamente la partición DEV con 8 mundos nuevos (r0, 0..7) y cuatro brazos A/B/C/D. No abrir CONFIRM ni los mundos LLM hasta congelar la implementación, el modelo y los prompts.

Validar el contrato de proyectos separados, nulidad de fuentes eliminadas, cambios temporales, procedencia, límites de presupuesto, consulta relacional y coste. Auditar desde JSON sin depender del generador. Las pruebas E13-A originales siguen intactas. Sin datos personales, sin producción, sin escribir en master.

## Corrección técnica antes de generar ningún mundo DEV

El primer run 42018382240 ejecutó correctamente contratos y exportó el modelo, pero falló al importar el módulo de investigación cuando se invocó por su ruta directa (ModuleNotFoundError: observatory). El fallo ocurrió ANTES de generar un solo caso DEV; no abrió ni consumió conjuntos. Solo se sustituyó el comando CLI por python -m observatory.e13b; ninguna política, semilla, métrica, hipótesis ni fuente cambió. Esta edición lanza el primer DEV ejecutable.
