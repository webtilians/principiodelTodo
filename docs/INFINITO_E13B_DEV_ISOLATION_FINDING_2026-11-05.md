# E13-B — resultado DEV temprano: bloqueo por aislamiento de proyectos

**5 de noviembre de 2026. Estado: E13B_DEV_BLOCKED_PROJECT_FILTER; NO HAY RESULTADO científico de recuperación E13-B.** No se generó ninguna muestra CONFIRM ni LLM. No se ha modificado ninguna memoria personal o el producto en master.

## Secuencia reproducible

- Prerregistro E13-B publicado primero en commit 657b9815c3749f54f4411db32b2d4dc4deed1b67, antes del código y las pruebas. Cuatro brazos A actual, B conjuntos, C grafo temporal, D híbrida; aislamiento proyecto/tenant y cero datos resucitados son gates duros G0/G3.
- Primer DEV [run 42018382240](https://github.com/webtilians/principiodelTodo/actions/runs/42018382240) pasó los tests y exportó el ranker, pero abortó antes de generar mundos por una invocación incorrecta de módulo Python. Corrección puramente de CLI a python -m observatory.e13b, sin variar casos o reglas.
- Segundo DEV [run 42018628212](https://github.com/webtilians/principiodelTodo/actions/runs/42018628212) pasó contratos y exportó ranker, pero abortó en la consulta del primer mundo porque la protección de aislamiento detectó que el resultado del producto incluía un recuerdo con un proyecto distinto al solicitado. No se ha puntuado DEV completo ni se ha elegido arquitectura ganadora.
- Se reprodujo el comportamiento aisladamente con solo dos enunciados 100 % sintéticos, sin modelos ni datos privados: [run 42019082859](https://github.com/webtilians/principiodelTodo/actions/runs/42019082859), SUCCESS técnico, resultado semántico **strict_project_isolation=false** y **direct_product_returned_foreign_record=true**.

## Causa observada

El API auténtico de AgentMemory admite recall(query, project=...). Su función auxiliar _is_about(record, project) acepta (1) metadata.project == project exacto O (2) que el valor buscado aparezca como subcadena del TEXTO del recuerdo, incluso si metadata.project apunta a otro proyecto. Ejemplo mínimo:

- Record 1: project=ProyectoNube, content=ProyectoNube: Entidad lima estado: cobre.
- Record 2: project=TENANT_B_ProyectoNube, content=TENANT_B_ProyectoNube: Entidad lima estado: violeta.
- recall('estado actual entidad lima', project='ProyectoNube') devuelve ambos. La salida raw del ensayo lista metadata projects de dos valores diferentes: ProyectoNube y TENANT_B_ProyectoNube.

**Interpretación precisa:** el parámetro project de la versión actual es una pista de relevancia temática, no un filtro estructural estricto ni una autorización multiusuario. En un producto de memoria con varios usuarios/proyectos no puede usarse como única frontera de aislamiento. El experimento E13-B se diseñó para exigir esa frontera; NO se debe eliminar la comprobación para obtener porcentajes artificialmente altos ni presentar una fuga entre cuentas reales: solo se ha probado un filtro cruzado en una misma SQLite sintética.

## Decisión metodológica

**G0/G3 NO verificables como PASS con el baseline actual en este diseño**, de modo que detener la confirmación hasta decidir y validar por separado una política explícita de aislamiento: (a) un filtro exacto por metadata.project, distinguible de una búsqueda temática amplia; y (b) un namespace de usuario autenticado para verdaderos multiusuarios. Hacer esa distinción en una rama correctiva y con un control de no regresión, o limitar el estudio a un scope por proyecto claramente documentado y preregistrar un experimento NUEVO antes de abrir datos. No retocar esta confirmación tras detectar el problema; reservar todas sus semillas.

El hallazgo es sobre el comportamiento real de la interfaz existente, no sobre el rendimiento de B, C ni D. PR #84 permanece DRAFT. Es un descubrimiento útil y concreto sobre la fiabilidad de INFINITO, pero NO un resultado E13-B PASS ni FAIL del mecanismo relacional.