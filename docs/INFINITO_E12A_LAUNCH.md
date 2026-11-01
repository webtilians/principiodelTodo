# E12-A — Lanzamiento único de decisión autónoma de búsqueda

**1 de noviembre de 2026.** [Protocolo preregistrado](./INFINITO_E12A_AUTONOMOUS_SEARCH_PROTOCOL.md) congelado antes de código en commit `7886b0b51eb38be600c41e58a91cc686afb4f32e`. La implementación es un prototipo aislado del núcleo original.

**[CI offline #41324622370](https://github.com/webtilians/principiodelTodo/actions/runs/41324622370): SUCCESS**, regresiones y tests de no fuga de archivo, cuatro clases y controles.

Este fichero es el único disparador del experimento de pago **REAL**: `gpt-5.6-luna` + `text-embedding-3-small`. Primero preflight, luego DEV balanceado de ocho mundos `5100000+` con cuatro brazos, y solo si la procedencia es real y válida, CONFIRM de 32 mundos virgenes `5200000+`, ocho por clase.

Se reportará acierto individual, búsquedas, llamadas/tokens del LLM, abstenciones y la utilidad **acierto−0,08×búsquedas**, ya fijada. Los mundos confirmatorios quedan consumidos al abrirse incluso si la confirmación falla. **No** ajustar prompts, parámetros, distractores o criterio después de DEV o CONFIRM ni repetir el mismo conjunto para obtener un resultado positivo.

El control fuerte de `no_verificado` podría empatar o superar al enrutador. Ese resultado refutaría la necesidad práctica de un controlador de modelo para este entorno y será publicado.
