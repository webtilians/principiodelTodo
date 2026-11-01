# E10-D — Lanzamiento único autorizado

**1 de noviembre de 2026.** Protocolo congelado antes de escribir código en `2ba9a4ae77b50d6155fd78dd91911183e877c974`.

Resultados previos inmutables:
- E10-B con LLM real: RAG semántico 24/24 e INFINITO 2/24, confirmación `1800000+` **consumida**.
- E10-C: auditoría con 12 mundos nuevos, 32/48 recuerdos no capturados en INFINITO original; gate abierto recupera 48/48 y elimina 8/48 en construcción de contexto por colisión multilingüe de «plan».
- [CI offline de E10-D #41312762796](https://github.com/webtilians/principiodelTodo/actions/runs/41312762796) **SUCCESS**: se conserva el núcleo intacto, nuevo gate declarado, corrección plan ES/EN, fixtures no solapados y tests de regresión.

**Este fichero es el único disparador de la comparación LIVE**, con cinco brazos, modelo `gpt-5.6-luna`, embeddings `text-embedding-3-small` y cambios experimentales congelados. La secuencia CI requiere API verificada → **DEV `2100000+` 6 pares** → comprobar todas las respuestas reales y cinco brazos → **CONFIRM `2200000+` 24 pares** una sola vez → reportar valores incluso si el candidato falla.

**No se permiten** nuevas reglas de dominio, tuning del gate o filtros de contexto tras observar DEV/CONFIRM. Las respuestas con error de API no se sustituyen por offline. Si hay fallo después de abrir `2200000+`, considerar la partición consumida.

Si el candidato no supera el **RAG semántico cronológico**, cerrar E10 y continuar E11/E12 con RAG como baseline. No modificar ni fusionar `src/infinito3/` ni abrir holdout F9.
