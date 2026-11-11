# F10-S lanzamiento único sobre MCP real

**2026-11-04.** Prerregistro científico publicado en commit `30cc583dd2c29d6b08bcd42c4cf84c5bd89d1dd9` ANTES de código de simulador, pruebas, train o datos; decisión científica bloqueada, sin retoques por ver resultados.

Este archivo activa UNA ejecución del workflow `infinito-f10s-two-agent-coherence.yml` contra producto auténtico en commit `2e563ddb194cb3a0193b77a22cc97575a8baba73`, en 4 réplicas independientes. Por réplica: 2 procesos MCP simultáneos contra un SQLite temporal compartido, 24 casos disjuntos: visibilidad, actualización exclusiva, olvido selectivo, reinicio, concurrencia, revalidación de contexto. Se abren 3º/4º procesos frescos para verificar lectura tras reinicios; auditor separado de los archivos de resultados, sin importar MCP ni sqlite3.

Temporales y nombres de archivo fuera de memorias personales y producción, no se tocará el servicio remoto. Fallo reproducible se informa como FAIL, error técnico como INCONCLUSIVE; reserva `32300000+` intacta, PR #77 sigue draft, no auto-merge.