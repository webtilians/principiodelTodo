# F10-S-FIX — ejecución original de la hipótesis de bloqueo inmediato

2026-11-04. Protocolo prospectivo commit `733000c019d4712b1c1e63b56210af5c96623ff9`, publicado DESPUÉS del hallazgo F10-S FAIL y ANTES de implementar la modificación `BEGIN IMMEDIATE` o de abrir cualquier caso 32500000+.

Este archivo dispara el workflow `infinito-f10s-fix-comparison.yml`: cuatro réplicas, cada una con dos versiones del producto (original y parche aislado), con dos procesos MCP reales compartiendo una SQLite temporal POR BRAZO, 24 casos con exactamente los mismos seeds nuevos. Auditor sin importar MCP/SQLite y gates congelados. Las semillas 32700000+ de confirmación quedan selladas.

El cambio se limita a empezar una transacción de escritura SQLite antes de leer la versión activa del hecho exclusivo, con read+supersede+insert atómicos entre procesos; otros archivos del producto no se cambian. Sin datos de usuario, sin MCP remoto, sin bóveda ni despliegue. No incorporar a master automáticamente. Reportar FAIL o PASS según gates sin ajustar tras OOD.