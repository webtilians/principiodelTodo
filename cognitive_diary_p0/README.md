# INFINITO Cognitive Diary P0 — día ficticio y reproducible

Experimento **aislado**, sin modelos ni herramientas reales, cuentas, memoria personal, claves, servidor, API ni despliegues. El `supervisor` y sus `hijos` son **actores de una secuencia programada**. Su texto es un ejemplo, no una inferencia de un LLM ni una mejora demostrada.

Desde la raíz del repositorio:

```bash
python -m cognitive_diary_p0 --run --output-dir work/cognitive-diary-p0
python -m cognitive_diary_p0 --replay --output-dir work/cognitive-diary-p0
python -m cognitive_diary_p0 --export --format markdown
python -m pytest -q tests/test_cognitive_diary_p0_v3.py
```

Se generan `journal.json` (cabeceras encadenadas SHA-256 y contenidos por separado) y `diary.md` (cronología). Repetir `--run` produce exactamente los mismos bytes. `--replay` comprueba integridad y que coincida con la secuencia de datos ficticios, sin llamar a un modelo. No hay gastos ni tokens facturados.

**Guion:** tres informes ficticios (dos verificados y uno sin verificar); el observador omite el tercero; el investigador propone una secuencia incompleta, que un árbitro sintético independiente rechaza; el supervisor consulta **las tres fuentes originales**, detecta la omisión, reconsidera una interpretación anterior y propone una pauta para el observador. Esa pauta queda `pending_owner_approval`, sin aplicación.

**Contrato de seguridad de esta simulación:** las cabeceras de auditoría son append-only; los contenidos viven en un almacén separado para que un olvido retire tanto el origen como sus derivados transitivos, reteniendo solamente metadatos e identificadores sintéticos. El hash de cadena detecta alteraciones en el fichero, **pero no ofrece autenticidad frente a un atacante que pueda reescribirlo y recalcular los hashes**. Las huellas SHA-256 de contenido no son privadas ante ataques de diccionario; una implementación real requeriría HMAC protegido, cifrado autenticado, borrado de claves/cachés/copias y una política aprobada para metadatos. Los permisos de usuario/proyecto son solo barreras demostrativas in-process, **no una implementación real de autenticación multi-tenant**. No fusionar este módulo con la memoria real ni darle acceso a claves de usuario.

**Límites:** solo prueba contratos y trazabilidad, no utilidad cognitiva. P1 tendría que validar trabajo de LLM real en condiciones ciegas, presupuesto limitado, contenido sintético, selección de evidencias, gobernanza y protección contra prompt injection. El servidor de INFINITO no posee claves de memoria personales desbloqueadas entre sesiones: esta prueba no intenta saltar esa frontera.
