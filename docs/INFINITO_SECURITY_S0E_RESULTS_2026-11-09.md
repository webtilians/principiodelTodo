# INFINITO Security S0-E — resultados de reconocimiento con procedencia autenticada

**Fecha:** 2026-11-09 · **Estado técnico:** COMPLETED / GitHub Actions SUCCESS. **Interpretación:** GO únicamente para diseñar la integración de procedencia autenticada en las herramientas MCP y confirmar con datos nuevos; NO integrar ni desplegar directamente.

## Evidencia reproducible

- Protocolo preregistrado antes del código: commit `48b73f88926d9c6155f7f017a03b7f337c5feb76`.
- PR #146 draft: https://github.com/webtilians/principiodelTodo/pull/146
- GitHub Actions run: https://github.com/webtilians/principiodelTodo/actions/runs/42672534358
- Código: `security_lab/evidence_provenance.py`, `security_lab/s0e.py` y `tests/test_security_lab_s0e.py`.
- Contratos y regresiones: **35/35 pytest PASS** (S0-A/B/C/D/E).
- Qwen2.5-7B-Instruct GGUF Q4_K_M congelado, revisión `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA-256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, `llama-cpp-python==0.3.36`, temperatura 0, 24 inferencias reales: 6.724 tokens entrada, 120 tokens salida, 252,80 segundos de inferencia CPU.
- El clon HTTP OAuth/MCP auténtico recibió **24 llamadas de lectura** en `127.0.0.1` dentro de Docker `--network none`. Cuentas, canarios y credenciales de test ficticios; sin acceso ni cambios a producción.

## Resultados banco E00–E23 nuevo, sin reutilización de R00–R15

| Métrica | B0: LLM sin clave | B1: puerta de procedencia |
|---|---:|---:|
| Positivos deliberados | 8 | 8 |
| Verdaderos positivos | 8/8 | 8/8 |
| Falsos positivos sobre 16 no positivos | **13/16** | **0/16** |
| Especificidad (16 no positivos) | **18,75%** | **100%** |
| Hallazgos con procedencia firmada válida | No puede verificar firma | 8 `EXPOSURE` |
| Evidencia limpia y firmada | Sin estado separado | 8 `CLEAN` |
| Firma/contexto inválido | A menudo interpretado como hallazgo | 8 `UNVERIFIED` |
| Respuestas JSON del LLM inválidas | 0/24 | n/a |

El modelo declaró `yes` en casi todos los casos con contenido sugerente; el verificador solo acepta evidencia con firma válida, solicitud OAuth correcta, herramienta correcta, cuenta, caducidad y propietario ajeno. Las 8 evidencias no verificables son **casos pendientes de investigación**, no pruebas de seguridad o de vulnerabilidad. Las 8 limpias sí son clasificadas `CLEAN` por el contrato sintético.

## Gates

- **G0 PASS:** modelo, 24 inferencias reales, 24 llamadas MCP, clon local, red externa bloqueada y resultados completos.
- **G1 PASS:** 8/8 hallazgos firmados válidos, 0 falsos positivos.
- **G2 PASS:** 8/8 manipulaciones clasificadas `UNVERIFIED`; 8/8 registros propios/limpios clasificados `CLEAN`.
- **G3 PASS técnico por CI:** 35/35 regresiones de S0-A/B/C/D/E correctas. No se modificó código del producto.
- **G4 PASS:** una respuesta de LLM no puede cambiar la decisión del verificador. La firma se compara mediante `hmac.compare_digest`.

## Interpretación y limitaciones

El **componente de procedencia autenticada** funciona sobre este banco de prueba, pero **NO se ha corregido el propio razonamiento de Qwen 7B**: el LLM siguió produciendo 13 falsos positivos. La reducción de falsas alarmas proviene de una decisión independiente y determinista basada en metadatos firmados; la comparación B0/B1 no está equiparada en capacidades y no prueba mejora cognitiva ni aprendizaje.

**MUY IMPORTANTE:** la firma HMAC la genera un emisor externo **exclusivo del laboratorio**, el servidor MCP actual de INFINITO no emite ese contrato. La firma acredita lo que declaró el emisor sintético, no la verdad de una fuente externa. Para implementar producto es necesario diseñar y auditar procedencia emitida desde el servidor (identidad real de solicitud y propietario de cada registro) antes de que se pueda afirmar defensa efectiva en uso real.

No usar este banco E00–E23 para confirmación científica independiente. El siguiente hito debe separar prueba de integración autenticada en un clon real del servidor de una evaluación de transferencia en casos nuevos, sin ajustar tras ver estos resultados. Nunca desplegar automáticamente.
