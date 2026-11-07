# INFINITO F12-F — Resultado de la verificación de estrategias contra experiencias realmente observadas

**Fecha:** 6 de noviembre de 2026  
**Estado de ejecución:** GitHub Actions SUCCESS, contrato de modelo congelado y tests PASS.  
**Conclusión:** **PASS de guardarraíl técnico en banco controlado**, **NO evidencia confirmatoria de aprendizaje ni transferencia**. No integrar automáticamente en el producto.

## 1. Trazabilidad científica

- **Prerregistro** anterior al código F12-F y a toda inferencia: `docs/INFINITO_F12F_GROUNDED_STRATEGY_VERIFICATION_PREREG_2026-11-06.md`; commit **9fb60b3c231da91cf9352c0f31dccc34af8c4106**.
- **Implementación:** `observatory/f12f_evidence_validator.py`; **contratos:** `tests/test_f12f_evidence_validator.py`; **workflow:** `.github/workflows/infinito-f12f-source-validation.yml`.
- [**GitHub Actions #42207405281**](https://github.com/webtilians/principiodelTodo/actions/runs/42207405281), commit de lanzamiento **277d1086a1964022af10908eb514879f65df668e**. Job de tests de aislamiento y regresión S4/S5: SUCCESS. Job de inferencia real 7B y audit de procedencia: SUCCESS.
- Artefacto `f12f-real7b-grounded-source-validator`, ID **13497776981**, hash ZIP **sha256:ff750b7c3dc66b20833a9f16f7f582db0cdb96ea833cfb0873d2debfef8ae5d0**; contiene `full_audit.json` (todas las trazas y respuestas crudas) y `summary.json`; retención de Actions 30 días.
- Modelo exactamente congelado de F12-D/E: `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, revisión HF **8911e8a47f92bac19d6f5c64a2e2095bd2f7d031**, SHA256 REAL de pesos **65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423**, `llama-cpp-python 0.3.36`, CPU, `temperature=0`, sin fine-tuning. **37 inferencias reales**, **14.131 tokens entrada**, **906 salida**, **659,58 segundos** de inferencia CPU medida.
- Se utilizó **SQLiteCognitiveMemoryStore real de INFINITO** con `HashEmbeddingProvider`, pero `path=":memory:"` efímero, exclusivamente experiencias sintéticas. No se leyeron ni escribieron la memoria de usuario, el servidor MCP, secretos o `src/infinito3`. F11 DEV/CONFIRM siguen intactos.

## 2. Experiencias nuevas y verdad pública disponible

Ocho episodios independientes V0..V7, cuatro perfiles nuevos y sus revisiones públicas R1 y R2. El LLM eligió las acciones de herramientas `inspect`, `apply_delta`, `apply_sigma` y `verify`; la compatibilidad real **nunca** entró en el verificador ni en la memoria. El agente recibió únicamente feedback de cada operación realmente ejecutada. Los ocho episodios culminaron en `verify` positivo.

| Fuente | Perfil | Revisión | Acciones realmente ejecutadas | `verify` |
|---|---|---|---|---|
| V0 | ember | R1 | inspect → apply_delta (aceptada) → verify | PASS |
| V1 | moss | R1 | inspect → apply_delta (rechazada) → apply_sigma (aceptada) → verify | PASS |
| V2 | lunar | R1 | inspect → apply_delta (aceptada) → verify | PASS |
| V3 | harbor | R1 | inspect → apply_delta (rechazada) → apply_sigma (aceptada) → verify | PASS |
| V4 | ember | R2 | inspect → apply_delta (rechazada) → apply_sigma (aceptada) → verify | PASS |
| V5 | moss | R2 | inspect → apply_delta (aceptada) → verify | PASS |
| V6 | lunar | R2 | inspect → apply_delta (rechazada) → apply_sigma (aceptada) → verify | PASS |
| V7 | harbor | R2 | inspect → apply_delta (aceptada) → verify | PASS |

**Importante:** los simuladores codifican reglas sintéticas para generar los resultados de las herramientas, pero **C0, C1, C2 y el LLM extractor solo leen las trazas públicas**. La comprobación de procedencia es **observacional**: no demuestra que la misma operación vaya a funcionar en servicios futuros.

## 3. Afirmaciones que generó realmente Qwen7B

Ocho inferencias independientes generaron ocho propuestas individuales JSON con `source_id`, perfil, revisión, operación, resultado observado y verificación posterior. **8/8** cumplieron el esquema formal; **7/8** afirmaron hechos realmente visibles; **1/8** fue contradicha por el propio episodio citado:

- **V7 (única afirmación falsa):** el LLM escribió `{"source_id":"V7","profile":"harbor","revision":"R2","operation":"sigma","observed_outcome":"accepted","verified_after_operation":true}`.
- **Lo que ocurrió:** V7 ejecutó `inspect → apply_delta → verify`; `apply_delta` fue aceptado y verificado. **`apply_sigma` nunca se ejecutó en V7**.
- El control ingenuo **C0** aceptó la afirmación porque V7 existía; el control de recencia **C1** también porque V7 era R2 y reciente; el validador **C2** la rechazó como **`NOT_IN_OBSERVED_EVENTS`**.

Este caso es una nueva manifestación de la misma clase de error hallada en F12-E: **una cita legítima acompañada de una afirmación causal no respaldada**.

El LLM realizó también una síntesis global del banco. La respuesta fue JSON sintáctico, pero **no cumplió el contrato de proposiciones**: empleó `operation: apply_delta` o `verify` y textos literales en `observed_outcome` donde se exigía `delta/sigma` y `accepted/rejected`. La totalidad de esa salida quedó marcada como **`INVALID_SCHEMA`**, sin remapear ni reparar campos. Por tanto **no se validó una política global coherente**. El resultado 7/8 se refiere exclusivamente a propuestas **por fuente**, no a la síntesis global.

## 4. Controles adversariales — 56 pruebas derivadas de registros visibles

Se predefinieron **7 casos por cada una de 8 experiencias**: una proposición correcta y seis controles por alteración de acción, resultado, versión, ID, verificación o aplicabilidad temporal. Hay **8 positivos** y **48 negativos** según las trazas públicas. Estas etiquetas están construidas por diseño a partir del mismo esquema que utiliza C2; constituyen una **prueba técnica de contrato**, **no un test independiente de inteligencia ni estimación estadística de errores reales**.

| Validador | TP (correctas aceptadas) | FP (falsas aceptadas) | TN | FN (correctas rechazadas) |
|---|---:|---:|---:|---:|
| C0 — cita existente | 8 | **24** | 24 | 0 |
| C1 — cita + revisión más reciente | 4 | **12** | 36 | **4** |
| **C2 — hechos observados, orden de verificación, revisión objetivo** | **8** | **0** | **48** | **0** |

La mera recencia **pierde evidencia histórica cierta**: R1 deja de ser la última versión, pero sigue siendo correcta para tareas R1. La verificación por revisión de C2 permite diferenciar una afirmación histórica correcta de una afirmación aplicable al objetivo R2.

C2 valida que la acción realmente se ejecutó, que la respuesta de la herramienta afirma lo que se cita y que **`verify` sucedió después de una aceptación**. Una aceptación sin verificación queda registrada como observación parcial; una acción rechazada como observación negativa; ninguna se eleva a evidencia de éxito. Una proposición válida para R1 se marca `STALE_FOR_TARGET` si se propone para R2.

## 5. Gates preregistrados, sin ajustar umbrales

| Gate | Resultado | Limitación |
|---|---|---|
| G0 — LLM real congelado, fuentes SQLite públicas, sin oráculo | **PASS** | Verificado por contrato y SHA de modelo; piloto aislado |
| G1 — cero promociones incorrectas | **PASS** | 0 falsos positivos en 56 controles; 7/8 propuestas LLM aceptadas y 1 falsa rechazada |
| G2 — conservar >=95 % positivos, bloquear >=95 % alterados | **PASS** | 8/8 positivos conservados, 48/48 negativos bloqueados |
| G3 — comparar controles C0/C1/C2 | **PASS** | Conteos exactos en registros |
| G4 — informar sobre límites, no declarar aprendizaje | **PASS de alcance** | No se ejecutaron políticas aprendidas sobre tareas nuevas |

**Estado literal de la ejecución:** `TECHNICAL_GUARDRAIL_PILOT`. **Conclusión: PASS técnico para este verificador estrecho de evidencias; no PASS científico de aprendizaje generalizable.**

## 6. Recomendación de próxima investigación

**No integrar todavía el validador ni las estrategias en producción**, porque:
- Su formato exige afirmaciones JSON estructuradas; el propio LLM falló en la síntesis global. El lenguaje natural de estrategias requiere un normalizador auditable previo, que también puede alucinar.
- El contraste adversarial deriva de la misma semántica observable que el verificador, por lo que es en parte tautológico. Los 0 FP/0 FN no pueden extrapolarse a registros reales heterogéneos o maliciosos.
- La aceptación comprueba **fidelidad al registro**, no **validez de transferir la política** de un servicio a otro ni efectos posteriores.
- El actor 7B resolvió 8/8 casos de la familia sintética con hasta cinco acciones; esto no demuestra que la memoria haya causado un mejor rendimiento.

El siguiente estudio **F12-G** debería explorar una arquitectura de dos fases: LLM convierte texto de estrategia en proposiciones auditablemente estructuradas, C2 valida cada una y abstiene/rechaza los hechos sin prueba, y un agente de evaluación decide entre **A0 sin memoria**, **A1 RAG episódico competente**, **A2 hipótesis sin validar**, **A3 hipótesis filtradas por C2**. Requerirá casos/operaciones más variados, revisión R1/R2 obsoleta, iguales presupuestos de contexto y herramientas, replicación independiente y una heurística cronológica sencilla como control adicional. Si las estrategias aceptadas no mejoran a A1, no hay evidencia incremental; si mejoran pero el texto global vuelve a ser inválido, no rescatar manualmente. Nuevo prerregistro antes de acceder a datos futuros.

**No se abrieron DEV/CONFIRM originales F11; no se cambió el producto ni el MCP. PR en borrador.**
