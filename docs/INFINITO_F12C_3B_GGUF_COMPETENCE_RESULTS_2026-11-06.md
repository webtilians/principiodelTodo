# INFINITO F12-C — Qwen2.5-3B GGUF congelado: 4/6, NO-GO por estrato
**Fecha: 2026-11-06.** Estado de infraestructura **SUCCESS**. Estado preregistrado de competencia: **NO_GO_FOR_MEMORY_STUDY**. No hay aprendizaje, memoria INFINITO, fine-tuning ni prueba científica confirmatoria.

## Procedencia y cronología
1. [Prerregistro F12-C](INFINITO_F12C_3B_GGUF_COMPETENCE_PREREG_2026-11-06.md), commit **9600e578ef074f2a3e117c5a88b1350d72dfab49**, registrado **antes del código y de observar L0..L5**. Se fijó Qwen2.5-3B-Instruct Q5_K_M, seis incidencias nuevas, las mismas seis herramientas y el umbral de **al menos 4/6 y 2/3 en cada clase**.
2. Primera ejecución GitHub Actions [#42150788888](https://github.com/webtilians/principiodelTodo/actions/runs/42150788888) terminó **BLOCKED por pip**: falta una rueda CPU de llama-cpp-python 0.3.16. El paso de inferencia **no se ejecutó**, ninguna tarea se abrió.
3. Se documentó [corrección de dependencia previa a inferencia](INFINITO_F12C_PREINFERENCE_DEPENDENCY_FIX_2026-11-06.md), commit **51946bbca359f3c5765b5204b322752369e81152**, y se fijó `llama-cpp-python==0.3.36` (commit **50130647e33bb5070614942416bcac5fac258ebd**). **Modelo, prompts, física, particiones, casos y gates no se cambiaron.**
4. Segunda ejecución [**GitHub Actions #42150913035**](https://github.com/webtilians/principiodelTodo/actions/runs/42150913035) **SUCCESS**: tests de aislamiento y backend falso PASS, instalación de wheel PASS, inferencia de LLM real PASS, artefacto guardado PASS.
5. Artefacto: `f12c-qwen-3b-q5-frozen-competence`, ID **13472094132**, digest ZIP **sha256:68f33a28883d2adaf01866f546ec0ec62c1bf813d74c2bb43ea620eba62c8174**, con `preflight_report.json` y `summary.json` (retención 30 días).

## Modelo verificado y costes
- Repositorio de pesos: `bartowski/Qwen2.5-3B-Instruct-GGUF`; `Qwen2.5-3B-Instruct-Q5_K_M.gguf`, cuantización Q5_K_M.
- HF revisión inmutable: **f302c64a2269a69fb27b2f9473b362f5bb8e78d8**.
- SHA256 realmente computado de pesos GGUF cargados: **e4180cbad78d8848bd10839e7fd07edd573c4a5b01148c2b3f7ce2763a2a0938**.
- `llama-cpp-python 0.3.36`, llama.cpp CPU, ChatML, `n_ctx=4096`, 4 hilos, `temperature=0`, sin modificación de pesos.
- **29 inferencias reales** (29 acciones en seis tareas), **14.029 tokens de entrada**, **170 tokens de salida**, **124,43 s de inferencia local medida**. No confundir con coste total ni tokens de un proveedor de pago. **29/29** acciones eran JSON válido y estaban en el catálogo.

## Resultado completo por caso
| ID | Tipo privado al actor | Secuencia observada de acciones | Éxito |
|---|---|---|---|
| L0 | Autorización | audit_access → audit_dependency → repair_access → audit_access → audit_dependency → repair_dependency | 0 |
| L1 | Dependencia | audit_access → audit_dependency → repair_dependency → verify | **1** |
| L2 | Autorización | audit_access → repair_access → audit_dependency → repair_dependency → verify | **1** |
| L3 | Dependencia | audit_access → audit_dependency → repair_dependency → verify | **1** |
| L4 | Autorización | audit_access → audit_dependency → repair_access → audit_access → audit_dependency → repair_dependency | 0 |
| L5 | Dependencia | audit_access → audit_dependency → repair_dependency → verify | **1** |
| **TOTAL** | 3 autorización + 3 dependencia | **29 acciones** | **4/6** |

Resultado por clase: **autorización 1/3**, **dependencia 3/3**.

### Análisis de errores sustentado por las observaciones
- En L0 y L4, después de `audit_access`, el modelo recibió **“access audit: a problem was detected in this subsystem.”**; tras `repair_access`, recibió **“Repair applied successfully. Recovery must be verified.”** Sin embargo, continuó con otras auditorías y trató de `repair_dependency` al final. Nunca ejecutó `verify`; por tanto la reparación no se consideró éxito. **Estos dos fallos fueron de seguimiento y terminación, no de incapacidad de ejecutar la reparación correcta.**
- L2 completó verificación tras una reparación de acceso y una tentativa innecesaria de reparación de dependencia (rechazada).
- L1, L3 y L5 fueron resueltos por el modelo al auditar dependencia, repararla y verificar.
- En los seis casos se ejecutó la reparación del subsistema realmente averiado al menos una vez, pero el criterio exige **verificación positiva**, no inferir éxito por haber ejecutado una reparación.
- Las cadenas de acciones L0/L4 resultaron idénticas en estos dos casos particulares; no interpretar 6 casos como evidencia suficiente de generalización en producción.

## Gates preregistrados
| Gate | Criterio | Resultado |
|---|---|---|
| G0 | LLM auténtico, identidad y SHA de pesos, todas las decisiones atribuibles | **PASS** |
| G1 | >=4/6, con >=2 éxitos en **cada** clase | **FAIL** (4/6, pero acceso 1/3) |
| G2 | >=90% acciones JSON válidas y reacción al feedback | **PASS** (29/29) |
| G3 | Trazas completas, costes, información pública y pesos registrados | **PASS** |
| G4 | Sin memoria/aprendizaje, DEV/CONFIRM reservados | **PASS de contrato** (no hay prueba externa del sistema en producción) |

**Decisión fijada ANTES de resultados: NO-GO para un estudio de aprendizaje con memoria sobre este modelo y este contrato.** Haber alcanzado 4/6 en total no suple 1/3 en autorización. No ajustar umbrales o prompts retrospectivamente.

## Lectura científica limitada y próxima hipótesis posible
- El 3B demuestra mayor competencia descriptiva en **este** banco que los pilotos previos, pero **no existe un A/B limpio** respecto a F12-A (0,5B) o F12-B (1,5B): se cambiaron los casos y la cuantización/modelo. No atribuir diferencias a número de parámetros de forma causal.
- La debilidad dominante ahora es **disciplina de finalización y verificación de una acción con éxito observado**, más que el reconocimiento de una falla o incapacidad para usar herramientas.
- Un modelo 7B cuantizado o un servicio remoto realmente verificado permitiría estudiar un agente más competente, con nuevos casos y preregistro separado. Alternativamente, una investigación autónoma separada podría probar si un LLM aprende a monitorizar subobjetivos a partir de evidencia, pero eso sería **aprendizaje**, no rescatar retroactivamente esta prueba.
- No se ha actualizado la memoria de producción, ni `src/infinito3`, servidor MCP, datos personales, ni se ha abierto ninguna partición F11 DEV/CONFIRM. PR de investigación en borrador, no fusionar automáticamente.

**Reproducción:** `python -m unittest tests.test_f12c_gguf_competence -v`, después `python -m observatory.f12c_gguf_competence --out f12c_competence_artifacts` en el workflow versionado.
