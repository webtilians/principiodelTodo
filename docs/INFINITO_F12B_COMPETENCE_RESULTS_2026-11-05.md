# INFINITO F12-B — Preflight de competencia de LLM real congelado

**Fecha 2026-11-05. Estado técnico: SUCCESS. Gate científico/operativo de competencia: NO-GO (1/4 tareas).** No iniciar estudio de aprendizaje con memoria sobre este modelo, no abrir DEV/CONFIRM y no modificar los casos K0–K3 tras observar estos resultados.

## Procedencia y condiciones prospectivas

- Prerregistro en `docs/INFINITO_F12B_FROZEN_LLM_COMPETENCE_PREREG_2026-11-05.md`, commit **fa59ed5c50bf120dfb0f7e5e722b6f8845a2be79**, publicado **antes** del código F12-B y de la ejecución de tareas nuevas.
- Rama `experiment/f12b-frozen-llm-competence-preflight-20261105`, PR #99 abierto en borrador contra `master`, sin modificar el producto ni MCP.
- GitHub Actions [run 42092342452](https://github.com/webtilians/principiodelTodo/actions/runs/42092342452), commit de lanzamiento **8e950817d5b78efdf87e10fac9d09c70d05490d8**: contratos de aislamiento y de backend ficticio **SUCCESS**, ejecución del LLM real **SUCCESS**, artefactos con resultados por acción **SUCCESS**.
- Artefacto de Actions: **`f12b-frozen-1p5b-competence-preflight`**, ID **13444989870**, hash ZIP **sha256:cb1ce99694f408d77042156e8771f722306ce182e0c4352c9fa34ec339e277da**, con `preflight_report.json` y `summary.json`.
- Reproducir contratos: `python -m unittest tests.test_f12b_llm_competence -v`; preflight: `python -m observatory.f12b_llm_competence --model Qwen/Qwen2.5-1.5B-Instruct --out f12b_preflight_artifacts`. Librerías pinneadas por workflow. El estudio no utiliza ningún archivo del piloto F12-A como datos.

## Proveedor y presupuesto real

- Modelo **Qwen/Qwen2.5-1.5B-Instruct** (público, local, weights congelados, greedy `do_sample=False`, sin fine-tuning); revisión exacta de Hugging Face: **989aa7980e4cf806f80c7fef2b1adb7bc71aa306**.
- `torch==2.5.1+cpu`, `transformers==4.49.0`; **21 decisiones** atribuidas a inferencia real de LLM; **10.377 tokens de entrada**, **140 tokens de salida**, **170,19 s** acumulados de inferencia del modelo (no es latencia de workflow ni coste total).
- Misma instrucción base, catálogo de seis herramientas y máximo seis acciones en cada una de cuatro incidencias nuevas. `inspect`, `audit_access`, `audit_dependency`, `repair_access`, `repair_dependency`, `verify`. Causa interna invisible al actor; solo observó feedback de las herramientas que eligió.
- Sin memoria ni políticas, sin estrategia experta, sin API externa de operaciones y sin datos personales. Los contratos de herramientas se explican como API pública, no proporcionan la secuencia correcta.

## Resultado preregistrado: fracaso del gate mínimo

| Caso | Categoría privada al actor | Acciones realmente elegidas por el LLM | Éxito |
|---|---|---|---|
| K0 (broker de recibos) | Autorización | audit_dependency → repair_dependency → verify → inspect → repair_access → audit_access | **0/1** |
| K1 (reconciliación) | Dependencia | audit_dependency → repair_dependency → verify | **1/1** |
| K2 (cola de traducción) | Autorización | audit_dependency → repair_dependency → verify → inspect → repair_access → audit_access | **0/1** |
| K3 (normalización) | Dependencia | inspect → repair_dependency → audit_dependency → verify → repair_access → audit_access | **0/1** |
| **Total** | 2 autorización + 2 dependencia | **21 acciones; 100 % JSON con herramienta válida** | **1/4 (25 %)** |

- **G0 (LLM real sin fallback): PASS**. Todas las decisiones las generó el modelo cargado, no un script ni el fake de tests.
- **G1 (competencia ≥2/4 incluyendo cada categoría): FAIL**. Solo 1/4, de dependencia; 0/2 de autorización.
- **G2 (validez ≥90% y reacción al feedback): PASS**. **21/21** herramientas válidas y cambios de acción después de señales/errores.
- **G3 (integridad, aislamiento y registros): PASS**. Cada acción contiene respuesta cruda del LLM, observación, coste y SHA del prompt; pesos identificados por revisión, sin consultar DEV/CONFIRM.

### Diagnóstico causal sustentado por trazas

- **K0 y K2**: el modelo inicialmente auditó dependencia; recibió que estaba saludable. Intentó reparar la dependencia equivocada, pidió verificar (fracasó) y finalmente inspeccionó una pista de autorización. Después intentó `repair_access` **antes** de `audit_access` (precondición pública); llegó a auditar el componente correcto **en el sexto y último paso**, sin poder reparar ni verificar.
- **K1**: siguió correctamente la secuencia `audit_dependency`, `repair_dependency`, `verify`, con verificación positiva en el tercer paso.
- **K3**: inspeccionó una pista de dependencia pero reparó **antes** de auditar; luego auditó la dependencia y pidió verificar **sin volver a reparar**. El simulador rechazó la verificación; las dos acciones finales no resolvieron la incidencia.

El LLM aparentemente interpreta parte del feedback, pero **no planifica de manera suficientemente fiable precondiciones, reparación y verificación dentro del presupuesto**. Esta explicación está limitada a estos cuatro casos, no constituye una estimación general de calidad del modelo.

## Límites y decisión

Este estudio NO compara limpiamente el efecto causal de tamaño 0,5B vs 1,5B: además de cambiar modelo se usaron **cuatro casos nuevos y contratos públicos de herramientas más explícitos**. F12-A no pudo resolver tareas, pero el 25 % aquí NO prueba que la diferencia de parámetros haya causado la mejora. La muestra es únicamente un preflight; no hay validez inferencial, IC ni prueba de transferencia y no se entrenó nada. Tampoco se ha demostrado memoria de INFINITO ni política autónoma.

**Decisión predefinida: NO-GO para estudio de memoria con este modelo.** Preservar K0–K3 como casos consumidos de preflight, no reusar como confirmación. Si se investiga un modelo más capaz (por ejemplo 3B/7B cuantizado o un proveedor remoto competente), usar nuevos casos, nuevo protocolo prospectivo, y mantener la misma prueba de competencia antes de generar estrategias. No modificar F11/F12-A ni integrar cambios en producción.

**Estado final:** piloto con LLM real completado, mejora cognitiva de INFINITO NO estudiada ni demostrada.
