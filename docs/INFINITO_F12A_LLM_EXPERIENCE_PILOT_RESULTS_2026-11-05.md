# F12-A — Resultados del primer piloto con LLM REAL congelado e INFINITO SQLite

**Fecha: 2026-11-05. Estado técnico: SUCCESS para la inferencia local y persistencia sintética. Conclusión: NO SE DEMOSTRÓ APRENDIZAJE; piloto no apto para evaluación científica.**
No se consumieron DEV/CONFIRM de F11; no se usaron datos de usuario, secretos, memoria MCP de producción ni herramientas reales.

## Trazabilidad y secuencia prospectiva

1. Prerregistro F12-A: `docs/INFINITO_F12A_LLM_EXPERIENCE_PREREG_2026-11-05.md`, commit **04e90b1a1c996645d003febc58feae62650cdd91**, ANTES de código.
2. Primer proveedor GitHub Models, usando permiso `models: read`, no produjo inferencia verificable: [run 42052897453](https://github.com/webtilians/principiodelTodo/actions/runs/42052897453) lanzó JSONDecodeError; el diagnóstico seguro [42053094521](https://github.com/webtilians/principiodelTodo/actions/runs/42053094521) constató HTTP 200 pero `text/plain` de cuatro bytes, no completions JSON. Se clasificó **BLOCKED_REAL_LLM_PROVIDER**; **no** se fabricaron respuestas.
3. Enmienda de proveedor ANTES de cualquier dato nuevo: `docs/INFINITO_F12A_LOCAL_LLM_PROVIDER_AMENDMENT_2026-11-05.md` commit **4b2fe0aa8eae0d0fd577ef5617663ffa6ccabff3**. Modelo abierto local congelado **Qwen/Qwen2.5-0.5B-Instruct**, sin fine-tuning ni secretos. 
4. [**GitHub Actions local Qwen run 42053625649**](https://github.com/webtilians/principiodelTodo/actions/runs/42053625649), commit de lanzamiento **038b4b009ab3170780fd0ffaff22b8aa1d9632a7**, contratos tests SUCCESS, ejecución real CPU SUCCESS y ZIP de evidencias `f12a-genuine-local-qwen-llm-pilot` id **13428650998**, retención 30 días.

## Modelo y condiciones

- Modelo reportado: `Qwen/Qwen2.5-0.5B-Instruct`; **revisión de pesos**: `7ae557604adf67be50417f59c2c2f167def9a775`. `transformers==4.49.0`, `torch==2.5.1+cpu`, `do_sample=False`, mismo modelo y prompts de sistema entre las variantes.
- **41 inferencias reales**: 40 propuestas de acción + 1 síntesis de estrategia. 15.894 tokens de entrada, 815 tokens generados, 172,97 segundos de inferencia medidos en el backend local (no es coste total de ejecución).
- 2 tareas sintéticas TRAIN distintas; 2 tareas sintéticas heldout `P0/P1`, evaluadas cada una con A0 (sin memoria), A1 (memoria episódica INFINITO SQLite), A2 (memoria episodios + hipótesis sintetizada). Máximo cinco decisiones por intento, seis herramientas permitidas. Todas las variantes usaron el mismo modelo y límite de 1200 caracteres de contexto de memoria (no idénticos tokens reales).
- Memoria real del paquete `src/infinito3/persistent_memory.py`: `SQLiteCognitiveMemoryStore(path=":memory:")` con `HashEmbeddingProvider`; dos recuerdos de tipo EPISODIC, uno de tipo SEMANTIC etiquetado como `unverified_generated_hypothesis`. La persistencia SQLite fue **solo en memoria temporal del runner**; no se escribió en el MCP/servidor personal.
- Criterios y casos limitados a piloto. La inferencia remota de GitHub Models bloqueada nunca se combinó con inferencia local.

## Resultados objetivos observados

| Caso | Tipo de falla oculta al actor | A0 sin memoria | A1 episodios | A2 episodios + estrategia |
|---|---|---:|---:|---:|
| P0 servicio de exportación | Dependencia | 0/1 | 0/1 | 0/1 |
| P1 indexación de búsqueda | Acceso | 0/1 | 0/1 | 0/1 |
| **Total** |  | **0/2** | **0/2** | **0/2** |

Experiencias TRAIN: T0 fallo de acceso **no resuelto** y T1 fallo de dependencia **no resuelto**; **0/2 éxitos de entrenamiento**. El estado de las herramientas y la verificación final, no un juicio subjetivo del LLM, determinan el éxito.

Acciones en las ocho trayectorias (2 TRAIN + 6 evaluación, cinco llamadas cada una): **35 `repair_dependency`**, 4 `inspect`, **1 `audit_dependency`**, 0 `audit_access`, 0 `repair_access`, 0 `verify`. El modelo eligió `repair_dependency` incluso después de recibir `Change rejected` o señales de autorización, sin corregir su estrategia. No hubo salida de las rutas de recuperación.

Evidencia concreta:
- T0 (fallo de acceso): `repair_dependency` -> `inspect` (autorización rechazada) -> `repair_dependency` repetido tres veces.
- T1 (fallo de dependencia): `repair_dependency` -> `inspect` (esquema incompatible) -> `repair_dependency` repetido, aunque la herramienta indica que debía auditarse.
- P0 A1: intenta `audit_dependency` una vez, pero vuelve a insistir en `repair_dependency` sin terminar con `verify`.
- A2 propuso `repair_dependency` en los diez pasos de los dos casos heldout; menos flexible que el control A1, pero dos tareas no permiten generalizar.

### Fallo de fundamentación de la estrategia sintetizada

La única síntesis del modelo fue: **"Reinforce the repair of the dependency to ensure the target subsystem is audited before proceeding with the service restoration."** 
No citó ninguna trayectoria concreta, y `strategy_info.source_ids_with_success` estaba vacío. **Ningún episodio había tenido éxito.** Esto es una propuesta especulativa generada por el modelo, **no una política aprendida de evidencias causales validadas**. La infraestructura la guardó deliberadamente con `status=unverified_generated_hypothesis`; el resultado muestra que un mecanismo útil necesitaría exigir evidencia suficiente antes de promover hipótesis a estrategias reutilizables. No modificar retroactivamente el experimento para salvarlo.

El control A2 recibió información de esa hipótesis sin fundamento; su resultado adverso no prueba que la memoria estratégica perjudique universalmente: es incapacidad del modelo y falta de evidencia de entrenamiento en este piloto. Solo se evaluaron dos incidencias, no generalización OOD estructural, no IC95, ni muestra para inferencia.

## Gates prospectivos y decisión

- **P0: PASS técnico**: las 41 respuestas procedieron de ejecución de Qwen real con pesos congelados, no de un FakeDecisionBackend ni reglas codificadas que escogieran acciones.
- **P1: PARCIAL / NO CONFORME**: dos episodios reales de simulación almacenados mediante SQLite INFINITO, pero 0 trayectorias exitosas y la única hipótesis SEMANTIC no está fundamentada por evidencia exitosa ni citas. No demuestra aprendizaje.
- **P2: PASS de registro**: están conservadas las respuestas crudas, retornos, secuencias, estados observados, SHA de prompts, longitudes de contexto y uso del proveedor para todos los brazos, pero no debe tratarse como protocolo científico completo.
- **Resultado global: PILOTO TÉCNICO EJECUTADO; APRENDIZAJE NO DEMOSTRADO / NO-GO científico.** No promover estrategias a producto ni abrir DEV/CONFIRM.

## Próxima decisión que realmente cambiaría la investigación

El límite observado es **competencia mínima del agente y calidad de la experiencia**, no evidencia de que memoria estratégica sea inútil. Para estudiar aprendizaje acumulativo se necesitará primero un LLM capaz de completar tareas y recopilar éxitos/fallos útiles en un piloto con herramientas simuladas. Antes de gastar en un nuevo estudio:
1. preflight de uso correcto de herramientas (al menos *algunas* tareas completadas), con modelo robusto y congelado;
2. políticas solo cuando hay soporte trazable en experiencias observables, no producirlas automáticamente a partir de fracaso total;
3. controles con mismo LLM, mismo número de interacciones y presupuestos de contexto comparables;
4. tareas nuevas suficientemente variadas; incluir cambios de reglas donde memoria antigua pueda estar equivocada.

**No seguir optimizando sobre P0/P1**, ya examinadas. Redactar un nuevo protocolo con nuevos casos antes de inferencia científica. No fusionar automáticamente al producto.

---
El script está en `observatory/f12a_live_llm_experience.py`, tests `tests/test_f12a_live_llm_experience.py`, workflow `.github/workflows/infinito-f12a-local-llm-pilot.yml`. 
