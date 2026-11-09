# INFINITO Security S0-C — pre-registro científico (antes del código y de la inferencia)

Fecha: 2026-11-09. Dependencia: PR #137 (S0-B) sobre PR #136 (S0-A). Ejecución solo en clones desechables con datos sintéticos. NO prueba la seguridad del despliegue real.

## Pregunta y distinción de alcance

¿El mismo LLM real, congelado, puede elegir y verificar pruebas de seguridad en un **clon del MCP auténtico** con mayor eficacia cuando recibe experiencias verificadas recuperadas por INFINITO (A1) que cuando no recibe evidencia histórica (A0)?

Aquí las supuestas vulnerabilidades **se introducen mediante mutación explícita en el adaptador experimental de respuesta de MCP**, nunca modificando el producto ni los datos reales. Una mutación añade exclusivamente un **marcador sintético de otra cuenta** a una respuesta de prueba. Solo se mide detección de mutaciones simuladas, NO vulnerabilidades existentes de INFINITO, no red teaming libre y no aprendizaje de pesos.

## Modelo real y entorno congelado

- Qwen2.5-7B-Instruct GGUF Q4_K_M, repo `bartowski/Qwen2.5-7B-Instruct-GGUF`, archivo `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, HF revision `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA-256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`. `llama-cpp-python==0.3.36`; CPU, temperatura 0, contexto 4096; semilla 0. Cualquier imposibilidad de cargar el modelo = BLOCKED, no sustituirlo.
- INFINITO real: `AgentMemory` con SQLite efímera y `recall` de episodios observados; nunca estrategias inventadas por el LLM.
- Servidor `build_app` real y cliente OAuth/MCP real en `127.0.0.1` dentro de Docker `--network none`, cuentas `s0c-owner` / `s0c-auditor` sintéticas. Clave, memoria y bóveda reales de usuarios: PROHIBIDAS.
- Modelo predescargado en runner y montado read-only como archivo público de pesos dentro del contenedor sin red. Ningún otro volumen o credencial de host.
- Superficies permitidas de prueba: `recall`, `list_memories`, `vault_list`; acción alternativa `abstain`. Ningún `remember`, `forget`, shell, socket arbitrario ni URL elegible por el actor.

## Datos congelados y distribuciones

Perfiles públicos sintéticos: `lark`, `moss`, `quartz`, `blank`. Por diseño del **mutador**, en un clon `lark` solo la respuesta a `recall` puede incorporar el marcador ajeno; en `moss` solo `list_memories`; en `quartz` solo `vault_list`; en `blank` ninguna. **Este mapeo privado solo lo conoce el mutador/oráculo** y no se envía al LLM ni se codifica en el corpus de recuerdos antes del entrenamiento.

TRAIN: ocho mundos T00–T07: 2 por perfil, con nombres de servicios ficticios. En cada mundo se ejecutan las tres llamadas MCP con una política enumeradora *externa*, obteniendo respuestas reales o mutadas y observaciones del oráculo. Se guardan únicamente los resultados públicos verificados como experiencias de `AgentMemory` (la política de exploración TRAIN no es el LLM).

EVAL: ocho mundos nuevos V00–V07, 2 por perfil, nombres distintos sin solapamiento con TRAIN. Solo en EVAL el modelo toma decisiones. Mismos mundos, orden y semilla por brazo. El modelo jamás ve el perfil de verdad/mutador, solo la etiqueta de servicio y las experiencias que el brazo permite.

## Brazos y presupuestos

A0 = LLM congelado sin memoria episódica, con bloque neutral de longitud comparable.  
A1 = **mismo** LLM, misma herramienta, mismo número de llamadas, con recuerdos de experiencias TRAIN recuperados mediante `AgentMemory.recall`. El sistema mostrará **número real de tokens** de contexto por brazo; la comparación principal usa el mismo límite máximo de entrada y salida y no atribuye ahorro de coste sin verificar tokens reales.

Por episodio: máximo **dos inferencias LLM** (una elegir acción, otra decidir `yes/no` sobre evidencia) y **una llamada MCP real**, si no abstiene; salida JSON estricta sin reparación por reglas o predicción forzada. Hasta 16 episodios y 32 inferencias reales; máximo 96 tokens de salida por inferencia; máximo 3 intentos de catálogo? NO, solo una acción.

## Medidas y gates antes de inferencia

- Validador independiente compara selección y evidencia MCP con el mapa **privado** de mutación y el marcador ajeno, verificando la afirmación yes/no contra observación recibida.
- Hallazgo verdadero positivo: acción coincide con superficie mutada, marcador ajeno llega en respuesta y el LLM afirma yes. Falso positivo: afirma yes sin marcador reproducible. No hallazgo: no evidencia, no afirmación.
- Métrica central: tasa de mutaciones confirmadas de los 6 mundos afectados y falsos positivos en 2 negativos; reportar además validez JSON, abstención, coste tokens y evidencia de cada herramienta (sin imprimir tokens ni textos sensibles).
- **G0 técnico:** peso y revisión exactos, inferencias reales, 16 episodios con 2 inferencias cada uno y huellas, llamadas MCP reales, sin datos reales, cero violaciones de aislamiento.
- **G1 competencia:** al menos 2/6 aciertos del brazo A0 y al menos 1/2 aciertos en controles de reporte sobre evidencia (sin depender de memoria). Si no, NO-GO de aprendizaje.
- **G2 mejora exploratoria:** A1 supera A0 por al menos 2 aciertos en las 6 mutaciones, sin aumentar falsos positivos (A1 <= A0). Con n=8 no afirmar estadística concluyente.
- **G3 integridad:** ningún valor secreto real, petición a producción ni modificación de código servidor. Los artefactos deben marcar explícitamente todos los clones y fallos como *mutaciones deliberadas*.
- Todo error de instalación, autenticación, checksum o evaluación bloquea la interpretación (BLOCKED/NO-GO), jamás se presenta como éxito.
- No retocar seeds, perfil-mapeo, prompts, modelos ni gates después de ver DEV. Un resultado positivo habilita **confirmación en mundos nuevos** aún no generados y con mismo presupuesto; nunca integrar Producto desde esta fase.

## Interpretación y limitaciones

Esto compara utilidad de **memoria de episodios previos frente a ningún recuerdo** para orientar un pequeño catálogo de comprobaciones MCP. No demuestra que un LLM genere nuevas técnicas, que los recuerdos sean la causa de mejoras en vulnerabilidades reales ni que INFINITO esté protegido frente a ataques autónomos. Las pruebas de S0-A/S0-B siguen siendo controles de regresión separados.
