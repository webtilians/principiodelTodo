# INFINITO F10-C — Resultado DEV: precedencia acreditada con cinco informes contradictorios

**2026-11-02. DEV exploratorio terminado. Gate científico: `F10C_DEV_FAIL`.** Protocolo preregistrado **antes de escribir el generador de datos**: [`INFINITO_F10C_VERSIONED_RELATIONAL_PREREG.md`](./INFINITO_F10C_VERSIONED_RELATIONAL_PREREG.md), commit `2bd7393223b3c917746769f4264e39c57ecd420b`. [GitHub Actions #41447171697](https://github.com/webtilians/principiodelTodo/actions/runs/41447171697) **SUCCESS técnico**: cuatro trabajos de TRAIN/DEV y agregado científico independiente, modelos y JSON completos archivados en artifacts. [Run inicial #41447020599](https://github.com/webtilians/principiodelTodo/actions/runs/41447020599) FAIL **antes de generar mundos**, por nombre de una clave de compatibilidad (`latest_unsigned`) al reutilizar la intervención de credencial de F10-B; se corrigió el alias en `e989651cec2456955ca6d1309beb17db66519f70`, sin alterar diseño, modelo ni semillas. Solo la segunda ejecución abrió TRAIN y DEV.

## ¿Qué se probó?

**Familia nueva**: 24 claves y 60 observaciones públicas por mundo; cuatro consultas finales sobre *exactamente la misma* historia y seleccionadas solo al terminar. La clave consultada tiene **cinco informes**: uno firmado inicial y cuatro posteriores (acreditados/no acreditados, contradictorios, con fuente y timestamp públicos). Casos: `signed|stable`, `unsigned|stable`, `signed|revised`, `unsigned|revised`. Verdad = contenido del último informe firmado, calculable solo de datos públicos. Las cinco posiciones para cada clave y **los patrones de credenciales/contradicciones por celda son plantillas sintéticas específicas**; ésta es una limitación importante para la transferencia.

Cuatro réplicas × **512 mundos TRAIN por réplica** (`14100000+`, 2048 casos por réplica, 12 épocas, batch64, Adam0.002). DEV disjunto `14200000+`: **64 mundos por réplica** y cuatro consultas por mundo, 256 casos por réplica; **256 mundos distintos / 1024 consultas dependientes** en conjunto. CONFIRM `14300000+` no abierto. Ninguna etiqueta futura, resultado READ/SKIP o pertenencia al estrato entra como input del modelo: el matching de clave es un sesgo arquitectónico explícito.

Se entrenaron **tres brazos nuevos** sobre los mismos TRAIN por réplica:
1. `relational_read`: GRU64 + log de eventos y ranking entrenable por comparación de pares de informes de la clave; no tiene pesos fijados de preferencia por firma o tiempo.
2. `unary_read_control`: la misma arquitectura unaria **original F10-B**, reentrenada sobre la NUEVA familia F10-C (control indispensable de cambio de distribución y volumen TRAIN).
3. `recurrent_only_control`: GRU64 mismo formato público, sin memoria externa.

Además se cargaron, **sin reentrenar ni consultar sus datos**, los dos checkpoints originales congelados `memory_available` y `recurrent_only` de F10-B (Actions #41443064754) con SHA validado respecto de sus artifacts. Esto compara transferencia entre dominios, NO arquitecturas aisladas, ya que los actores nuevos ven un TRAIN y más ejemplos distintos.

## Resultado 1: todos los nuevos actores con READ aciertan el 100%; el comparador relacional NO mejora al unario

| Réplica | Relacional READ /256 | Relacional SKIP /256 | Unario reentrenado /256 | GRU-only nuevo /256 | F10-B READ congelado /256 | Δ READ−SKIP (95% IC por mundo) |
|---|---:|---:|---:|---:|---:|---|
| 0 | **256** | 131 | **256** | 158 | 157 | +48,8 pp [42,6;55,5] |
| 1 | **256** | 137 | **256** | 148 | 161 | +46,5 pp [40,6;52,3] |
| 2 | **256** | 134 | **256** | 151 | 162 | +47,7 pp [42,2;52,7] |
| 3 | **256** | 148 | **256** | 152 | 147 | +42,2 pp [36,7;47,7] |
| **Total** | **1024/1024** | **550/1024** | **1024/1024** | **609/1024** | **627/1024** | **+46,3 pp descriptivo, SIN tratar 1024 consultas como mundos independientes** |

Además, `f10b_frozen_recurrent` logra **488/1024** aciertos. El relacional supera al **viejo** F10-B congelado, pero el **control unario entrenado en los mismos mundos nuevos también logra 100%**. Por ello **no hay ninguna evidencia de ventaja específica de la comparación por pares**: diferencia total 0/1024, diferencia en el estrato `unsigned|revised` 0/256. La comparación de modelos difiere además en número de parámetros y recibe un matching de clave estructural; *no* se puede atribuir el éxito a descubrimiento general de validación de fuentes.

## Resultado 2: condiciones, efectos causales y comparación con checkpoint original

| Estrato | Relacional READ | Unario nuevo READ | F10-B congelado READ | Relacional D+ |
|---|---:|---:|---:|---:|
| `signed|stable` | 256/256 | 256/256 | 249/256 | 129 |
| `unsigned|stable` | 256/256 | 256/256 | 133/256 | 117 |
| `signed|revised` | 256/256 | 256/256 | 241/256 | 111 |
| **`unsigned|revised`** | **256/256** | **256/256** | **4/256** | **117** |

La nueva distribución resuelve completamente sobre estos DEV el anterior punto débil `unsigned|revised` para AMBAS arquitecturas entrenadas con ella, mientras que el F10-B congelado falla muy gravemente ahí (4/256). Este contraste es informativo sobre la **distribución de entrenamiento y patrones**; NO demuestra que la operación pairwise aprendiera una noción mejor o más general de precedencia.

**Causalidad READ del actor relacional en esta tarea y nuevos mundos:** `D_clean=+1` en **474/1024 casos**, `D_clean=0` en **550/1024**, `D_clean=-1` en **0/1024**. Todas las **16 celdas actor×firma×revisión** tienen >=8 positivos y >=8 no positivos sobre 64 mundos por celda. Los cuatro intervalos bootstrap por mundo para READ−SKIP son positivos; en 4/4 réplicas READ aventaja en >5 pp al GRU-only nuevo.

**No se aprendió a *decidir READ*:** el experimento fuerza consulta limpia. La ventaja causal de F4 está demostrada solo para este actor y esta tarea sintética.

## Resultado 3: ablations retrospectivas incluidas en el prerregistro, NO pruebas generales de autenticación

Intervenciones del mismo relacional con historia GRU inalterada:
- `READ_stale`, retirada de posteriores actualizaciones firmadas: aciertos **828/1024 = 80,9%**.
- `READ_forged`, alteración exógena del token público de credencial del último informe sin firma: **896/1024 = 87,5%**.
- Ambas alteraciones son **ataques de observador fuera de distribución**; sus efectos son muy heterogéneos entre réplicas: en 0 y 2 la caída por stale es cero, mientras en 1 y 3 es grande. Esto limita cualquier interpretación del mecanismo aprendido y su dependencia estable de la firma/versión.

El conjunto DEV es un mismo mundo base con 4 consultas; sus resultados no son 1024 mundos independientes. Las plantillas y 5 registros por clave podrían facilitar atajos a ambos modelos, incluso sin aprender una regla abstracta de vigencia.

## Resultado formal de preregistro

Agregador independiente `F10C_DEV_FAIL`:

| Puerta | Resultado |
|---|---|
| Integridad; fuente pública; splits, hashes de F10-B; 5 informes y 4 consultas emparejadas | **PASS técnico** |
| >=80% correctos en los cuatro estratos para >=3/4 réplicas | **PASS 4/4** |
| Superioridad de relacional sobre el control unario reentrenado, general y en `unsigned|revised` | **FAIL 0/4** (empatan 100%) |
| Utilidad causal READ−SKIP con bootstrap por mundos y vs GRU-only | **PASS 4/4** |
| Soporte D+/D<=0 en todas las 16 celdas | **PASS 16/16** |

No modificar arquitectura, épocas, criterios o generador contra este mismo DEV. No abrir confirmación `14_300_000+`, no entrenar aún cabeza READ.

## Próxima decisión científica propuesta, NO ejecutada

Antes de proponer F10-D más complejo, investigar en un **nuevo preregistro y nuevas semillas** si variar **número de informes, posición temporal de actualizaciones, distribución de fuentes, identidad de claves y longitud de historia** destruye el rendimiento de ambas arquitecturas. Mantener las políticas F10-C congeladas para un test de transferencia de **forma de tarea**; los resultados de este DEV no deben llamarse confirmación. Alternativamente comparar contra una implementación exacta pública de «último registro firmado» como techo/oráculo *basado solo en evidencia pública* y contra estrategias simples de «último informe», «último firmado» y «mayoría», para distinguir un atajo de una regla aprendida. Es un control de baselines algorítmicos y no sustituye generalización.

**Limitaciones:** entorno completamente sintético con token de firma pública, matching de clave por arquitectura, cuatro categorías/plantillas determinadas y consultas binarias al final; no se ha mostrado transferencia a F4 original MiniGrid, agentes LLM/RAG, ni capacidad para decidir cuándo recuperar. El resultado más importante es un **éxito descriptivo de la tarea nueva sin ventaja demostrada del mecanismo pairwise**.
