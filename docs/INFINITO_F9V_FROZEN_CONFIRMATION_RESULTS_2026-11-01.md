# INFINITO F9-V — CONFIRM independiente congelada: resultado mixto, gate completo FALLIDO

**2026-11-01.** Resultado permanente de confirmación en nuevos mundos `8300000+` con pesos y código del DEV **congelados**. No hubo reentrenamiento. Cada uno de los cuatro actores F9-P tiene 40 mundos originales/invertidos × 2 regímenes = 160 decisiones confirmatorias (640 en total). Al comienzo hay recorrido docente común hasta el primer cruce; después, continuaciones autónomas con F4 READ o SKIP solo en ese cruce.

## Cadena de custodia y auditoría
- Original DEV (antes de CONFIRM): [Actions #41345497955](https://github.com/webtilians/principiodelTodo/actions/runs/41345497955), 4 réplicas, gate de DEV pasó.
- Confirm 0,1,3: [Actions #41354569501](https://github.com/webtilians/principiodelTodo/actions/runs/41354569501), se ejecutaron una sola vez.
- Confirm índice 2: la primera ejecución abortó **antes de abrir sus mundos** por omisión del último carácter del hash de `lex_public.pt`. [Auditoría](./INFINITO_F9V_RECOVER_SEED2_AFTER_HASH_TYPO.md). Corrigió solamente el hash copiado (no cambió ningún `.pt` ni código F9-V). Se evaluó **solo** índice 2, y se fusionaron resultados originales inmutables mediante [Actions #41354903505](https://github.com/webtilians/principiodelTodo/actions/runs/41354903505), SUCCESS.
- Pesos SHA y SHA del código/protocolo bloqueados en [harness de confirmación](../observatory/f9v_frozen_confirm.py).
- **CONFIRM `8300000+` ya CONSUMIDO**, irreutilizable como holdout para reajustes. F9-T `6300000+`, F9-U `7300000+`, F9 histórico `900000+` y F9-W nuevo `9300000+` no se abrieron.

## Resultado de 640 decisiones

| Método | Primeras elecciones correctas | READ |
|---|---:|---:|
| `lex_hc` (cabeza aprendida con H/C) | **467/640 (73,0%)** | **38/640 (5,94%)** |
| `lex_public` (sin H/C) | 429/640 (67,0%) | 0 |
| `scalar_hc` control de coste dominante | 429/640 | 0 |
| `always` | **480/640 (75,0%)** | 640 |
| `never` | 429/640 | 0 |
| Oráculo de intervención — techo diagnóstico, no desplegable | 497/640 (77,7%) | 68 |

La cabeza H/C ahorra **94,1%** consultas de READ respecto a ALWAYS, y recupera **38 aciertos** adicionales respecto a NEVER (5,94 puntos). Contra ALWAYS pierde **13 aciertos**, 2,03 puntos.

### Contribución causal exacta de las lecturas

Las 640 intervenciones READ/SKIP produjeron **68 efectos positivos (D=+1), 17 negativos (D=−1), 555 neutros**. La cabeza `lex_hc` realizó READ en 38/68 positivos, **0/17 negativos, 0/555 neutros**:
- Precisión selectiva de READ = **38/38=100%** en este conjunto.
- Recall de oportunidades beneficiosas = **38/68=55,9%**. Quedan 30 beneficios no aprovechados.
- La cabeza `lex_public` no realizó ni una consulta.
- Estas métricas son del primer cruce contrafactual con **historia docente anterior**, no competencia general ni navegación completa.

### Réplicas y robustez

| Índice | lex_hc | always | never | READ lex_hc |
|---|---:|---:|---:|---:|
| 0 | 108/160 | 116/160 | 90/160 | 18 |
| 1 | 120/160 | 131/160 | 120/160 | 0 |
| 2 | 119/160 | 113/160 | 119/160 | 0 |
| 3 | 120/160 | 120/160 | 100/160 | 20 |

El gate confirmado se fijó **antes de abrir** `8300000+`: >=90% de los aciertos de ALWAYS, <=80% de sus lecturas, al menos tres de cuatro réplicas con caída <=5 aciertos, y >=3pp sobre cabeza sin H/C.

**Resultado por gate:**
- Preservar >=90% de ALWAYS: **SÍ**.
- Ahorrar >=20% READ: **SÍ**.
- Al menos tres réplicas con caída <=5: **NO** (índices 0 y 1 perdieron 8 y 11, respectivamente; solo índices 2 y 3 cumplen).
- Mejorar >=3pp vs public: **SÍ** (38/640=5,94pp).
- **Resultado final del gate compuesto: NEGATIVO**. No declarar confirmación plena del mecanismo; conservar subefectos positivos como resultado limitado.

### Intervalos de incertidumbre

Bootstrap de **2.000** remuestreos estratificados por actor y bloques de 4 estados de un mismo mundo, fijado antes de confirmar:
- `lex_hc – never`: **+5,94 pp**, IC 95% aproximado **[+4,69, +7,34] pp**.
- `lex_hc – always`: **−2,03 pp**, IC 95% aproximado **[−3,91, −0,16] pp**.
- `lex_hc – public`: **+5,94 pp**, mismo intervalo de arriba porque public y never tomaron las mismas decisiones en este conjunto.

Estos IC describen incertidumbre de muestreo de mundos bajo los cuatro checkpoints evaluados; no cuantifican incertidumbre general sobre entrenamientos futuros ni establecen que el sistema aprenda autónomamente políticas nuevas.

### Papel de las historias F4

Por `history_f4_normal` la cabeza H/C logró 147/160 pares con ambas pistas correctas, y por `history_f4_postcue_off` **0/160 pares**. La capacidad es sensible a la persistencia de F4 durante la historia previa: una lectura tardía no repara sistemáticamente un blackout.

## Juicio científico

La hipótesis fuerte, **robustez entre réplicas**, no ha superado confirmación. Pero la selectividad real de las 38 consultas y los 38 aciertos extra frente a nunca leer es una señal reproducida en este protocolo restringido. No hay soporte para navegación autónoma de extremo a extremo ni para generador de políticas por aprendizaje abierto.

El experimento F9-W de clases positivas se realiza en **nuevas** semillas de TRAIN/DEV. Ningún modelo podrá validarse de nuevo con CONFIRM `8300000+`.
