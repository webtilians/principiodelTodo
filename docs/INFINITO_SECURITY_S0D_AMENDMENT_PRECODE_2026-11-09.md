# S0-D — enmienda prospectiva de diseño D2 (antes de código e inferencia)

Fecha: 2026-11-09; sucede al protocolo `INFINITO_SECURITY_S0D_PROTOCOL_2026-11-09.md` commit `f62354f`. **Este documento está comprometido antes de implementar S0-D o consultar cualquier respuesta del LLM.** G1 (D1), modelo, orden y perfiles DEV no cambian.

## Motivo

El protocolo inicial proponía que TRAIN contuviera los **ocho pares** (perfil,revisión) que aparecerían en DEV. Eso hace que el control A2 de lookup exacto acierte 6/6 por construcción y, por tanto, que el criterio comparativo **A1 > A2** sea matemáticamente imposible. Corregimos esa falta de identificabilidad antes de generar observaciones, sin alterar el criterio de éxito.

## Revisión congelada de TRAIN (8 mundos, 24 llamadas MCP, 8 recuerdos)

T00 amber/R1, T01 amber/R2, T02 cedar/R1, T03 indigo/R1, T04 quiet/R1, T05 quiet/R2, T06 cedar/R1 (servicio réplica), T07 indigo/R1 (servicio réplica).

**Pares retenidos:** cedar/R2 e indigo/R2. DEV D00–D07 sigue abarcando **cada uno de los ocho pares** una vez en servicios totalmente nuevos. El mapa privado de respuestas mutadas sigue exactamente la tabla del protocolo original.

A2 lookup construye un diccionario de los pares realmente observados, y para un par no observado utiliza el resultado del perfil en su revisión observada más reciente; si no hay perfil conocido, `recall`. No utiliza la tabla privada ni transformaciones de revisión preprogramadas. El LLM A1 dispone de experiencias TRAIN y puede intentar inferir la regularidad R1→R2 a partir del par amber sin que se le proporcione la respuesta correcta de los pares retenidos. A2 puede acertar hasta **4 de las 6 mutaciones** por exact match; las dos combinaciones nuevas permiten una comparación exploratoria A1>A2 no trivial.

Se mantienen los gates G0, G1, G2, G3, G4 del protocolo original, con esta salvedad prospectiva de cobertura de TRAIN. No se modificará tras ver los resultados; los dos pares retenidos quedan consumidos como DEV y no se podrán utilizar como confirmación independiente.

## Salvaguarda epistemológica

La evidencia de solo un cambio R1→R2 no basta para atribuir aprendizaje de reglas generalizables. Un resultado favorable es **solo una señal exploratoria** y requiere replicas preregistradas con más revisiones, reglas cambiantes y mundos no vistos.
