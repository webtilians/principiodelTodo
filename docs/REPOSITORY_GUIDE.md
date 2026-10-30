# Organización y unificación del repositorio

Fecha: 30-10-2026. Rama canónica: **master** (ya era la predeterminada).

## Qué se reúne
La punta F9-R contiene 276 commits posteriores a master y la cadena F9-A–R
(excepto la auditoría paralela #15). Se añaden íntegramente:
- INFINITO 3.0, PR #2.
- Auditoría F9-I de retención, PR #15.
- Cadena experimental hasta F9-R, PR #25, que incluye #16 y sus sucesores.

No se reemplazan módulos con variantes antiguas: las aportaciones laterales son
archivos nuevos y se verifican sus blobs. Se conservan los commits mediante
padres de merge, sin squash ni reescritura. Los resultados científicos y sus
protocolos mantienen sus rutas y contenido.

## PR absorbidos por la integración

| PR original | Rama | Punta conservada |
|---|---|---|
| [#2](https://github.com/webtilians/principiodelTodo/pull/2) | `infinito-3.0` | `3e56a0e2579f2e8a1af70e2fa2a7d0e87faf6e2a` |
| [#4](https://github.com/webtilians/principiodelTodo/pull/4) | `fix/lab-flow-audit-20261023` | `4bcaa2bf6fcaca603f3bd83eb390bea71a7bbcf1` |
| [#5](https://github.com/webtilians/principiodelTodo/pull/5) | `experiment/f9-external-minigrid` | `0ecb2b667e1f217452ad4e23139bc64ff64a5f81` |
| [#6](https://github.com/webtilians/principiodelTodo/pull/6) | `experiment/f9-windows-remote-worker` | `5b2a95a19f988dab8f8bd9be93c2c2ff058ec09c` |
| [#7](https://github.com/webtilians/principiodelTodo/pull/7) | `experiment/f9b-causal-memory` | `a73983028ede8e9e6f1035e3b47af502d69dd8f1` |
| [#8](https://github.com/webtilians/principiodelTodo/pull/8) | `experiment/f9b-behavior-audit` | `5d9927a2263370b29c933f8df69c450aacbe7f37` |
| [#11](https://github.com/webtilians/principiodelTodo/pull/11) | `experiment/f9e-f4-autonomous` | `24f5e83afa5582f0e907547fcb5d20d2d7537a96` |
| [#12](https://github.com/webtilians/principiodelTodo/pull/12) | `experiment/f9f-f4-memory-content-audit` | `11eae1500842a7166eda3ea9abbfe868d711157d` |
| [#13](https://github.com/webtilians/principiodelTodo/pull/13) | `experiment/f9g-context-memory` | `6fd371f4ce8881639c7398dd3bd77c0da17c72a4` |
| [#14](https://github.com/webtilians/principiodelTodo/pull/14) | `experiment/f9h-contextual-f4-autonomous` | `575c4522cdafa7d0f9742fc6456d1a16c46b6a73` |
| [#15](https://github.com/webtilians/principiodelTodo/pull/15) | `experiment/f9i-memory-retention-audit` | `f2b7d9b1c83ac182ca20dda0362f8a13bebe131c` |
| [#16](https://github.com/webtilians/principiodelTodo/pull/16) | `experiment/f9i-selective-event-memory` | `b08f35787d5323ff746c1b4a78aa5f3673014e3c` |
| [#17](https://github.com/webtilians/principiodelTodo/pull/17) | `experiment/f9j-recurrent-memory-attribution` | `d9482a82722d32eb9e4c5c889ccc9cf96991b44a` |
| [#18](https://github.com/webtilians/principiodelTodo/pull/18) | `experiment/f9k-temporal-recurrent-probes` | `2ea34e0ee34ffa0913c1bdf6cf80c70bb374844d` |
| [#19](https://github.com/webtilians/principiodelTodo/pull/19) | `experiment/f9l-f4-interruption-survival` | `baa6176d237a1308fd21e72599d746d59dc775ec` |
| [#20](https://github.com/webtilians/principiodelTodo/pull/20) | `experiment/f9m-temporal-dropout-training` | `344327a8bdeb5ae1cfa9d52e0abcf0fee1ee3c9d` |
| [#21](https://github.com/webtilians/principiodelTodo/pull/21) | `experiment/f9n-autonomy-fork-decomposition` | `8c23f21bbfdfdf6da1daf12fe2ce945c09ceac9f` |
| [#22](https://github.com/webtilians/principiodelTodo/pull/22) | `experiment/f9o-long-f4-blackout-training` | `d79deb9fee6153f173df9a35828208b5f0888abe` |
| [#23](https://github.com/webtilians/principiodelTodo/pull/23) | `experiment/f9p-ppo-sustained-f4-blackout` | `51b66202a18b8cb110a6ec832c3f62ed94a81c6e` |
| [#24](https://github.com/webtilians/principiodelTodo/pull/24) | `experiment/f9q-causal-fork-recall` | `dda0290f3a9e17225206d335e1215b309f3de6df` |
| [#25](https://github.com/webtilians/principiodelTodo/pull/25) | `experiment/f9r-recurrent-state-drift` | `7c66a9205d2903023831e478708cf9f467e40d60` |

## Qué significa una rama atrasada
Una rama experimental conserva la fotografía exacta de su experimento.
Tras integrar sus commits, puede aparecer muchos commits por detrás de master.
Eso es normal: no hay que fusionar master de vuelta en todas las ramas históricas.
El siguiente trabajo comienza desde master.

`main` es una línea histórica diferente, no la rama principal activa; no se
sincroniza ni elimina aquí. Las variantes `freeze/*`, `baseline/*`, `eval/*`,
`feat/*`, `handoff/*` y otras ramas sin PR abierto permanecen donde estaban.
Tampoco se actualiza `deploy/lab-portal-vps`. Integrar variantes mutuamente
alternativas indiscriminadamente podría reemplazar comportamiento posterior.
Todo el inventario con SHA está en
[repository-snapshot-2026-10-30.json](repository-snapshot-2026-10-30.json).

## Conservación y recuperación
- Ninguna rama histórica se borra ni se fuerza.
- Ningún archivo científico se mueve, borra o modifica.
- El README previo está [conservado byte a byte](history/README_BEFORE_UNIFICATION_2026-10-30.md).
  Es una fotografía documental: sus enlaces relativos reflejan la ubicación
  original y algunas afirmaciones históricas están superadas por informes nuevos.
- La prueba de preservación comprueba hashes Git de los archivos integrados y
  que todas las puntas de los PR originales son antepasados del commit de integración.
- Los artifacts de Actions tienen su caducidad original. El inventario de Git
  preserva código e informes versionados, no convierte artifacts temporales en
  archivos permanentes. Los informes conservan los enlaces y SHA de procedencia.

## Pruebas y experimentos
La CI de integración ejecuta tests técnicos, no confirmaciones ni benchmarks
con servicios externos. Los workflows experimentales conservan sus disparadores
históricos; están disponibles para inspección y ejecución manual cuando el
protocolo lo permita. No volver a usar particiones consumidas como confirmación
nueva. En F9-R, 850000+ está consumida; 900000+ sigue reservada.

Para cambios nuevos: rama desde master, PR contra master, tests del módulo
afectado y protocolo separado si cambia la pregunta científica.
