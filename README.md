# INFINITO

Repositorio de investigación sobre memoria, estado recurrente y control de agentes.

**Rama de referencia: `master`.** La integración reúne el núcleo INFINITO 3.0,
el laboratorio y la cadena experimental hasta **F9-R**, además de las dos
aportaciones F9-I. Integrar código no convierte resultados exploratorios en
conclusiones científicas confirmadas.

## Empieza aquí

| Buscas | Entrada |
|---|---|
| Último experimento: F9-R | [Resultados y límites](docs/INFINITO_F9R_RECURRENT_STATE_DRIFT_RESULTS_2026-10-30.md) |
| Todos los experimentos | [Índice de protocolos y resultados](docs/EXPERIMENT_INDEX.md) |
| Ramas y PR anteriores | [Guía de integración e historial](docs/REPOSITORY_GUIDE.md) |
| Núcleo cognitivo INFINITO 3.0 | [Arquitectura](docs/INFINITO_3_ARCHITECTURE.md) |
| Portal del laboratorio | [Instalación y uso](lab_portal/README.md) |
| Documentación anterior | [README conservado íntegramente](docs/history/README_BEFORE_UNIFICATION_2026-10-30.md) |

## Estado científico actual

F9-R estudia la deriva del estado recurrente cuando se interrumpe F4.
Congelar H/C no alcanzó el criterio primario de rescate; restaurar el estado
natural justo antes de la bifurcación produjo recuperación parcial y
heterogénea. La llegada al cruce usa un prefijo docente: no demuestra autonomía
completa. Consulta el informe para las condiciones, denominadores y controles.

Las confirmaciones F9-R `850000+` ya están consumidas. El conjunto histórico
`900000+` sigue reservado. Las pruebas de integración no vuelven a ejecutar
confirmaciones ni entrenamientos científicos.

## Estructura

- `src/infinito3/`: núcleo cognitivo 3.0.
- `src/`: componentes históricos conservados.
- `observatory/`: experimentos y auditorías de memoria.
- `lab_portal/`: servidor y herramientas del laboratorio.
- `tests/`: pruebas técnicas.
- `docs/`: protocolos, informes e inventario del repositorio.

## Cómo continuar

Crea las nuevas ramas de trabajo desde `master` y abre los PR contra `master`.
Mantén cada experimento con su protocolo, pruebas y resultados identificados.
Las ramas antiguas se conservan como referencias; que estén por detrás de
`master` después de integrar su trabajo es normal y no significa pérdida.

La integración se valida en GitHub Actions con pruebas del núcleo, contratos
F9 y comprobación de historial/contenido. No despliega ni ejecuta modelos con
servicios externos. La [guía del repositorio](docs/REPOSITORY_GUIDE.md)
describe qué ramas se han integrado y cuáles conservan variantes independientes.

[Licencia MIT](LICENSE).
