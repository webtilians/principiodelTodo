# INFINITO F10-N — manifiesto permanente de 12 memorias explícitas congeladas

**Registro archivístico posterior a la evaluación OOD, NO nuevo prerregistro.** Los 12 SHA-256 siguientes fueron calculados al terminar TRAIN y guardados ANTES de abrir DEV/OOD en cada réplica. La auditoría independiente [Actions #41714672031](https://github.com/webtilians/principiodelTodo/actions/runs/41714672031) verificó esos hashes con las copias originales del [run #41714113859](https://github.com/webtilians/principiodelTodo/actions/runs/41714113859). La recuperación técnica NO generó datos nuevos.

| Réplica | Memoria | SHA-256 original |
|---|---|---|
| 0 | `reward_table` | `700fc7367d64595858add978011da18be59802745138a4ee1467e3d9e18be41f` |
| 0 | `learned_effects_uniform` | `7f016945e2a58f202927b287428884ad746a82655dcb8706a08e1de2aa304556` |
| 0 | `learned_effects_directed` | `9746316bf18944bd2e3088a064e14ae2ab0b1857ce4fe866abe7e8ad72b276da` |
| 1 | `reward_table` | `cbae00542809f925647f86d93222ffbb3f3543faf2918624c17153d267a7be55` |
| 1 | `learned_effects_uniform` | `c8250050cbec1f01fd452fc6ca016871cffed589abf0b5f9bc6047e9f3711be6` |
| 1 | `learned_effects_directed` | `e23648cfa2d8ce13cc8e35f85d76abfd4e97de97499656c4e8de37471647bce1` |
| 2 | `reward_table` | `040b14c73e3479b278c9ba23905f1e47bfc398b74e2d488489dbccc3da9fccf2` |
| 2 | `learned_effects_uniform` | `5b6dd34afcaa192d27fadfe6027538d7f304f2b3b0e4846bb9e8dcff6e6b222c` |
| 2 | `learned_effects_directed` | `f1ea5a8889a2fb29864d9d52649af9054f72f762f374972d69f19e6d58f55854` |
| 3 | `reward_table` | `505c40e9ad7ed4777e8a9c1c677af4c77f825b9d04e1183bffec177e007f7d09` |
| 3 | `learned_effects_uniform` | `4ac6e9dc96956440ac23ad81d75781ebdbcb3be1903474ad61e2eaa27e623dbb` |
| 3 | `learned_effects_directed` | `914275445bc785dcf4a554628e2bd4b47e8ea33c990d99f3ab0ea6086508033e` |

Las condiciones A/B **comparten la misma memoria de recompensas**, mientras C y D poseen sus propias memorias de transiciones. El SHA del log uniforme de 98.304 episodios/r es común a A/B/C; el log dirigido de 98.304 a D. Todos los archivos originales de entrenamiento, las predicciones congeladas y los JSON de auditoría están en los artefactos de Actions; estos SHA permiten identificar futuras copias exactas de las memorias si caducan los artefactos. Las cinco políticas originales de F10-L por réplica permanecieron idénticas y pueden verificarse en [manifiesto F10-L](INFINITO_F10L_FROZEN_CHECKPOINT_SHA_MANIFEST.md).

**Estado científico original: F10N_EFFECT_MEMORY_COMPOSITION_PASS**. SRX tuvo cero ejecuciones TRAIN y fue propuesto por C y D 2.048/2.048 veces en OOD. Esto no equivale a confirmación en semillas independientes; F10-N CONFIRM 26300000+ no abierta.
