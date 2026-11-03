# INFINITO F10-M — Manifiesto permanente de 16 checkpoints originales SHA-256

**Registro de archivo POSTERIOR al experimento, no preregistro.** Hashes extraídos del artefacto original `f10m-original-independent-fixed-audit/f10m-fixed-science.json` del [GitHub Actions #41707618978](https://github.com/webtilians/principiodelTodo/actions/runs/41707618978), cuyos SHA fueron cotejados con los archivos `.pt` de los cuatro jobs de entrenamiento en el auditor independiente. Publicar estos hashes no añade ningún entrenamiento ni altera los resultados congelados. Criterios originales: [prerregistro previo a los datos](INFINITO_F10M_AUTONOMOUS_PROGRAM_DISCOVERY_PREREG.md).

| Réplica | Controlador | SHA-256 de checkpoint |
|---|---|---|
| 0 | `global_bandit` | `e84188d289a352599f429fbf0757e8fdcb94cab07bd75b6e3ccf2e608a9b1e5c` |
| 0 | `context_mlp` | `f3d073cacfae325ae77bd0d46716c862416fae9959087e468fac1c8edffa33df` |
| 0 | `factorized` | `4e4e677a2d61aa1a81e1840307c5b67b603dc6c67dc9648dae91533cdc8f15e0` |
| 0 | `paired_factorized` | `44e6c2d65ad83ab1dc3d830c8438de55284633ccba0c93e9a5786639adc9eef7` |
| 1 | `global_bandit` | `23eb03b999b6acd73c72a0504fdf220768ee8d1cf8d3339a55d8dec83fadd7a7` |
| 1 | `context_mlp` | `06f09515b53d1912b98465eae2546409518ce4d3922c9e3d2af91ddca0a0c178` |
| 1 | `factorized` | `a7802ff91b060851d7e53bc2afcd708bf029b9001a7da6f38da8b21f5760bfbe` |
| 1 | `paired_factorized` | `7b6952324ae78ff9de77d7ba93774c1b015073dfa60f9de47cebe9dad039cd27` |
| 2 | `global_bandit` | `9821d35e018d3e9b66f1243f226b2963a7bc3baaa5b93f636739aa842c7f45fe` |
| 2 | `context_mlp` | `d2dfee8a91c79c222cd54668888395eed456a2b708528661a94f0b63d941721b` |
| 2 | `factorized` | `5acbca220c3072d87800a57cb01232cceca07e620d5d080ac4e02cecca01e15c` |
| 2 | `paired_factorized` | `157e5148a3fc745d6c427c666451cb2033c8d9d365d3c6a04840cfbabe6f934e` |
| 3 | `global_bandit` | `534137d72a280f446806afa4c2330a95d114ea5c478dbfe9da7ae8d341ecf95e` |
| 3 | `context_mlp` | `72dc24687fc584383af0685998ea61fafa36d6415f046a80bfbfc5bb46e6a4ad` |
| 3 | `factorized` | `2b7eb914e66df088fa11dc213e803cc02e7b55fb2532c4366fae3b301fe4eb10` |
| 3 | `paired_factorized` | `2c779169c10d533ccef6b9d9ed0895646ede24dc8536d1788fcc6fe1b97c2187` |

Las habilidades utilizadas por cada controlador provienen exclusivamente de los modelos `sel_uniform_pair` + `xor_two_actions` originales de F10-L, validados contra el manifiesto G0 del run **#41685499939**. Sus SHA están archivados en [manifiesto F10-L](../docs/INFINITO_F10L_FROZEN_CHECKPOINT_SHA_MANIFEST.md). **F10-M tuvo resultado científico FAIL**; no sustituir checkpoints tras inspeccionar OOD. Archivos de matrices de predicción y curvas completos en los artefactos de Actions (sujetos a caducidad). No se modifica `master` ni producto.
