# F9-Y — Lanzamiento TRAIN/DEV con hipótesis congelada

Fecha 2026-11-02. **Prerregistro anterior a cualquier nuevo mundo**, commit `8555f8e720c31d25731d9ae1848bc2bbd393b9b3`, [protocolo](./INFINITO_F9Y_CONDITIONAL_VALUE_PREREG.md). Las modificaciones posteriores consisten en implementación, tests y CI; NO pueden cambiar el gate científico sin etiquetarse como desviación.

La única ejecución permitida en esta fase utiliza cuatro actores F9-P originales, 128 mundos nuevos TRAIN `10100000+10000*index+j` y 32 mundos nuevos DEV `10200000+10000*index+j` por actor, cada uno con 2 cues y 2 historias. Total 2048 estados TRAIN y 512 DEV. Entrenamiento de modelos originales `deploy_hc`/`deploy_public`, modelos diagnósticos `controlled_hc`/`controlled_nuisance` y 4 entrenamientos *leave-one-combination-out* de ambos controles. **Los indicadores cue/historia diagnósticos son privilegiados**; no son una política operativa.

El criterio se aplica sin elegir hiperparámetros por DEV. Mantener cerrados `10300000+` y todos los holdouts previos: F9-V `8300000+`, F9-W `9300000+`, F9-X `9600000+`, F9-T `6300000+`, F9-U `7300000+`, F9 histórico `900000+`.

Esta publicación activa **una sola** ejecución del workflow `infinito-f9y-conditional-value.yml` y no modifica F9-P ni F4.
