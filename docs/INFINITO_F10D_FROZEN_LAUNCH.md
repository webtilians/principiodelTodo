# INFINITO F10-D — Lanzamiento de transferencia congelada

**2026-11-02.** El [prerregistro completo F10-D](./INFINITO_F10D_FROZEN_STRUCTURAL_TRANSFER_PREREG.md) se fijó en el commit `f4888f557b7092660fca9c5a0e04520e21b26fd8` ANTES de desarrollar generador/ejecutor y abrir mundos.

Este lanzamiento ejecuta SOLO mundos nuevos `15_200_000+10_000*index+j` con `index=0..3` y `j=0..63`, sin TRAIN de ninguna clase. Los modelos proceden EXCLUSIVAMENTE de los artifacts originales F10-C run #41447171697 y F10-B run #41443064754; se validan SHA de cinco checkpoints frente a los manifiestos originales.

Los nuevos 60 informes públicos por mundo soportan cuatro queries finales sin fuga, una por estrato acreditado/obsoleto, con 3/4/6/8 informes objetivo rotados en cuadro latino y cuatro regímenes de fuente balanceados. Se evalúan READ, SKIP, stale, forged sobre el mismo actor GRU y el mismo historial público; se comparan contra otros actores congelados y baselines exactos construidos solamente de la información pública. Los criterios G1,G2,G3 quedan congelados y un agregado independiente valida las 16 celdas y 4 modos de fuente. No se cambia F10-B ni F10-C.

**NO** abrir `15_300_000+`, confirmaciones previas ni entrenar lector. Aun con PASS, resultados de esta tarea sintética no demuestran transferencia a arquitectura F4 MiniGrid/RAG/LLM.

La publicación del presente archivo dispara `.github/workflows/infinito-f10d-frozen-structural-transfer.yml` una vez.
