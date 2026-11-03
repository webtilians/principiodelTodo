# INFINITO F10-L / F10-L-C — Manifiesto permanente de 20 checkpoints SHA-256 originales

**Registro archivístico posterior a la confirmación, no una nueva preinscripción.** Los hashes que siguen proceden de `f10l-g0.json` del **run original #41685499939**, generado y congelado **antes de abrir OOD y CONFIRM**, y fueron validados byte a byte contra los archivos `.pt` descargados al confirmar en el run **#41699416468**. Código original congelado `b5c32cbfb50a24414607e3a24e30e139ca92fede`. Sin reentrenamiento ni cambios de parámetros.

| Réplica | Brazo | SHA-256 original |
|---|---|---|
| 0 | `xor_onpolicy` | `ce01bb2b793656b1e8815411a709fb26338825585fb100e48fc50696ed300c93` |
| 0 | `xor_two_actions` | `775b6c2f3935c7989a8e30ae31c96eb7775a66068ee83e95c00606ee84f80abe` |
| 0 | `sel_onpolicy` | `2ece44164d8a80656e49df7d73fbcfce093a100f5a28e3023f1674d669a8b8f1` |
| 0 | `sel_uniform_credit` | `3e5bf41dfe1b0a52354f598e5ee734cc8d8a9bcc9ddbe32de15c0a99455b6992` |
| 0 | `sel_uniform_pair` | `14c9d7ed2d3df072375f5f304f696f2d6b8df26e0996c1379cbe1a11acdc9f41` |
| 1 | `xor_onpolicy` | `00edc2b47f83e9ed738dda8f50bf350ccf9f2a34ef1b043d7d6ceca4eaa3c1a1` |
| 1 | `xor_two_actions` | `3f4df5875787102760d738871f50f169de3cdd2a3f3d819561b0c8cd09ab20c3` |
| 1 | `sel_onpolicy` | `1ea4d620a2ef76a0da3ca8f1a014aa331dfee8b05d301016e2c0c1f1da1bb610` |
| 1 | `sel_uniform_credit` | `0ca0e578f3345517e7e7d56683d924a59a089d9538156000ab4349cb0617fcdf` |
| 1 | `sel_uniform_pair` | `dcc6cb38eca8097f6e00c0c62610c7ec8cedaf44fc74c4960610908931f0bba6` |
| 2 | `xor_onpolicy` | `1e4461aa5b91f5460a793116bd350b68143f6a46c8d45d84fa41c6e904e09821` |
| 2 | `xor_two_actions` | `d7945483cfd6b5fc632e98a542515499f65b7a8ce491dd1a9b0b40f2c418c35d` |
| 2 | `sel_onpolicy` | `632d084fcaf887659ecda0f2dc8ca40e20e75ceb1455cb4871b464883c5a9d7f` |
| 2 | `sel_uniform_credit` | `41c2f94777e058cc5b30f5952abcf014fb6bf5224cbd75f7e35162b39816648e` |
| 2 | `sel_uniform_pair` | `ec4055600644d4a80997823a719af9df6c33393cda5d0501d8df870eff5297c4` |
| 3 | `xor_onpolicy` | `1a3f6f57d0bf107745e5566c94a1ba64b028f4db91064dc50e492e2881d5d373` |
| 3 | `xor_two_actions` | `e485318dcd94c933b3c4efdc71970250dc5e856475e46ca5177aa7a737b4ad24` |
| 3 | `sel_onpolicy` | `8d646198e0ea53119dbdcaefc7c0241c5991d437121077f8b12a42f56f878a2a` |
| 3 | `sel_uniform_credit` | `0d78786fb92b66136ebda9eaee206e0085d35eeea536f99608d9efd3b2336fdd` |
| 3 | `sel_uniform_pair` | `832cb46fcb052ff11d79b81cf6edd7810593286ae74cbd42e00cdb1bfa2979f7` |

La política de composición primaria utiliza exclusivamente `sel_uniform_pair` + `xor_two_actions` de **la misma réplica**. Los otros tres brazos se conservan para controles diagnósticos; no se seleccionaron por rendimiento de confirmación. Para reproducir las salidas originales y verificar G3/G4, consultar [informe F10-L-C](INFINITO_F10L_INDEPENDENT_CONFIRM_RESULTS_2026-11-03.md), [prerregistro](INFINITO_F10L_INDEPENDENT_CONFIRM_PREREG_2026-11-03.md) y los artefactos de los dos runs GitHub Actions citados. Este documento no contiene los binarios; los SHA permiten validar una copia.
