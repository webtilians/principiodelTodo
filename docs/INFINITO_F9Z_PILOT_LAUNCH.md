# F9-Z-A — lanzar piloto de factibilidad, sin entrenar predictor

2026-11-02. El protocolo preregistrado se congeló en commit **`bc19d5fe15fd0679a5dbcdf55c161c17f8b18f3e`**, antes del código y de abrir los mundos nuevos.

Se autorizan **únicamente** 24 mundos `11_100_000+10_000*index+j` para cada uno de los cuatro actores F9-P congelados; 9 tratamientos completos `capacity N ∈ {1,4,8}` × `READ window W ∈ {1,4,8}`, pistas key/ball contrafactuales, historias normal/postcue_off. Cada brazo se compara con SKIP desde el mismo estado, entorno, observación pública y escritor F4; ningún positivo se selecciona para formar el dataset.

El objetivo es contabilizar soporte causal con umbral fijado: **el mismo tratamiento** debe producir >=3 D+ y >=3 D<=0 para cada una de las 16 celdas actor×cue×historia. Si no ocurre, `SUPPORT_NOT_ACHIEVED` y STOP. Se verificará equivalencia `N=8, W=1` con F9-V original en mundos nuevos y SKIP invariante entre W.

No se entrenan cabezas, no se altera el actor, no se abre `11_200_000+`, `11_300_000+`, `11_400_000+` ni confirmaciones anteriores. Esta edición dispara `.github/workflows/infinito-f9z-causal-support.yml`.
