# INFINITO F10-L — lanzamiento único, auditar primitivas ANTES de abrir composición

**2026-11-03. Prerregistro prospectivo**: [INFINITO_F10L_PRIMITIVE_CREDIT_PREREG.md](INFINITO_F10L_PRIMITIVE_CREDIT_PREREG.md), commit \`7fc030e2d5a119e9d9895b3c96c0e5dbd16561ad\`, escrito ANTES de desarrollar el generador F10-L, entrenar, abrir DEV u OOD. **No retocar criterios o pesos después de ver resultados**.

Esta marca lanza UNA vez el workflow \`.github/workflows/infinito-f10l-credit-gated-composition.yml\`.

Fase 1, cuatros réplicas:
- XOR-A REINFORCE estándar, XOR-B ejecuta 0 y 1 y aprende de diferencias de recompensa observadas.
- Selector-A on-policy, selector-B exploración uniforme con gradiente de score, selector-C MISMA exploración uniforme con diferencia de retorno entre slots ejecutados.
- Igual presupuesto de 65.536 episodios/actor, 512 updates, mismas observaciones y pesos iniciales por clase.
- El actor NO recibe etiquetas de respuesta ni winner; el feedback XOR binario de dos acciones revela indirectamente la verdad y no es una prueba de autonomía cognitiva amplia.
- Guardar SHA de 20 checkpoints antes de generar DEV. Usar solo TRAIN XOR 23100000+/SEL 23105000+, DEV XOR 23200000+/SEL 23205000+.
- Auditor independiente de 2.000 bootstrap por bloque de cuatro y G0: XOR-B >=95% + SEL-C >=85%, CI95 inferiores >90% y >75% en >=3/4 mismas réplicas.

**SOLO SI G0 PASS:** fase 2 descarga checkpoint SHA ORIGINALES, genera OOD composición jamás entrenada 23400000+, comprueba zero-shot de seleccionar último firmado/leer valor/aplicar XOR flip1, ablaciones shuffle y flip0. SI G0 FAIL, el workflow omite esos jobs y produce \`F10L_PRIMITIVE_COMPETENCE_FAIL\` sin abrir ningún OOD. **CONFIRM 23300000+ sellada**.

Todo vive en PR #62 draft sobre rama de investigación; NO tocar \`master\`, \`src/infinito3\` ni datasets Bemula.
