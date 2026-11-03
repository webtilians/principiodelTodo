# INFINITO F10-L-C — LANZAMIENTO ÚNICO DE CONFIRMACIÓN INDEPENDIENTE

**Fecha 2026-11-03.** Este documento dispara el workflow \`.github/workflows/infinito-f10l-original-confirmation.yml\`.

## Contrato científico congelado

Prerregistro prospectivo original [INFINITO_F10L_INDEPENDENT_CONFIRM_PREREG_2026-11-03.md](INFINITO_F10L_INDEPENDENT_CONFIRM_PREREG_2026-11-03.md), commit **\`dea0b788aeef3ae449c4062da828fae24fb37457\`**, creado **ANTES** de escribir generador/inferencia/confirmación. **No modificar sus umbrales científicos tras ver datos**.

- Código F10-L original congelado en commit \`b5c32cbfb50a24414607e3a24e30e139ca92fede\`.
- **20 modelos ORIGINALES** y sus SHA extraídos de [Actions #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939). Ningún reentrenamiento ni selección de checkpoint.
- Preflight de equivalencia bit-a-bit de los 2048 casos OOD existentes frente a generador nuevo usando únicamente **semillas OOD previamente abiertas**; 0 semillas de CONFIRM en preflight.
- Antes de la primera semilla CONFIRM, obtener los SHA originales y comparar el resultado de inferencia con la rutina **F10-L original inalterada**. La rutina de confirmación reutiliza literalmente esa inferencia original; cambia únicamente la fuente de datos.
- Abrir nueva partición **\`23300000+10000r+j\`**, para r=0..3 y j=0..127, con cuatro variantes contrafactuales cada una: **2048 casos nuevos**. Mismo protocolo de generación OOD original (9/11 informes y 4 modos de procedencia). Grabar todas las predicciones y checksums.
- Auditor independiente, 2000 remuestreos **por mundo base**. C-G3 precisión >=80%, límite inferior IC95>70% y mínimo >=70% por longitud/fuente/variante. C-G4 reducción al barajar F4 y eliminar XOR >=20pp e IC inferior>0; ambos criterios >=3/4 réplicas. **PASS solo con ambos gates**.
- La confirmación prueba reproducción dentro de la misma familia sintética, **NO** descubrimiento autónomo de operadores, ni superioridad del estimador paired sobre RL convencional (esas puertas G1/G2 fallaron en el experimento original).

**STOP tras una sola ejecución**. No retocar modelos, evaluador ni generador mirando los resultados; reportar FAIL cuando proceda. PR #63 permanece en borrador, \`master\` y el producto intactos.
