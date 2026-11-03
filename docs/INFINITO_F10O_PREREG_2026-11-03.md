# F10-O — protocolo prospectivo: efectos desde observaciones terminales

Fecha 2026-11-03. Registrar este documento en GitHub ANTES de implementar, generar TRAIN o abrir OOD. No es F10-N-C adicional. Producto base master f6bd6239bfc1734c04c233a5cf8cef050c6d54c2; laboratorio dependiente, inmutable, f1ba7db6364cb92f0443d63a4d13eb527049f60d. Artefactos F10-L run 41685499939 verificados contra G0. F10-N, sus memorias, confirmaciones y semillas permanecen intactas.

## Pregunta y alcance
¿Un modelo recurrente de operaciones, supervisado únicamente con respuesta final observable y recompensa, transfiere a MEMORY+FLIP1 y propone SRX sin haberla ejecutado? Se retiran before/after, selected, read_observed, invalid y estados de trabajo. Los nombres P/S/R/X solo son identificadores categóricos. El objetivo y contexto SOURCE/FLIP siguen dados; esto NO prueba observaciones naturales sin diseño humano, nuevos operadores, ni identificabilidad causal universal. Es prueba de observación terminal parcial en el mismo mundo sintético.

## Datos y presupuesto fijos
Nueva partición TRAIN 27100000 + 10000*r + 1000*t + j (r=0..3,t=0..2,j=0..255); DEV 27200000 mismo esquema j=0..63; OOD 27400000 +10000*r+j, j=0..127. Cuatro variantes por mundo como F10-N. Rango 27300000 reservado sin generación ni apertura. Nuevas semillas, misma familia, no validación externa. No cambiar archivos originales: cargar generador original por copia de módulo con nuevas constantes en memoria.
TRAIN: 3072 casos, 16 pases, dos programas uniformes por caso: 98304 episodios por réplica, todos los brazos reciben idéntica secuencia. 84 programas (longitud 0..3), excluido SRX; MEMORY+FLIP1 ausente TRAIN/DEV. RNG orden 276000 +100*r+epoch; programas 286000 +100*r+epoch. Inicialización torch 296000+r. Un único intento, sin ajuste tras DEV/OOD.

## Brazos
T: GRU de efectos latentes con contexto público (source,flip,public_bit), embedding acción 8, estado 32, inicialización lineal contexto→estado, dos cabezas lineales (éxito binario y respuesta terminal binaria). 3 pasos máximo con padding enmascarado, estado conserva valor en padding. Adam lr=.003, batch=192, 512 actualizaciones, BCE éxito + BCE respuesta terminal, sin dropout, sin selección de checkpoint. Aprende la respuesta de la ejecución real, nunca la verdad oculta como objetivo de respuesta.
R: arquitectura e inicialización idénticas, BCE éxito + 0*BCE respuesta. Control de recompensa de capacidad y cómputo comparable.
S: arquitectura idéntica, BCE éxito + BCE respuesta terminal permutada dentro de cada batch con RNG 306000+r. Control de información auxiliar. Misma cuenta de parámetros, pasos, ejemplos y candidatos; informar tiempos reales, no afirmar igualdad exacta de FLOPs ni capacidad con tablas.
B: memoria aditiva de recompensa F10-N original. I: memoria de efectos instrumentados F10-N original como control positivo de información privilegiada, nunca mezclada con T/R/S. Solo I recibe trace. Igual presupuesto de interacción, no igualdad de cómputo con redes.
T/R/S planifican antes de ejecutar por máximo sigmoid(éxito) - .01*longitud sobre los 85 programas (SRX habilitado únicamente en evaluación); empate: orden original del catálogo. Sin observar memoria ni respuesta de prueba al elegir. La cabeza terminal auxilia el aprendizaje, no proporciona una regla XOR manual. Evaluar órdenes permutadas de IDs de acción (P→S→R→X→P en entrada del predictor) como diagnóstico, sin reentrenar.

## Congelación, auditoría, análisis
Guardar SHA de modelos, memorias, protocolo, código y logs TRAIN antes de construir DEV/OOD de cada réplica. No entrenar, seleccionar ni ajustar con DEV/OOD. Antes y después comparar los 20 checkpoints originales contra G0. Conservar todas las filas de evaluación con mundo, variante, plan, respuesta y controles. Evaluar normal, shuffle F4, X desactivada y public_randomized sobre cada caso y plan fijo. Auditor separado reconstruye métricas a partir de filas y verifica SHA; no se denomina auditor humano ni réplica independiente de implementación.
Bootstrap pareado 2000 remuestreos por mundo base, conservando juntas cuatro variantes, RNG 316000+r. Réplica es unidad para criterio global. No usar 2048 casos como independientes. Intervalos percentiles 95%; familias de comparaciones son conjunción, no elegir el mejor brazo después.

## Umbrales fijos (todos necesarios en >=3 de 4 réplicas)
G0: T >=80% en cada contexto DEV.
G1: T OOD >=80%, límite inferior IC95 >70%, cada variante >=70%, cada n=9/11 y modo de fuente >=70%; SRX elegido >=90%.
G2: caída T normal-shuffle >=20pp y normal-X_disabled >=20pp, ambos límites inferiores IC95 >0.
G3: T supera a R, S y B por >=10pp cada uno, con límite inferior IC95 >0 para cada diferencia pareada.
G4 diagnóstico: I conserva competencia (OOD >=80%) y se informa T-I; no cambia umbrales.
Integridad obligatoria: cero SRX/TRAIN, cero MEMORY+FLIP1/TRAIN, semillas disjuntas, SHA conservados, presupuestos exactos, ningún campo interno en entrada T/R/S. Fallo técnico = INCONCLUSIVE; corrida íntegra sin gates = F10O_TERMINAL_TRANSFER_FAIL; gates completos = F10O_TERMINAL_TRANSFER_PASS. Ningún fallo autoriza retocar esta corrida. No concluir imposibilidad desde un modelo fallido.

## Decisión producto
Un PASS permite diseñar confirmación independiente y piloto aislado; no permite integración automática. FAIL implica no-go del mecanismo aprendido y priorizar captura/recuperación/contexto y olvido del producto. Incluso con FAIL se puede recomendar solo instrumentación de observaciones saneadas en laboratorio/shadow, sin acciones ni entrenamiento sobre usuarios. Evaluar APIs actuales, privacidad, borrado de memorias derivadas, permisos, latencia, coste y beneficio contra RAG a presupuesto de tokens idéntico. Producto no se modifica. Ningún merge a master.
