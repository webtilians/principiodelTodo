# INFINITO F10-M — lanzamiento único del ensayo de selección autónoma de secuencias

**Prerregistro ORIGINAL anterior al código F10-M y a todos los datos**: [INFINITO_F10M_AUTONOMOUS_PROGRAM_DISCOVERY_PREREG.md](INFINITO_F10M_AUTONOMOUS_PROGRAM_DISCOVERY_PREREG.md) commit **\`7f21d7393ad85ab32dcacd991e6df9d56fadcd78\`**. La rama y PR #65 son de INVESTIGACIÓN, no de producto. No retocar los criterios tras OOD.

Este archivo desencadena exactamente una ejecución de \`.github/workflows/infinito-f10m-autonomous-program-discovery.yml\`.

- **Los operadores congelados**: se descargan del run ORIGINAL F10-L \`41685499939\`, con verificación SHA-256 de TODOS los cinco checkpoints por réplica frente al informe G0 de aquel run. El selector \`sel_uniform_pair\` y el transformador \`xor_two_actions\` se utilizan únicamente en inferencia, sin actualizar parámetros.
- **Programas candidatos**: exactamente **85** cadenas enumeradas sistemáticamente, de hasta tres acciones de \`{P,S,R,X}\`, con ANSWER terminal. El agente ensaya operaciones válidas e inválidas con sus costes; NO recibe el programa correcto.
- **Cuatro controladores**: preferencia global (A), MLP contextual (B), política aditiva/composicional (C), crédito por recompensa de DOS programas realmente ejecutados (D); mismo presupuesto **98.304 episodios ejecutados por brazo y réplica**, cuatro réplicas. Ninguna recompensa por intentos no ejecutados.
- **TRAIN** PUBLIC0, PUBLIC1, MEMORY0: \`25100000+\`. **DEV** solo las mismas tareas: \`25200000+\`. **OOD** MEMORY1 jamás entrenada: \`25400000+\`, con 9/11 informes y cuatro modos de procedencia. **CONFIRM \`25300000+\` SELLADA.**
- Guardar **16 checkpoints de controladores SHA** antes de generar DEV/OOD; 20.480 filas DEV/OOD evaluadas en condiciones congeladas; auditor estadístico separado sin entrenamiento y 2000 bootstrap por mundo base. G0/G1/G2/G3 requieren al menos 3/4 réplicas para declararse PASS; G4 paired credit solo diagnóstico, no requisito.
- No convertir un workflow técnico SUCCESS en PASS si los gates científicos fallan. Mantener fallos, escenarios, semillas y modelo originales; publicar negativa si corresponde; nunca introducir cambios al producto \`master\` o al benchmark Bemula.

La pregunta limitada es: **¿aprendió el agente a elegir una secuencia completa a partir de su propia experiencia y puede transferirla a una combinación nueva?** No se busca ni se reclama invención libre de operaciones, replanteamiento online, autonomía cognitiva general o capacidades comerciales.
