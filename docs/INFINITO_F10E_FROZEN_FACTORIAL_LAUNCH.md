# INFINITO F10-E — Único lanzamiento de factorial causal con actores originales congelados

**2026-11-02**. El protocolo [`INFINITO_F10E_CREDENTIAL_TIME_FACTORIAL_PREREG.md`](./INFINITO_F10E_CREDENTIAL_TIME_FACTORIAL_PREREG.md) quedó fijado **ANTES de implementar generador/ejecutor y antes de abrir mundos nuevos**. Su commit prospectivo con la corrección de dimensionalidad de entrada es `e7667782119c5034c565339cc9c839917bbe5cac`, conservando borrador previo `40c0e1bbf38f74294b6e7ac85ec84f821c42a097`.

Únicamente se usarán mundos **`16_200_000+10000*replica+j`**, `replica=0..3`, `j=0..107`, 108 nuevos mundos base/actor con cuatro variantes emparejadas cada uno (**1728 consultas dependientes de 432 mundos distintos**). Se cruzan tres cantidades de evidencias por clave (3, 5 y 8), tres pares de posiciones (34–35, 39–49 y 48–59), la acreditación de la actualización contradictoria y su precedencia respecto a otra confirmación acreditada. Sólo combinación actualización **acreditada y posterior** cambia la respuesta correcta. Credenciales y timestamps vienen de informes públicos; F4 es log de eventos públicos, no memoria del observador.

Checkpoint originales F10-C **run #41447171697** (relacional, unario, recurrent-only) y F10-B **run #41443064754** (READ, recurrent-only) descargados y auditados por SHA. **SIN entrenamiento, sin optimizador, sin ajuste de hiperparámetros, sin cabeza READ, sin reabrir confirmaciones o evaluaciones previas**.

El gate G1/G2/G3 y sus umbrales son los prerregistrados. La historia GRU **cambia entre variantes** porque cambian datos observables de verdad; READ/SKIP se compara **dentro de cada variante con idéntica historia pública**. El bootstrap se agrupa por mundo base. Este test diagnostica conducta causal en tarea sintética, no mecanismo neuronal interno ni seguridad criptográfica. `16_300_000+` CONFIRM sellada.

La publicación de este documento lanza `.github/workflows/infinito-f10e-credential-time-factorial.yml` exactamente una vez.
