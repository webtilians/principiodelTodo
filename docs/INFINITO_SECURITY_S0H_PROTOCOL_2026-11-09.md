# INFINITO Security S0-H — aislamiento OAuth y endurecimiento mínimo de clave

**Fecha:** 2026-11-09. **Este protocolo precede al código y a los resultados S0-H.** Base S0-G #150 / S0-F #147. No alteramos K00–K15 ni A00–A09 usados para diagnóstico. Los ensayos se ejecutan sobre un clon OAuth/MCP, con perfiles ficticios nuevos, en Docker sin red externa, con el flag experimental activado solo en el test.

## Prioridades por riesgo y propósito

El objetivo es confirmar que una firma emitida por el servidor corresponde al almacén realmente autorizado por OAuth y que ni una sesión de otro usuario, una app distinta ni un acceso revocado permiten leer ese almacén. Evitar implementar una infraestructura de rotación de claves no necesaria para investigación aislada.

El S0-G A08 mostró que el firmante carga una clave privada de fichero `0600` desde un directorio escribible por otros sin rechazarlo. **Corrección técnica mínima**: antes de crear/abrir la clave, comprobar que la raíz es un directorio real, no symlink, perteneciente al UID de servicio y que sus permisos no permiten escritura al grupo ni a otros. Si no lo cumple, error bloqueante sin generar ni cargar claves. Conservar clave antigua intacta cuando directorio sea inseguro. No alterar herramientas, esquema de firma, autenticación ni banderas.

## Banco prospectivo nuevo W00–W11

Se usarán tres cuentas ficticias inéditas `s0h-aurora`, `s0h-coral`, `s0h-duna`, con dos clientes OAuth distintos para aurora y memorias diferentes, incluido texto que pretende ser propiedad de otra cuenta. Marcas nuevas, no reutilizadas desde S0-C…G.

- W00: cuenta aurora, app A, sobre válido de su almacén.
- W01: cuenta coral, app C, sobre válido; no hay registro real de aurora.
- W02: cuenta duna, app D, sobre firmado de lista vacía.
- W03: segunda app B de aurora lee el mismo almacén permitido, pero firma `client_id` distinto.
- W04: el texto de coral falsifica `record_owner=aurora` y contiene un canario citado; el dueño firmado sigue siendo coral.
- W05: tras revocar solamente la app A de aurora con `Grants.drop_client`, una llamada HTTP MCP con su bearer previo devuelve 401.
- W06: la app B de aurora conserva el acceso y devuelve evidencia válida tras revocar A, sin poder atribuirse al client_id A.
- W07: la app C de coral conserva acceso y no ve la memoria de aurora después de la revocación de A.
- W08: la app B de aurora configurada en nivel `read` puede usar `recall_attested` pero no `remember` ni `forget`.
- W09: el verificador rechaza un sobre de aurora al exigir una cuenta o un client_id de coral (cross-session binding).
- W10: en llamadas de lectura concurrentes desde varias cuentas con clientes independientes, ninguna respuesta llega firmada con `record_owner` ajeno a la cuenta solicitante.
- W11: el directorio raíz `0777` o un enlace simbólico de directorio se rechaza antes de usar la clave de firma, mientras que uno privado `0700` funciona. Validación nueva del defecto A08.

## Gates

- **T0 técnico**: tests W00–W11 ejecutados realmente con OAuth y MCP local donde corresponda; código fuente S0-F solo cambia en validación del directorio; Docker `--network none`; no datos/credenciales de producción; regresiones S0-A…G y `test_remote_accounts_v3.py` pasan.
- **T1 separación**: sin leakage entre cuentas en W01/W04/W07/W10, 401 con token revocado W05 y token de otra app de aurora no afectado W06.
- **T2 firmas**: todas las firmas honestas ligadas al contexto OAuth y las de cuenta/cliente equivocado rechazadas W09; cambiar texto de una memoria no cambia dueño autenticado.
- **T3 permisos**: roots inseguros rechazados sin regeneración; A08 se convierte de `XFAIL` a caso resuelto (la prueba histórica permanece como evidencia de la vulnerabilidad previa, pero se actualiza su expectativa en la nueva rama).
- **T4 límites**: no afirmar protección anti-replay de nonce si el cliente lo reutiliza (A06); revocación de firma antigua A07 sigue **XFAIL conocido**; pinning/rotación integrales quedan para fase de producto, no bloquean clones de investigación.

**Lectura:** si el conjunto pasa, solo GO para investigación sintética con controles; no significa autorización para producción ni certifica ausencia de fugas. Tras esta verificación, regresar al estudio de utilidad real de memoria de agentes vs baseline sin memoria y selector de experiencias programado; no ampliar criptografía sin una hipótesis de agente que lo necesite.
