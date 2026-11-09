# INFINITO Security S0-H — resultados de aislamiento OAuth y permisos del directorio de claves

**Fecha:** 2026-11-09. **Estado:** `COMPLETED` técnico en clones sintéticos. **Decisión:** continuar investigación de seguridad; NO aprobar despliegue ni afirmar invulnerabilidad.

## Referencias reproducibles

- Prerregistro previo al código: `docs/INFINITO_SECURITY_S0H_PROTOCOL_2026-11-09.md`, commit `4f874d16027048913540baac4e245b538542ee60`.
- Rama `experiment/security-s0h-tenant-isolation-20261109`, PR draft [#151](https://github.com/webtilians/principiodelTodo/pull/151), apilado sobre S0-G #150 y S0-F #147.
- GitHub Actions [run 42691853381](https://github.com/webtilians/principiodelTodo/actions/runs/42691853381) `SUCCESS`.
- Ejecución en Docker `--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, usuario no-root y raíz temporal. Solo cuentas, claves, OAuth tokens y memorias ficticios.

## Resultados

- **76 PASS + 1 XFAIL** en suite de seguridad S0-A…S0-H (8,12 s).
- **28 PASS** en regresión OAuth/MCP del producto (7,30 s).
- **12/12 nuevos casos W00–W11 PASAN** en un clon OAuth/MCP con tres cuentas ficticias y dos aplicaciones autorizadas para una de ellas.
- **W00–W04:** cada firma se vincula al `subject_account` y `client_id` del solicitante; una memoria de coral que cita `record_owner` de aurora no cambia su propietario autenticado, y la cuenta vacía devuelve lote vacío firmado.
- **W05–W07:** revocar los grants OAuth de la aplicación A para aurora produce HTTP **401** cuando se intenta usar su bearer antiguo, mientras la aplicación B conserva acceso a la misma cuenta y la cuenta coral conserva acceso solo a sus propias memorias.
- **W08:** aplicar nivel de permisos `read` a la app B permite `recall_attested` y niega `remember` y `forget`.
- **W09:** el verificador rechaza un documento firmado cuando se exige el `subject_account` o `client_id` de otra sesión.
- **W10:** llamadas concurrentes desde tres clientes OAuth independientes no dan registros cuyo `record_owner` firmado difiera de la cuenta solicitante.
- **W11:** se rechazan directorios de claves `0777` y enlaces simbólicos; el archivo privado no es regenerado o modificado al denegarse acceso.

### Corrección realizada

`ServerAttestor` en `src/infinito3/remote_provenance.py` valida con `lstat` que la raíz de la clave es un directorio real, no symlink, propiedad del UID del servicio y **no escribible por grupo ni por otros**. Si falla, lanza `PermissionError` antes de abrir/crear la clave; los demás comportamientos S0-F siguen iguales. El test diagnóstico A08 de S0-G pasa de `XFAIL` a `PASS` en esta rama. El antiguo resultado S0-G permanece como evidencia histórica.

### Limitación pendiente explícita

A07 sigue en **XFAIL**: no existe un protocolo implementado de revocación inmediata de firmas emitidas por una clave antigua comprometida o de migración segura del pin del cliente. Requiere ciclo de vida de claves, distribución y versión de confianza, relevante antes de producción con datos sensibles; no es imprescindible para seguir con clones offline que usan claves desechables.

A06 sigue siendo una propiedad conocida: reutilizar el mismo nonce permite verificar el mismo sobre dentro de sus 120 segundos; **no se afirma defensa anti-replay de uso único**. El cliente debe generar nonces nuevos (CSPRNG) por solicitud; una política de consumo único sería un cambio separado si el modelo de amenazas lo requiere.

## Interpretación científica y rumbo

La nueva prueba acredita comportamiento correcto en estos doce escenarios de acceso/tenant concretos; **no prueba ausencia de fugas**, resistencia a servidor comprometido, autenticidad del autor de cada memoria ni propiedad de cualquier registro compartido/importado. No se ha fusionado ni desplegado S0-H; el flag experimental continúa desactivado por defecto.

**Próximo experimento prioritario:** dejar de aumentar la infraestructura de firmas y volver al objetivo inicial de INFINITO Security: ¿la memoria episódica permite a un agente elegir mejores comprobaciones de seguridad en situaciones nuevas que un agente sin memoria y que un selector de experiencias programado, con igual presupuesto de tools/tokens y un verificador de evidencia estructurada común? Exigir holdout de combinaciones, criterios predefinidos y clon MCP auténtico. No reutilizar los bancos dev de S0-C/D/E/F/G/H para confirmación.
