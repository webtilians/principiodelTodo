# INFINITO Security S0 — protocolo previo al laboratorio

**Fecha:** 2026-11-09  
**Estado:** protocolo anterior a la implementación y a cualquier resultado científico  
**Issue:** #134

## Pregunta
¿Podemos ejecutar pruebas adversariales reproducibles sobre las mismas clases de memoria remota que usa INFINITO, con dos usuarios sintéticos, sin acceder a producción ni revelar contenido de otro usuario?

## Alcance del clon
Primer corte **sin LLM ni servidor HTTP**: `People`, `SealedMemories`, `AgentMemory`, separación lógica de proyectos y `remote_apps`. No prueba OAuth completo, bóveda remota, fotos, concurrencia multiproceso, ataques a aplicaciones web ni seguridad real del despliegue. Ninguna conclusión sobre resistencia a agentes IA autónomos todavía.

Los únicos datos admitidos son frases sintéticas y marcadores de prueba. Los ficheros se crean en un directorio temporal y se destruyen en el mismo proceso. No se toman volcados, backups, tokens o memorias de producción. Ejecución recomendada en Docker con red deshabilitada, usuario sin privilegios, raíz de solo lectura y tmpfs temporal; el script no llama servicios remotos.

## Baseline y escenarios congelados (S0-A)
1. Control positivo: cada propietario recupera su propio marcador de memoria.
2. Aislamiento: Bruno no recupera el marcador de Ana y viceversa.
3. Token ficticio inválido: no se resuelve a ninguna cuenta.
4. Clave errónea: abrir el fichero sellado de otro usuario falla autenticación AES-GCM.
5. Eliminación con id ajeno: no borra el recuerdo de otro usuario.
6. Cifrado en reposo: los marcadores no están en bytes de ficheros sellados.
7. Filtro de proyectos etiquetados: `ProyectoNube` no mezcla `TENANT_B_ProyectoNube`.
8. Permisos de herramientas: perfil `read` no escribe, `add` no borra.
9. Datos potencialmente adversariales: un recuerdo con lenguaje de instrucciones se presenta como **dato no confiable**. Esto prueba solo formato, **no resistencia real de un LLM a prompt injection**.
10. Presencia de configuración de producción: el runner debe abortar antes de escribir un solo recuerdo.

## Gates
**PASS técnico** solo si pasan todos los controles positivos y negativos, no hay acceso a endpoints y se confirma el aislamiento del contenedor durante CI. Cualquier anomalía es **FAIL** y debe documentarse; no se corrige ni modifica el gate tras ver resultados sin nueva versión del protocolo.

**GO hacia S0-B** si el clon pasa: añadir agentes con acciones acotadas, simuladores de amenazas y validadores independientes; después OAuth y bóveda con fixtures sintéticas. **NO-GO a despliegue** siempre en esta fase. El objetivo científico posterior (memoria frente a sin memoria, mismo presupuesto y mundos nuevos) requiere prerregistro propio.

## Limitaciones y reglas de seguridad
La autorización no puede depender del comportamiento del LLM. Un test unitario no es una auditoría. Los perfiles de proyecto no sustituyen la separación de cuentas. Los límites de red de Docker son una barrera del sistema operativo: el bloqueo opcional de sockets Python no lo reemplaza. Prohibido apuntar el runner al servidor remoto, usar cuentas reales o fusionar automáticamente.
