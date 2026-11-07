# INFINITO — conexión automática para seis clientes

**Diseño y primer MVP — 7 de noviembre de 2026. Sin desplegar.**

## Objetivo

Que un nuevo usuario llegue al enlace /connect, elija su asistente y autorice
el acceso a la misma memoria, con el mínimo de pasos. Se reutilizan el MCP
público /mcp y los endpoints OAuth 2.1 existentes (descubrimiento, registro
dinámico de clientes, PKCE y refresh). No se duplica almacenamiento ni
autenticación, y no se pide la contraseña de otro proveedor.

## Flujo propuesto

1. La portada y el portal de recuerdos enlazan a /connect.
2. El usuario selecciona ChatGPT, Claude, Cursor, Codex, Claude Code u otro MCP.
3. Cursor ofrece un deeplink de un clic con JSON base64 que solo contiene la URL
   pública. Cursor confirma la instalación y solicita permiso OAuth.
4. ChatGPT y Claude abren sus configuraciones oficiales. El usuario introduce
   la URL y autoriza; las restricciones de planes y espacios de trabajo siguen.
5. Codex y Claude Code admiten un comando opcional del cliente local, con
   previsualización y confirmación. Se ejecutan sus CLI oficiales, no ediciones
   silenciosas de archivos ni shell.
6. Otro cliente compatible puede copiar un JSON sin claves ni tokens.
7. El usuario verifica la memoria cruzada: guardar un hecho de prueba en un
   asistente, recuperarlo en otro y borrarlo desde el portal.

## Archivos

- src/infinito3/remote_connect.py: URL canónica, deeplink Cursor, comandos,
  JSON y página web localizada en ES/EN.
- src/infinito3/remote_accounts.py: ruta pública GET /connect en Starlette.
- src/infinito3/remote_installer.py: operaciones locales con aprobación
  explícita; solo subprocess con argumentos, nunca shell.
- src/infinito3/remote.py: comando infinito-remote connect, con --dry-run y --yes.
- src/infinito3/remote_home.py y remote_portal.py: enlaces al nuevo flujo.
- tests/test_remote_connect_v3.py: tests unitarios de enlaces y ejecución.
- tests/test_remote_accounts_v3.py: integración de ruta/OAuth.
- remote/README.md: guía de conexión y limitaciones.

## Matriz de automatización

| Cliente | Primer MVP | Límite real |
|---|---|---|
| Cursor | Deeplink instalar con un clic | Confirmación y OAuth en Cursor |
| Codex | Instalador local + OAuth | CLI instalado y aprobación |
| Claude Code | Instalador local + OAuth | CLI instalado y autorización desde /mcp |
| ChatGPT | Abrir Plugins y dar URL MCP | No se puede instalar o autorizar silenciosamente |
| Claude | Abrir Conectores y dar URL MCP | Configuración y consentimiento en Claude |
| Otros MCP | JSON genérico para HTTP/OAuth | Cliente compatible y autorización propia |

Los permisos de lectura/escritura pueden variar por plan. La integración no
garantiza que ChatGPT o Claude consulten automáticamente la memoria antes
de cada respuesta. Ese problema requiere distribución y mecanismos del host.

Referencias:
- https://prod.cursor.com/docs/mcp/install-links
- https://prod.cursor.com/docs/mcp
- https://developers.openai.com/learn/docs-mcp
- https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt
- https://code.claude.com/docs/en/mcp

## Seguridad

- HTTPS obligatorio, excepto localhost para desarrollo; se rechazan URLs con
  usuario, contraseña, query, fragmentos y rutas privadas de memoria.
- El deeplink solo contiene un URL público. No transporta tokens ni secretos.
- Ningún componente de /connect accede a memorias o contraseñas.
- El usuario siempre concede permisos a cada cliente mediante OAuth existente.
- El instalador exige confirmación o aprobación explícita --yes; --dry-run solo
  muestra un plan. No utiliza shell ni modifica directamente JSON del cliente.
- Si una configuración llamada infinito ya existe, la CLI oficial decide si
  rechaza el cambio. No se realiza sobreescritura forzada.
- La página envía CSP restrictiva, no-store y no-referrer.
- No se crean cuentas sin consentimiento, ni se altera la autenticación del servidor.

## Qué falta antes de producción

- Ejecutar los tests integrados y probar de extremo a extremo las seis
  plataformas con cuentas de prueba y permisos reales.
- Añadir errores de conexión guiados y mecanismo para ver/revocar los clientes
  OAuth sin exponer tokens.
- Publicar INFINITO en catálogos de plugins/marketplaces cuando se apruebe.
- Crear instaladores firmados en Windows/macOS/Linux, sin ejecutar comandos
  remotos obtenidos dinámicamente.
- Revisar seguridad de concurrencia y restauración de fotos en producción.
- Opcional: hooks de recuperación automática por cliente, con autorización
  independiente y desinstalación reversible.
- No fusionar esta rama ni desplegarla hasta revisión.

## Pruebas

    python -m pytest tests/test_remote_connect_v3.py tests/test_remote_accounts_v3.py -q

Criterios: seis opciones presentes en ambos idiomas, endpoint común y seguro,
Cursor deeplink decodificable, ninguna credencial en enlaces, previsualización
sin ejecución, 401 del MCP sin token, descubrimiento OAuth intacto y conexión
real después de que el titular autorice.
