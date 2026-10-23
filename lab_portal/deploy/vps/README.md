# Laboratorio en un servidor, con HTTPS y actualización automática

Instala el portal en un servidor Ubuntu que ya tiene **Caddy** (el servidor de LAIN, junto a LAIN y DH Trails). El portal queda en `https://lab.<ip-con-guiones>.sslip.io/` con certificado HTTPS. Después **se actualiza solo** cada vez que llega un commit nuevo a su rama de GitHub.

## Instalar

Como `root` en el servidor:

```bash
git clone --depth 1 --branch experiment/f8-laboratory-portal https://github.com/webtilians/principiodelTodo.git /tmp/principiodelTodo
bash /tmp/principiodelTodo/lab_portal/deploy/vps/setup.sh
```

`setup.sh` se copia a `/opt/infinito-lab-deploy` y prepara:

| Qué | Dónde |
|---|---|
| Servicio `infinito-lab` (usuario del sistema `infinito-lab`) | `127.0.0.1:8200`, publicado por Caddy |
| Código (clon superficial de la rama) | `/opt/infinito-lab` |
| Entorno de Python | `/opt/infinito-lab-venv` |
| Clave `LAB_API_TOKEN`, generada al azar una vez | `/etc/infinito-lab/lab.env` (solo root) |
| Ejecuciones y resultados | `/var/lib/infinito-lab/runs` |
| Sitio de Caddy | `/etc/caddy/sites.d/infinito-lab.caddy` |

Volver a ejecutarlo es seguro: conserva la clave y los resultados. Si Caddy no acepta la configuración, retira el sitio del laboratorio y deja los demás como estaban.

## Actualización automática

`infinito-lab-update.timer` pregunta a GitHub cada 2 minutos por el último commit de la rama de `/etc/infinito-lab/branch`. Si hay uno nuevo:

1. Descarga el código e instala `lab_portal/requirements.txt` si ha cambiado.
2. Importa el portal con el usuario del servicio **antes** de tocar el que está funcionando.
3. Reinicia el servicio y comprueba que responde.
4. Si algo falla, vuelve a la versión anterior y no reintenta ese commit hasta que llegue otro.

**Si hay un experimento en marcha, espera a que termine**, porque reiniciar el servicio lo cortaría. Instala la versión nueva en la primera revisión después de que acabe.

## Comandos en el servidor

| Comando | Para qué |
|---|---|
| `infinito-lab-status` | Dirección, rama, versión instalada, si hay un experimento y últimas actualizaciones |
| `infinito-lab-update` | Instalar ya el último commit, si lo hay |
| `infinito-lab-update --force` | Reinstalar aunque no haya commit nuevo |
| `LAB_BRANCH=experiment/x infinito-lab-update` | Seguir otra rama a partir de ahora |
| `infinito-lab-token` | Mostrar la clave que pide el portal |
| `infinito-lab-token nueva` | Cambiar la clave (la anterior deja de valer) |

Registro de las actualizaciones: `journalctl -u infinito-lab-update`. Registro del portal: `journalctl -u infinito-lab`.

## Recursos

El servidor tiene **1 CPU y 2 GB** compartidos con LAIN y DH Trails. El servicio usa `CPUWeight=10` y `Nice=10`: los experimentos aprovechan la CPU libre, pero ceden en cuanto LAIN la necesita. La memoria está limitada a 600 MB (`MemoryMax`); un F8 estándar usa unos 55 MB. Los límites están en `systemd/infinito-lab.service`.

La duración máxima de una ejecución es `LAB_MAX_SECONDS` en `/etc/infinito-lab/lab.env` (4 horas por defecto). Después de cambiarla: `systemctl restart infinito-lab`.

## Cambiar este kit

El actualizador automático solo cambia el código del portal, nunca este kit: una rama que no tenga esta carpeta no rompe el servidor. Para aplicar un cambio hecho aquí, cuando ya esté en la rama que sigue el servidor:

```bash
bash /opt/infinito-lab/lab_portal/deploy/vps/setup.sh
```
