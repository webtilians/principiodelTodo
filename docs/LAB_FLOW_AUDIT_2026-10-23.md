# Auditoría del flujo del laboratorio — 23/10/2026

Base auditada: `cc4777f93c89c391d06a4e7fbbdf730d9de0e862`, rama
`experiment/f8-laboratory-portal`. Correcciones aisladas en
`fix/lab-flow-audit-20261023`; no se cambia master ni se despliega producción.

## Evidencia y límites

- HTTPS público: `GET /` devuelve 200; `/assets/lab.js` y `/assets/lab.css` son accesibles.
- Sin credenciales, `/api/ping`, `/api/catalog` y `/api/jobs` devuelven 401.
  Después, con la clave proporcionada por el usuario, ping y catálogo respondieron
  correctamente: seis familias y datos en `/var/lib/infinito-lab/runs`.
- 19 tests locales pasan en Windows/Python 3.12. Incluyen procesos reales de las
  seis familias F4, F5, F6, F7, F7b y F8 (perfiles rápidos), catálogo, monitorización,
  informe, resumen, ZIP íntegro y lectura persistente desde otro intérprete.
- Además, Uvicorn real en localhost: F4 `065b4669b47b4543ab086f8f2d2b497f`
  y F8 `8a39aab9322945e7a2e52f9af4e8141a` completaron el recorrido HTTP.
  Se reinició el servidor y los informes conservaron exactamente el mismo SHA-256.
  Son comprobaciones técnicas, no nuevos resultados científicos confirmatorios.
- Pruebas remotas autenticadas por HTTPS: F4 `e5eff332920343f38fee94b1c6970c37`
  y F8 `232cf5c32c8a4808bef35d1f69c7c3e9` completadas; informes, resúmenes y ZIP
  descargados y comprobados, sin registrar la clave. La interfaz autenticada muestra
  las seis familias, las ejecuciones, logs y gráficas. Se lanzó además F4 desde el navegador.
- No se ha reiniciado el VPS. Falta conexión SSH autorizada para esta auditoría.
  La configuración efectiva de systemd/Caddy, el volumen de producción y la versión
  del backend remoto siguen sin verificar (ping todavía no devuelve SHA). El HTML público no demuestra la versión del backend.

Los tres archivos estáticos remotos coinciden con la base auditada al normalizar
finales de línea; esto no demuestra la versión del backend ni el temporizador activo.

## Fallos corregidos

1. Se recortaban 100 carpetas por UUID antes de detectar actividad: un trabajo activo
   podía quedar fuera del listado y permitir otro simultáneo. Se recorren todos los
   registros y se ordenan por fecha, no por UUID aleatorio.
2. Un fallo de creación del supervisor dejaba un trabajo `queued` sin PID para siempre.
   Ahora queda `failed` y devuelve 503; una cola huérfana sin PID pasa a `interrupted`
   tras 30 segundos de margen para el arranque. Se espera al supervisor en un hilo
   para recoger su salida y evitar procesos zombis mientras vive el servidor.
3. En Windows `os.kill(pid, 0)` puede terminar el proceso consultado. Se usa una consulta
   de handle sin señales; un test verifica que el proceso sigue vivo tras la consulta.
4. Faltaba vincular los resultados con una versión del código: `/api/ping` y `job.json`
   incluyen `revision` (SHA capturado al importar el servidor, o null sin Git).
   Requiere checkout limpio y no modificar código durante trabajos; no es una instantánea
   de dependencias ni un hash de cambios locales sin commit.

## Cómo llegan los cambios (lectura del código, no certificación del VPS)

- **Kit VPS**: setup instala `infinito-lab-update.timer`. Cada 2 minutos consulta
  `git ls-remote`; hace fetch/checkout/reset de `/etc/infinito-lab/branch`, instala
  requisitos si cambia su hash y reinicia `infinito-lab`. Comprueba HTTP `/` y tiene
  rollback de código si falla. No es un webhook ni un `git pull` ni GitHub Actions.
  Comprueba con pgrep si hay supervisor activo antes de actualizar.
- **Docker Compose**: construye una imagen local con el repositorio. No contiene
  auto-pull ni actualizador. Requiere actualizar checkout y reconstruir imagen.
  Persiste en volumen `infinito_lab_data`; mantener proyecto Compose/volumen estable.
- **GitHub Actions**: el workflow del portal hace tests; no conecta por SSH al VPS,
  no publica imagen ni despliega. Esta corrección añade cobertura para la rama de
  auditoría y PRs hacia la rama experimental, y ejecuta las nuevas pruebas de flujo.
- **Catálogo**: es `lab_portal/catalog.py`, cargado al arrancar. Subir solamente un
  script de experimento no lo registra ni actualiza un servidor ya en ejecución.

## Verificación y despliegue en VPS

Ejecutar dentro de una sesión SSH ya autorizada, sin copiar claves a chats/logs:

```bash
sudo systemctl is-active infinito-lab infinito-lab-update.timer
sudo systemctl list-timers infinito-lab-update.timer --no-pager
sudo cat /etc/infinito-lab/branch
sudo git -C /opt/infinito-lab rev-parse HEAD
sudo cat /var/lib/infinito-lab-deploy/deployed
sudo journalctl -u infinito-lab-update -n 50 --no-pager
```

No mostrar `/etc/infinito-lab/lab.env` ni ejecutar el comando que imprime el token.
Antes de actualizar, impedir nuevas ejecuciones desde otras sesiones, esperar a que
terminen las actuales y respaldar `/var/lib/infinito-lab/runs`. El chequeo pgrep no
constituye un bloqueo transaccional: otra sesión podría lanzar un trabajo entre
el chequeo y el reinicio. El rollback actual restaura código, no revierte paquetes
Python. Por esas limitaciones, hacer la primera prueba en ventana de mantenimiento.

Para probar **esta rama**, una vez revisada, elegir explícitamente la rama que seguirá
el servidor (el comando persiste esa elección; no modifica master):

```bash
sudo env LAB_BRANCH=fix/lab-flow-audit-20261023 infinito-lab-update
sudo git -C /opt/infinito-lab rev-parse HEAD
sudo cat /var/lib/infinito-lab-deploy/deployed
```

Comparar ambos SHA con GitHub y con `/api/ping.revision`. Cambios en el kit de despliegue
requieren volver a instalar `setup.sh`; el temporizador no actualiza su propia copia.
Para volver a la rama anterior usar `LAB_BRANCH=experiment/f8-laboratory-portal`
con el mismo actualizador. Esto elige su último commit, no un rollback exacto de versión.

Prueba autenticada reproducible (la clave queda solo en el entorno del proceso):

```bash
sudo bash -c '
  set +x
  set -a
  . /etc/infinito-lab/lab.env
  set +a
  cd /opt/infinito-lab
  /opt/infinito-lab-venv/bin/python scripts/check_lab_portal_flow.py \
    --url https://lab.178-105-103-4.sslip.io --experiment f4 \
    --output /var/lib/infinito-lab-deploy/acceptance-f4
'
```

El comprobador crea un trabajo real rápido, guarda su ID inmediatamente, espera,
comprueba JSON/resumen/ZIP y guarda huellas. Si hay un trabajo activo, no crea otro.
Si vence su espera, no cancela automáticamente: consultar el ID guardado.
Repetir con F8 y otra carpeta. Después de terminar, reiniciar el servicio durante la
ventana acordada, volver a abrir ambos trabajos y descargar informes; comparar las
huellas antes/después. La prueba HTTP sola no acredita persistencia tras un despliegue.

## Variante Docker

Con el checkout limpio, sin trabajos en marcha, y desde el mismo directorio/proyecto
Compose utilizado originalmente:

```bash
git fetch origin
git switch fix/lab-flow-audit-20261023
git pull --ff-only origin fix/lab-flow-audit-20261023
docker compose -f lab_portal/docker-compose.yml up -d --build
```

No ejecutar `down -v`: elimina datos. Comprobar `/api/ping`, ejecutar el comprobador,
recrear el contenedor sin eliminar volúmenes y comparar informes. Este recorrido
Docker no se ejecutó en esta auditoría.

## Añadir un experimento

En una rama experimental: implementar módulo con CLI y salida `report.json`, añadir
parámetros/presets/dependencias al catálogo, adaptar `summaries.py`, añadir etiquetas
al frontend y pruebas. Ejecutar tests, publicar rama/PR, desplegar deliberadamente
la rama elegida y verificar SHA/catálogo/trabajo/informe/persistencia. Los tests fijan
las familias existentes; actualizar sus expectativas al incorporar una nueva.

```bash
python -m pip install -r lab_portal/requirements.txt httpx
python -m unittest discover -s tests -p 'test_lab_portal*.py' -v
```

## F8b

No se implementa ni se lanza F8b en esta auditoría. El usuario condicionó ese paso
al flujo comprobado, y faltan el despliegue y la persistencia tras reinicio del VPS.
El objetivo documentado sigue siendo adaptación online de la física bajo cambios
ocultos, distinto de la inferencia de régimen ya implementada en F8.
