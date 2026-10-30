# INFINITO F9 — Worker Windows (32 GB RAM) + laboratorio VPS

Estado: implementación en rama experimental. **No desplegado todavía en producción**.
Arquitectura: Windows hace *pull saliente* por HTTPS y ejecuta los entrenamientos
pesados; el VPS sólo gestiona trabajos y conserva informes/modelos.

## Requisitos

- PC Windows con 32 GB de RAM, Python 3.11, Git, PowerShell, Internet HTTPS.
- El VPS conserva su servicio FastAPI, Caddy, 1 CPU y 2 GB RAM compartidos.
- **No requiere abrir puertos en Windows ni dar acceso SSH al PC**.
- Dos claves distintas: \`LAB_API_TOKEN\` para el navegador (ya existente) y
  \`LAB_WORKER_TOKEN\` nueva, sólo para el proceso trabajador. **Nunca publiques
  claves en un issue, conversación o commit**.

## Paso 1. Preparar servidor (sin experimentos en marcha)

Revisar rama antes de desplegarla:

- [Cambios del worker](https://github.com/webtilians/principiodelTodo/tree/experiment/f9-windows-remote-worker)
- [Tests de seguridad y extremo a extremo](../../tests/test_f9_remote_worker.py)

Por SSH en el VPS, con privilegios de administrador:

~~~bash
sudo systemctl status infinito-lab --no-pager
sudo cat /etc/infinito-lab/branch
sudo cat /var/lib/infinito-lab-deploy/deployed
# Comprueba en el portal que NO haya trabajo en ejecución.
python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
~~~

El último comando genera una clave NUEVA (no es la del navegador).
Cópiala de forma privada a un gestor de contraseñas. Edita con
\`sudoedit /etc/infinito-lab/lab.env\` y añade estas dos líneas (sustituye el
marcador por la clave real generada; **no compartas el archivo**):

~~~text
LAB_F9_REMOTE_ENABLED=1
LAB_WORKER_TOKEN=TU_CLAVE_ALEATORIA_DE_64_CARACTERES
~~~

No cambies \`LAB_API_TOKEN\`; debe seguir siendo la clave de acceso a la web.

Luego, durante una ventana sin trabajos:

~~~bash
sudo env LAB_BRANCH=experiment/f9-windows-remote-worker infinito-lab-update
# Importante: el actualizador del VPS es una copia instalada fuera del repositorio.
# Reinstalar su versión corregida, para que respete los trabajos REMOTOS F9:
sudo bash /opt/infinito-lab/lab_portal/deploy/vps/setup.sh
sudo systemctl is-active infinito-lab infinito-lab-update.timer
sudo git -C /opt/infinito-lab rev-parse HEAD
sudo cat /var/lib/infinito-lab-deploy/deployed
~~~

\`setup.sh\` es idempotente y, según su configuración, conserva token y datos
preexistentes. Aun así, **haz copia de seguridad** de
\`/var/lib/infinito-lab/runs\` y de la configuración antes de actualizar.
El actualizador del timer protege trabajos remotos activos al reanudarse,
pero sigue existiendo una pequeña carrera entre detectar inactividad y
recibir una nueva petición; evita actualizaciones durante experimentos.

El servidor NO instalará PyTorch ni ejecutará entrenamientos F9: sólo
los despachará al PC por la cola remota.

## Paso 2. Preparar PC Windows

En PowerShell:

~~~powershell
git clone --branch experiment/f9-windows-remote-worker https://github.com/webtilians/principiodelTodo.git
cd principiodelTodo

# Primera ejecución: crea .venv-f9, instala Torch CPU + MiniGrid + SB3
powershell -ExecutionPolicy Bypass -File lab_portal/start_f9_windows.ps1
~~~

Si ya tenías el repositorio clonado:

~~~powershell
git fetch origin
git switch experiment/f9-windows-remote-worker
git pull --ff-only origin experiment/f9-windows-remote-worker
powershell -ExecutionPolicy Bypass -File lab_portal/start_f9_windows.ps1
~~~

El script pide la **clave del worker** sin mostrarla ni guardarla en disco
y empieza a escuchar la cola en el VPS. Queda activo mientras la terminal
esté abierta; no es un servicio de Windows instalado automáticamente.
Para ejecutar otra vez sin reinstalar dependencias:

~~~powershell
powershell -ExecutionPolicy Bypass -File lab_portal/start_f9_windows.ps1 -SkipInstall
~~~

\`-InstallOnly\` prepara dependencias y sale sin conectarse. Usa CPU para
evitar la instalación y el consumo de VRAM de paquetes CUDA innecesarios
en esta primera etapa; los modelos pequeños de MiniGrid pueden entrenar
de manera secuencial sin GPU. Si la instalación de PyTorch falla, no sigas
y revisa versión de Python y arquitectura.

**El hash Git debe coincidir exactamente con el del VPS**. Si cambia la
rama en GitHub mientras ambos procesos están activos, el worker no
aceptará trabajos de otra revisión. Finaliza trabajos, actualiza y reinicia
ambos, manteniendo los resultados antiguos.

## Paso 3. Iniciar el primer experimento en el navegador

Visita https://lab.178-105-103-4.sslip.io/ con tu **clave de navegador**
de siempre. En la cabecera debe decir **F9 PC CONECTADO**. Selecciona
**F9-A · MiniGrid Memory** y el perfil **PRUEBA**:

| Parámetro | Valor |
|---|---:|
| Semillas de entrenamiento | 1 |
| Pasos PPO por semilla | 128 |
| Episodios de validación | 3 |

Pulsa iniciar. El trabajo aparece en la cola del servidor, el PC lo reclama,
entrena **random, PPO y RecurrentPPO LSTM**, conserva checkpoints y devuelve
report.json y los logs. En la pestaña **Ejecuciones** aparecerá
\`completed\` y podrás descargar informe, comparativa y ZIP.

**128 pasos no implican aprendizaje útil**: esta prueba sólo confirma el
circuito. No abras aún el test retenido. Una vez comprobado el circuito,
pasa a entrenamientos de desarrollo más amplios en tu PC.

## Seguridad y límites de diseño

- La cola admite SOLO \`f9_memory\`, \`f9_doorkey\` y \`f9_dynamic\`.
- Los parámetros se vuelven a validar en Windows; no llegan shell scripts.
- La comunicación usa HTTPS y una clave de worker separada.
- Se exige mismo SHA Git en servidor y PC, y un lease por trabajo.
- ZIPs limitados a 64 MiB, rutas limitadas a informes, manifiesto y checkpoints;
  se rechaza recorrido de carpetas y archivos no previstos.
- La interfaz mantiene **una ejecución total simultánea**: también protege
  CPU y recursos del VPS.
- Si se apaga el PC, el trabajo se marca interrumpido tras cinco minutos sin
  latido; si la cola no encuentra trabajador en 30 minutos, caduca. Los datos
  parciales permanecen en Windows en
  \`observatory_results/windows_worker/<id>/output\`. Reinicia el experimento
  desde el portal cuando el PC vuelva a estar disponible.
- El trabajador detiene el subproceso al recibir cancelación de la web, o
  al superar el límite de duración configurado.
- Los resultados enviados al portal son sólo de **desarrollo**, sin
  abrir ni optimizar contra el test retenido.
- Las máquinas deben disponer de almacenamiento suficiente para los ZIP
  de checkpoints. La subida completa se limita a 64 MiB por trabajo.

## Comprobaciones y diagnóstico

- En servidor (sin imprimir claves): \`infinito-lab-status\`.
- GitHub CI: tests \`test_f9_remote_worker.py\` y
  \`test_f9_windows_agent.py\` (este último entrena PPO real).
- Windows no conecta: verifica URL HTTPS, clave, reloj del sistema, firewall
  saliente y que \`LAB_F9_REMOTE_ENABLED=1\`.
- El PC informa «CÓDIGO DISTINTO»: actualiza Git en una ventana sin trabajos.
- Si el worker aparece pero no hay tarea: ejecuciones en cola bajo otra
  revisión, tarea expirada o no hay trabajo pendiente.
- Si aparece \`failed\`: consulta logs y la carpeta local de ejecución;
  comparte solo el log sin tokens para poder depurarlo.

## Próxima etapa científica

Este trabajador es infraestructura: no demuestra que INFINITO sea mejor que
PPO/LSTM. Tras probar el flujo, entrenar baselines fuertes y, sólo después,
integrar memoria INFINITO con ablaciones y presupuesto de cálculo equivalente.
