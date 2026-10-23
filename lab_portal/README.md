# INFINITO Laboratory — portal de experimentos

Portal web **real y autoalojado**, con FastAPI y ejecutor de procesos independientes. Permite ejecutar **F4, F5, F6, F7, F7b y F8** con más semillas y episodios. F8 combina inferencia del estado oculto, planificación predictiva e inspección costosa en un mundo simulado separado de V5.1.

[Plan de F8](../docs/INFINITO_NEXT_STEP_F8_2026-10-23.md). [F8 — Primeros resultados verificados](../docs/INFINITO_F8_INTEGRATED_WORLD_RESULTS_2026-10-23.md). El siguiente objetivo F8b (adaptación física online bajo cambios desconocidos) sigue pendiente.

## Ejecutar en tu PC o servidor

Desde la **raíz del repositorio** (rama `experiment/f8-laboratory-portal`):

```bash
python -m pip install -r lab_portal/requirements.txt
# Linux / macOS
export LAB_API_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(36))')"
python -m lab_portal
```

En Windows PowerShell:

```powershell
py -m pip install -r lab_portal/requirements.txt
$env:LAB_API_TOKEN = (py -c "import secrets; print(secrets.token_urlsafe(36))")
py -m lab_portal
```

Abre **http://127.0.0.1:8765/** y pega el valor de `LAB_API_TOKEN` cuando te lo pida. Por defecto el servidor escucha **solo en localhost**, no está publicado en Internet. La clave queda en `sessionStorage` de esa pestaña del navegador, no se envía a GitHub.

### Docker en tu propio servidor

```bash
cp lab_portal/.env.example lab_portal/.env
# Edita LAB_API_TOKEN con un secreto aleatorio real
cd lab_portal
docker compose up -d --build
```

El contenedor enlaza el puerto **solo a 127.0.0.1 del servidor**. Para acceder desde otro ordenador usa túnel SSH o configura un proxy inverso con HTTPS, autenticación adecuada y control de acceso. **No publiques el puerto de este ejecutor directamente** ni expongas una clave de laboratorio a través de HTTP sin TLS. Si utilizas un túnel SSH: `ssh -L 8765:127.0.0.1:8765 usuario@servidor`, luego entra en `http://127.0.0.1:8765` desde tu equipo.

**Restricción Docker predeterminada**: 2 CPU, 4 GB RAM. Ajusta estos límites conscientemente si necesitas pruebas más largas. No requiere GPU para F4–F7b.

## Qué puedes hacer

- Seleccionar seis familias de experimentos, incluido F8 (el mundo integrado de inferencia de régimen y sensores; no es el motor histórico V5.1).
- Elegir preajustes **prueba / estándar / largo** y personalizar solo parámetros permitidos dentro de límites definidos.
- Lanzar un experimento por vez como **proceso independiente**, sin depender de que la pestaña permanezca abierta.
- Ver estado, terminal, errores, resultados tabulados y barras de comparación por escenario. Descargar reporte JSON o ZIP de logs y resultados con las semillas usadas.
- Cancelar un proceso activo y conservar datos parciales de investigación.
- Mantener todo separado por ID en `observatory_results/portal_runs/<uuid>/`; configura `LAB_DATA_DIR` si prefieres disco diferente, por ejemplo una unidad con mayor capacidad.

## Seguridad y reproducibilidad

El API exige `X-Lab-Token`, nunca ejecuta comandos introducidos por el navegador, no ofrece un terminal arbitrario y utiliza un catálogo de argumentos numéricos con límites. No hay acciones GitHub de escritura ni credenciales del repositorio en el servicio. Se permite **una ejecución simultánea**: es deliberado para no saturar el servidor. El proceso supervisor conserva los resultados aunque cierres el navegador, y limita la duración máxima mediante `LAB_MAX_SECONDS` (por defecto **4 horas**, configurable hasta 24 horas).

Se guarda el protocolo, parámetros, registro, informe completo y estado de salida. El JSON de cada experimento conserva sus semillas y métricas propias. El portal **no compara escalas científicas distintas entre experimentos** ni afirma conciencia basándose en indicadores.

Ten en cuenta: la interrupción abrupta del propio servidor puede matar los trabajadores; en ese caso el panel marca la ejecución como interrumpida. Cancelar un proceso finaliza también sus hijos en sistemas POSIX.

## Comprobación técnica

```bash
python -m unittest discover -s tests -p 'test_lab_portal.py' -v
```

El workflow `INFINITO Lab Portal` ejecuta esas pruebas en GitHub Actions. Para generar resultados largos desde la web, el portal debe estar ejecutándose en tu ordenador o en tu servidor; **GitHub Pages es estático y no puede lanzar estos procesos por sí solo**.


## Ejecutar F8 desde el portal

Selecciona **F8 · Mundo activo integrado** en el catálogo. Prueba primero el
perfil **rápido**. Después, elige estándar o largo y ajusta la semilla inicial
para ejecutar replicaciones independientes. El panel muestra la recompensa
neta de las ocho políticas por escenario y permite descargar el informe
completo con diferencias emparejadas por semilla.

**Actualizar el portal si ya lo habías arrancado:** detén el servidor web,
abre una terminal en la raíz del repositorio y ejecuta:

~~~bash
git switch experiment/f8-laboratory-portal
git pull origin experiment/f8-laboratory-portal
python -m lab_portal
~~~

Si instalaste el portal usando únicamente el ZIP anterior, necesitarás
descargar/actualizar el repositorio completo: el paquete previo contenía
el portal, pero no necesariamente los módulos de simulación. No actualices
el código mientras haya una simulación en curso: termina y descarga primero
sus resultados.

**F8 es exploratorio y tiene limitaciones explícitas.** Aprende parámetros
de transiciones mediante calibración offline y actualiza online su creencia
sobre un régimen oculto. Todavía NO modifica online el modelo de la física,
NO descubre regímenes arbitrarios y NO demuestra conciencia subjetiva.
