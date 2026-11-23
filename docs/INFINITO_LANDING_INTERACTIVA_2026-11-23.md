# Landing interactiva de INFINITO (23/11/2026)

Página de campaña para presentar INFINITO e INFINITO Teams a inversores: <https://infinitomemory.com/landing/>.
Parte de la propuesta y el brief técnico del 23/11/2026 (escenas de dispositivos alrededor de un cerebro central; al
pulsar una, el dato viaja al cerebro, queda recordado y vuelve como propuesta o acción). El mismo día se decidió
enfocarla como campaña publicitaria: todo el potencial de INFINITO y sus objetivos futuros, sin etiquetas de estado por
escena.

No es un experimento ni cambia el núcleo (`src/infinito3/`): es una página estática en `landing/` que Caddy sirve
directamente desde la copia del repositorio que ya tiene el servidor.

## Qué hay

| Fichero | Para qué |
|---|---|
| `landing/index.html` | Secciones: Cómo funciona, Por qué ahora (con la frase «INFINITO será parte de ti»), Teams, Hoja de ruta, Confianza, CTA. |
| `landing/escenas.js` | **Textos y escenas** del escenario interactivo. JSON estricto dentro de un `window.INFINITO_LANDING = {…}`. |
| `landing/landing.js` | Dibuja tarjetas, cables y paquetes; reproduce cada escena; recorrido guiado y modo expo. |
| `landing/landing.css` | Estilo (oscuro con brillos cian) y diseño adaptable a escritorio, tableta y móvil. |
| `landing/assets/` | La imagen del cerebro del portal y el icono ∞ (el `.svg` va con `git add -f`: `.gitignore` excluye `*.svg`). |
| `remote/deploy/vps/infinito-remote.caddy` | `redir /landing → /landing/` y `handle_path /landing/*` con una CSP que solo permite los propios ficheros. |
| `tests/test_landing_v3.py` | Contratos: 4–6 escenas reproducibles, nada externo ni en línea, anclas válidas, la ruta en Caddy. |

## Hoja de ruta que muestra

01 Memoria universal (conseguido) · 02 Motor cognitivo (en marcha) · 03 INFINITO Teams (en marcha) · 04 Apps móviles
(siguiente) · 05 Hogar y dispositivos · 06 Robots (objetivos) · 07 El agente total (horizonte). Sin fechas: si se
quieren poner años objetivo, van en el `road-state` de cada fase en `index.html`.

Las cifras son las que publica la portada (Bemula-Lite 45,9 frente a 26,3; 79 ramas, 50+ protocolos, 750+ ejecuciones).

## Uso en el stand

- **Ver recorrido** reproduce las seis escenas seguidas; el robot recupera la foto de Tailandia que guardaron las gafas.
- **`/landing/?expo`** arranca solo el recorrido y lo repite sin fin, vaciando la memoria en cada vuelta. Si alguien
  pulsa una escena, el recorrido se para y vuelve a arrancar tras 45 s sin tocar nada.
- Funciona también abriendo `landing/index.html` como archivo, sin conexión (scripts clásicos, sin módulos ni CDN).
- Con «reducir movimiento» activado en el sistema (Windows: *Efectos de animación* desactivado) se paran las
  animaciones decorativas, pero las escenas pulsadas siguen animándose; en `?expo` se anima todo.

## Para cambiar textos o escenas

Editar `landing/escenas.js` respetando el JSON estricto (comillas dobles, sin comas finales) y ejecutar
`python -m pytest tests/test_landing_v3.py -q`. Campos de una escena: `id`, `side` (`left`/`right`), `art`
(`persona`, `gafas`, `portatil`, `jardin`, `robot`, `empresa`), `label`, `title`, `event`, `action`, `packet`,
`packetArt` (opcional), `memory` o `recalls`, `think` (líneas del análisis), `output` (`to`: `apps`, `self` u otra
escena; `packet`; `result`) y `explain`.

## Desviaciones del brief

- **Sin React/Next/Tailwind/Framer Motion.** El servidor instala cada commit de `master` sin paso de compilación ni
  Node, y la página tenía que funcionar sin red en el stand y bajo una CSP estricta. HTML, CSS, SVG y JavaScript sin
  dependencias, con escenas y textos en un objeto JSON configurable.
- **«Solicitar demo»** abre un diálogo que invita a pedir la demo y lleva a crear la memoria o a la portada. Falta un
  contacto público (correo o formulario); cuando exista, se añade al diálogo de `index.html`.
