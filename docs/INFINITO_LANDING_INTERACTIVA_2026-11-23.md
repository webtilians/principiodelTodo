# Landing interactiva de INFINITO (23/11/2026)

Página para presentar INFINITO e INFINITO Teams a inversores: <https://infinitomemory.com/landing/>.
Implementa la propuesta y el brief técnico del 23/11/2026 (escenas de dispositivos alrededor de un cerebro central; al
pulsar una, el dato viaja al cerebro, queda recordado y, si procede, vuelve como propuesta o acción).

No es un experimento ni cambia el núcleo (`src/infinito3/`): es una página estática en `landing/` que Caddy sirve
directamente desde la copia del repositorio que ya tiene el servidor.

## Qué hay

| Fichero | Para qué |
|---|---|
| `landing/index.html` | Estructura y secciones fijas: Cómo funciona, Por qué ahora, Hoy y mañana, Teams, Confianza, CTA. |
| `landing/escenas.js` | **Textos y escenas** del escenario interactivo. JSON estricto dentro de un `window.INFINITO_LANDING = {…}`. |
| `landing/landing.js` | Dibuja tarjetas, cables y paquetes; reproduce cada escena; recorrido guiado y modo expo. |
| `landing/landing.css` | Estilo (oscuro con brillos cian) y diseño adaptable a escritorio, tableta y móvil. |
| `landing/assets/` | La imagen del cerebro del portal y el icono ∞. |
| `remote/deploy/vps/infinito-remote.caddy` | `redir /landing → /landing/` y `handle_path /landing/*` con una CSP que solo permite los propios ficheros. |
| `tests/test_landing_v3.py` | Contratos: 4–6 escenas, cada una con su estado y nota, nada externo, nada en línea, la ruta en Caddy. |

## Escenas y honestidad

Cada escena lleva un estado visible en la tarjeta y en el panel, y una nota que explica el límite:

- **Hoy** (funciona en infinitomemory.com): Identidad, Ordenador.
- **Piloto**: INFINITO Teams (organizaciones, espacios cifrados, roles y permisos por app; la detección de tareas
  olvidadas es visión y así lo dice la nota).
- **Visión · demostración conceptual**: Gafas/foto (guardar fotos sí existe, por invitación), Jardín, Robot.

Las pruebas impiden marcar como «hoy» las gafas, el jardín o el robot, y exigen que toda escena de visión lo diga en
su nota. Las cifras de la sección «Hoy y mañana» son las mismas que publica la portada (Bemula-Lite 45,9 frente a 26,3;
79 ramas, 50+ protocolos, 750+ ejecuciones).

## Uso en el stand

- Pulsar **Ver recorrido** reproduce las seis escenas seguidas; el robot recupera la foto de Tailandia que guardaron
  las gafas.
- **`/landing/?expo`** arranca solo el recorrido y lo repite sin fin, vaciando la memoria en cada vuelta. Si alguien
  pulsa una escena, el recorrido se para y vuelve a arrancar tras 45 s sin tocar nada.
- Funciona también abriendo `landing/index.html` como archivo, sin conexión (scripts clásicos, sin módulos ni CDN).
- Con «reducir movimiento» activado en el sistema (Windows: *Efectos de animación* desactivado) se paran las
  animaciones decorativas, pero las escenas pulsadas siguen animándose; en `?expo` se anima todo.

## Para cambiar textos o escenas

Editar `landing/escenas.js` respetando el JSON estricto (comillas dobles, sin comas finales) y ejecutar
`python -m pytest tests/test_landing_v3.py -q`. Campos de una escena: `id`, `side` (`left`/`right`), `art`
(`persona`, `gafas`, `portatil`, `jardin`, `robot`, `empresa`), `label`, `title`, `event`, `action`, `status`,
`statusNote`, `packet`, `packetArt` (opcional), `memory` o `recalls`, `think` (líneas del análisis), `output`
(`to`: `apps`, `self` u otra escena; `packet`; `result`) y `explain`.

## Desviaciones del brief

- **Sin React/Next/Tailwind/Framer Motion.** El brief los daba como preferencia para un prototipo. El servidor instala
  cada commit de `master` con `git pull` y no tiene paso de compilación ni Node; la página tenía que funcionar además
  sin red en el stand y bajo una CSP estricta. Se hizo con HTML, CSS, SVG y JavaScript sin dependencias, con las
  escenas y textos en un objeto JSON configurable, que era el requisito de fondo.
- **«Solicitar demo»** abre un diálogo que invita a pedir la demo al equipo y lleva a crear la memoria o a la portada.
  No hay todavía un contacto público (correo o formulario) que poner; cuando exista, basta con añadirlo al diálogo de
  `index.html`.
