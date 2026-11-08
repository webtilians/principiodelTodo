"""The portal someone sees once signed in: a sidebar, a top bar and a vault home drawn after INFINITO's design.

Everything shown is real: the counts, the recent memories, the vault's names and the activity come from the person's
own memory, and the only picture is the brain. Each bubble in it is a neuron for one kind of memory: a plain link that
lights up under the pointer. CSS does that, so these pages still run no script, which keeps a memory's words from ever
running as code on a page that has just been given a password."""
import html
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlencode

BRAIN_FILE = Path(__file__).parent / "assets" / "brain.webp"
BRAIN_PATH = "/memory/brain.webp"
PHOTO_PATH = "/memory/photo"

TEXT = {
    "es": {
        "nav_home": "Inicio", "nav_vault": "Bóveda", "nav_memories": "Recuerdos", "nav_projects": "Proyectos",
        "nav_search": "Búsqueda", "nav_settings": "Ajustes", "find": "Buscar en tu memoria...",
        "pending_alert": "Lo pendiente", "help": "Cómo conectar tus IAs", "account": "Tu cuenta",
        "vault_lead": "Tu memoria segura e inteligente.",
        "vault_about": "Un espacio cifrado donde tus recuerdos, conocimientos y claves importantes se conectan a través "
                       "de una red neuronal personal.",
        "vault_on": "Bóveda activa", "vault_off": "Memoria sellada",
        "stat_nodes": "Nodos de memoria", "stat_memories": "Recuerdos", "stat_cipher": "Cifrado AES-GCM<br>en reposo",
        "ribbon": ("Neuronas", "Recuerdos", "Conocimiento", "Seguridad"),
        "recent": "Memorias recientes", "see_all": "Ver todas", "keys": "Claves protegidas", "activity": "Actividad",
        "see_history": "Ver historial", "activity_saved": "Nuevo recuerdo agregado", "open_key": "Abrir",
        "key_kind": "Clave protegida", "no_keys": "La bóveda es por invitación.", "no_keys_yet": "Aún no hay claves.",
        "no_memories": "Lo que le cuentes a tus IAs aparecerá aquí.",
        "now": "Ahora", "yesterday": "Ayer", "hours_ago": "Hace {n} h", "days_ago": "Hace {n} d",
        "map": "Tus recuerdos por tipo", "see_kind": "Pulsa para verlos",
        "projects_title": "Proyectos", "no_projects": "Ningún recuerdo tiene cliente o proyecto todavía.",
        "last_activity": "Última actividad", "quiet": "{n} días sin actividad", "still_open": "Sigue abierto",
        "settings_title": "Ajustes", "settings_account": "Tu cuenta", "settings_sealed":
        "Tu memoria está sellada en el disco con AES-256-GCM, bajo una clave que solo se abre con tu contraseña.",
    },
    "en": {
        "nav_home": "Home", "nav_vault": "Vault", "nav_memories": "Memories", "nav_projects": "Projects",
        "nav_search": "Search", "nav_settings": "Settings", "find": "Search your memory...",
        "pending_alert": "What is pending", "help": "How to connect your AIs", "account": "Your account",
        "vault_lead": "Your safe, smart memory.",
        "vault_about": "An encrypted space where your memories, knowledge and important keys connect through a personal "
                       "neural network.",
        "vault_on": "Vault active", "vault_off": "Memory sealed",
        "stat_nodes": "Memory nodes", "stat_memories": "Memories", "stat_cipher": "AES-GCM encryption<br>at rest",
        "ribbon": ("Neurons", "Memories", "Knowledge", "Security"),
        "recent": "Recent memories", "see_all": "See all", "keys": "Protected keys", "activity": "Activity",
        "see_history": "See history", "activity_saved": "New memory added", "open_key": "Open",
        "key_kind": "Protected key", "no_keys": "The vault is by invitation.", "no_keys_yet": "No keys yet.",
        "no_memories": "What you tell your AIs will appear here.",
        "now": "Now", "yesterday": "Yesterday", "hours_ago": "{n} h ago", "days_ago": "{n} d ago",
        "map": "Your memories by kind", "see_kind": "Click to see them",
        "projects_title": "Projects", "no_projects": "No memory has a client or project yet.",
        "last_activity": "Last activity", "quiet": "{n} days with no activity", "still_open": "Still open",
        "settings_title": "Settings", "settings_account": "Your account", "settings_sealed":
        "Your memory is sealed on disk with AES-256-GCM, under a key that only your password opens.",
    },
}

# Where each kind sits: on the bubbles of the brain picture, which is 725 by 455.
NEURON_SPOTS = {"pending": (206, 114, 44), "agreement": (375, 92, 33), "fact": (515, 145, 40), "assumption": (612, 183, 27),
                "decision": (258, 217, 42), "proposal": (107, 253, 39), "to_verify": (395, 285, 40), "none": (612, 264, 49)}
KIND_COLOURS = {"fact": "#67e8f9", "decision": "#a78bfa", "agreement": "#60a5fa", "pending": "#fbbf24",
                "proposal": "#4ade80", "assumption": "#f472b6", "to_verify": "#fb923c", "none": "#34d399"}
KIND_ICONS = {"fact": "doc", "decision": "check", "agreement": "people", "pending": "clock", "proposal": "bulb",
              "assumption": "help", "to_verify": "search", "none": "user"}

_ICONS = {
    "home": '<path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v9.5h4.5v-5.5h4v5.5h4.5V10"/>',
    "lock": '<rect x="5" y="10.5" width="14" height="10" rx="2.5"/><path d="M8 10.5V7.5a4 4 0 0 1 8 0v3"/><path d="M12 14.5v2.5"/>',
    "layers": '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12.5 9 5 9-5"/><path d="m3 17 9 4.5 9-4.5" opacity=".7"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4.5l2 2.5H19a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m20.5 20.5-5.3-5.3"/>',
    "gear": '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.6M12 18.6v2.6M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2.8 12h2.6M18.6 12h2.6M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/>',
    "bell": '<path d="M6 10a6 6 0 0 1 12 0c0 6 2.2 7.5 2.2 7.5H3.8S6 16 6 10Z"/><path d="M10 20.5a2.2 2.2 0 0 0 4 0"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.4 9.4a2.7 2.7 0 1 1 3.9 2.4c-.9.5-1.3 1-1.3 2"/><path d="M12 17v.2"/>',
    "shield": '<path d="M12 2.8 20 6v5.5c0 5-3.4 8.3-8 9.7-4.6-1.4-8-4.7-8-9.7V6l8-3.2Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    "key": '<circle cx="8" cy="15.5" r="4.5"/><path d="m11.2 12.3 8.3-8.3M16 7.5l2.5 2.5M13.5 10l2 2"/>',
    "wallet": '<rect x="3" y="6" width="18" height="13.5" rx="3"/><path d="M3 10.5h18"/><path d="M15.5 15h2.5"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5v13c0 1.7 3.6 3 8 3s8-1.3 8-3v-13"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "pulse": '<path d="M2 12h4l2.5-7 4.5 15 3-8h6"/>',
    "network": '<circle cx="12" cy="12" r="2.6"/><circle cx="4.5" cy="6" r="2"/><circle cx="19.5" cy="6" r="2"/><circle cx="4.5" cy="18" r="2"/><circle cx="19.5" cy="18" r="2"/><path d="m6.2 7.2 3.8 3M17.8 7.2l-3.8 3M6.2 16.8l3.8-3M17.8 16.8l-3.8-3"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="2.8"/>',
    "arrow": '<path d="M4 12h15"/><path d="m13.5 6.5 5.5 5.5-5.5 5.5"/>',
    "doc": '<path d="M6 3h8l4.5 4.5V21H6V3Z"/><path d="M14 3v4.5h4.5M9 12h6M9 16h6"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="m8 12 3 3 5-6"/>',
    "people": '<circle cx="8.5" cy="9" r="3.2"/><circle cx="16.5" cy="10" r="2.6"/><path d="M2.8 20c.5-3.6 2.8-5.5 5.7-5.5s5.2 1.9 5.7 5.5"/><path d="M15.8 14.6c2.8-.3 5 1.6 5.4 5.4"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5.5l3.5 2"/>',
    "bulb": '<path d="M12 3a6.5 6.5 0 0 0-4 11.6c.8.7 1 1.4 1 2.4h6c0-1 .2-1.7 1-2.4A6.5 6.5 0 0 0 12 3Z"/><path d="M9.5 20.5h5"/>',
    "user": '<circle cx="12" cy="8.5" r="4"/><path d="M4.5 20.5c.8-4.2 3.8-6.5 7.5-6.5s6.7 2.3 7.5 6.5"/>',
    "out": '<path d="M14 4h5v16h-5"/><path d="M4 12h11"/><path d="m8.5 7.5-4.5 4.5 4.5 4.5"/>',
    "image": '<rect x="3" y="4.5" width="18" height="15" rx="2.5"/><circle cx="8.5" cy="9.5" r="1.7"/><path d="m3.5 17.5 5-4.5 3.5 3 3.5-3.5 5 5"/>',
}

STYLE = """
:root { font-size: clamp(11.5px, .957vw, 20px); }
body.app { min-height: 100vh; font-size: 1rem; background:
    radial-gradient(ellipse 34% 9% at 86% 60%, rgba(124, 92, 246, .55), transparent 72%),
    radial-gradient(ellipse 30% 12% at 30% 57%, rgba(84, 86, 218, .42), transparent 74%),
    radial-gradient(ellipse 30% 9% at 56% 97%, rgba(24, 170, 220, .26), transparent 74%),
    linear-gradient(180deg, #050a1a, #040c20 60%, #041125); background-attachment: fixed; }
.frame { display: grid; grid-template-columns: 15rem minmax(0, 1fr); min-height: 100vh; }
.side { position: relative; padding: 1.6rem .15rem 1.2rem .75rem; display: flex; flex-direction: column; overflow: hidden;
  background: radial-gradient(ellipse 110% 20% at 30% 69%, rgba(30, 130, 240, .42), transparent 72%),
    linear-gradient(180deg, #0f1a38, #101f47 74%, #0a1a3e); border-right: 1px solid rgba(120, 150, 255, .16); }
.logo { display: block; width: 7.1rem; margin: 0 0 4.1rem 3.94rem; }
.logo svg { display: block; width: 100%; height: auto; overflow: visible; }
.nav { display: flex; flex-direction: column; gap: .7rem; position: relative; }
.nav a { display: flex; align-items: center; gap: 1.7rem; height: 3.5rem; padding: 0 1.1rem 0 1.25rem; color: #d8e0f6;
  font-size: 1.16rem; text-decoration: none; border: 1px solid transparent; border-radius: .95rem;
  transition: background .15s, color .15s; }
.nav a .ic { width: 1.8rem; height: 1.8rem; color: #b9c8ef; flex: none; }
.nav a:hover { background: rgba(130, 160, 255, .12); color: #fff; }
.nav a.on { color: #fff; font-weight: 700; border-color: rgba(120, 220, 255, .75);
  background: linear-gradient(90deg, #7d52f4, #3f4cc4 52%, #164880);
  box-shadow: 0 0 1.3rem rgba(125, 82, 244, .65), inset 0 0 .9rem rgba(160, 200, 255, .24); }
.nav a.on .ic { color: #fff; }
.nav .badge { margin-left: auto; min-width: 1.6rem; padding: .05rem .45rem; font-size: .85rem; text-align: center; color: #240a3a;
  background: #e879f9; border-radius: 99rem; }
.side .leave { margin-top: auto; position: relative; }
.side .leave button { display: flex; align-items: center; gap: .8rem; width: 100%; margin: 0; padding: .65rem 1.25rem;
  color: #aab6d8; font-size: .95rem; font-weight: 500; background: transparent; border: 0; }
.side .leave button:hover { color: #fff; filter: none; }
.side .leave .ic { width: 1.3rem; height: 1.3rem; }
.ic { display: inline-block; fill: none; stroke: currentColor; stroke-width: 1.7; stroke-linecap: round;
  stroke-linejoin: round; vertical-align: middle; }
.work { min-width: 0; display: flex; flex-direction: column; }
.top { display: flex; align-items: center; height: 5.3rem; padding: 0 1.75rem 0 2.65rem;
  border-bottom: 1px solid rgba(120, 150, 255, .16); }
.word { width: 22rem; flex: none; color: #fff; font-size: 1.95rem; font-weight: 300; letter-spacing: .34em;
  text-decoration: none; }
.find { position: relative; width: 44rem; margin: 0; }
.find .ic { position: absolute; left: 1.25rem; top: 50%; width: 1.55rem; height: 1.55rem; transform: translateY(-50%);
  color: #dbe5ff; pointer-events: none; }
.find input { width: 100%; height: 2.95rem; padding: 0 1rem 0 3.5rem; font-size: 1.05rem; color: #fff;
  background: rgba(24, 44, 92, .55); border: 1px solid rgba(120, 170, 255, .42); border-radius: 99rem; }
.find input::placeholder { color: #c3cfee; opacity: 1; }
.tools { margin-left: auto; display: flex; align-items: center; gap: 2.5rem; }
.tools a { position: relative; display: block; color: #e6edff; }
.tools .ic { display: block; width: 1.85rem; height: 1.85rem; }
.tools .dot { position: absolute; top: -.1rem; right: -.1rem; width: .55rem; height: .55rem; background: #e879f9;
  border-radius: 50%; box-shadow: 0 0 .5rem #e879f9; }
.tools a.me { margin-left: .55rem; display: grid; place-items: center; width: 2.95rem; height: 2.95rem; color: #fff; font-weight: 600; font-size: 1.15rem;
  text-decoration: none; text-transform: uppercase; background: radial-gradient(circle at 50% 30%, #6fb8ff, #1f4f9f 72%);
  border: 2px solid rgba(160, 210, 255, .75); border-radius: 50%; box-shadow: 0 0 .9rem rgba(90, 170, 255, .55); }
main.dash { flex: 1; width: 100%; max-width: none; margin: 0; padding: 0 1.25rem 1.65rem 2rem; }
.hero { position: relative; display: grid; grid-template-columns: 21.125rem 45.3rem 2.875rem 17rem; min-height: 31.06rem; }
.hero > .stats { grid-column: 4; }
.lead { padding: 3.95rem 0 0 1.75rem; }
.lead .title { display: flex; align-items: center; gap: .9rem; margin: 0 0 .9rem -.3rem; }
.lead .title .ic { width: 3.05rem; height: 3.05rem; color: #67e8f9; stroke-width: 1.5; filter: drop-shadow(0 0 .45rem rgba(103, 232, 249, .7)); }
.lead h1 { margin: 0; font-size: 3.12rem; line-height: 1; font-weight: 700; color: #fff; }
.lead .sub { margin: 0 0 1.9rem; font-size: 1.32rem; color: #e4ebff; white-space: nowrap; }
.lead .about { margin: 0 0 2rem; font-size: 1.03rem; line-height: 1.5rem; color: #cdd8f4; }
.pill { display: inline-flex; align-items: center; gap: .85rem; padding: .55rem 1.2rem; font-size: 1.02rem; font-weight: 600;
  color: #5ef2c3; background: rgba(8, 40, 56, .75); border: 1px solid rgba(94, 242, 195, .42); border-radius: 99rem; }
.pill .ic { width: 1.5rem; height: 1.5rem; }
.brainbox { position: relative; margin: -.3rem 0 0; aspect-ratio: 725 / 455; align-self: start; }
.brainbox img { display: block; width: 100%; height: 100%; mix-blend-mode: screen; filter: brightness(1.14) saturate(1.12);
  -webkit-mask-image: radial-gradient(ellipse 57% 68% at 50% 45%, #000 84%, transparent 100%);
  mask-image: radial-gradient(ellipse 57% 68% at 50% 45%, #000 84%, transparent 100%); }
.hit { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; }
.neuron { cursor: pointer; }
.neuron:focus { outline: none; }
.neuron .ring { fill: #fff; fill-opacity: 0; stroke: var(--c); stroke-opacity: 0; stroke-width: 3; transform-box: fill-box;
  transform-origin: center; transition: transform .15s, stroke-opacity .15s, fill-opacity .15s; }
.neuron .tip { opacity: 0; pointer-events: none; transition: opacity .15s; }
.neuron .tip text { text-anchor: middle; fill: #fff; paint-order: stroke; stroke: #050a1a; stroke-width: 5px;
  stroke-linejoin: round; }
.neuron .tip .name { font-size: 19px; font-weight: 700; }
.neuron .tip .about { font-size: 14px; font-weight: 500; fill: #dbe5ff; }
.neuron:hover .ring, .neuron:focus-visible .ring { transform: scale(1.14); stroke-opacity: 1; fill-opacity: .1;
  filter: drop-shadow(0 0 7px var(--c)); }
.neuron:hover .tip, .neuron:focus-visible .tip { opacity: 1; }
.stats { display: flex; flex-direction: column; gap: 1.5rem; padding: 4.7rem 0 0; }
.stat { display: flex; align-items: center; gap: 1.45rem; min-height: 5.3rem; padding: .6rem 1.4rem;
  background: linear-gradient(135deg, rgba(26, 44, 80, .82), rgba(9, 21, 40, .78));
  border: 1px solid rgba(100, 150, 230, .38); border-radius: 1.3rem; box-shadow: 0 0 1.3rem rgba(60, 120, 255, .14); }
.stat .ic { flex: none; width: 3rem; height: 3rem; color: #5eead4; stroke-width: 1.4;
  filter: drop-shadow(0 0 .35rem rgba(94, 234, 212, .6)); }
.stat b { display: block; font-size: 1.6rem; font-weight: 700; line-height: 1.75rem; color: #fff; }
.stat span { display: block; font-size: 1rem; line-height: 1.2rem; color: #d6e0f8; }
.stat.cipher { min-height: 6rem; }
.ribbon { position: absolute; left: 18.9rem; right: 19.5rem; bottom: 0; display: flex; justify-content: center; gap: 2.2rem;
  padding: .55rem 0 .5rem; font-size: .84rem; letter-spacing: .3em; text-transform: uppercase; color: #bcd2f7;
  border-top: 1px solid rgba(120, 190, 255, .55); border-radius: 1.3rem 1.3rem 0 0;
  background: linear-gradient(180deg, rgba(30, 120, 220, .16), transparent); }
.ribbon i { font-style: normal; color: #7fa3d9; }
.panels { display: grid; grid-template-columns: 28.5rem 27.7rem 27.7rem; gap: 1.2rem; margin-top: 1.45rem; }
.panel { padding: .8rem 1rem .95rem; background: linear-gradient(160deg, rgba(29, 45, 74, .84), rgba(22, 35, 66, .74));
  border: 1px solid rgba(100, 150, 230, .34); border-radius: 1.3rem; box-shadow: 0 0 1.4rem rgba(60, 120, 255, .12); }
.panel.teal { border-color: rgba(70, 200, 220, .42); } .panel.violet { border-color: rgba(150, 120, 245, .42); }
.panel > header { display: flex; align-items: center; gap: .95rem; margin: 0 .55rem .7rem .75rem; min-height: 2.6rem; }
.panel > header .ic { width: 2.3rem; height: 2.3rem; color: #e6edff; stroke-width: 1.5; }
.panel.teal > header .ic { color: #5eead4; }
.panel h2 { margin: 0; font-size: 1.32rem; font-weight: 700; color: #fff; }
.panel > header a { margin-left: auto; display: inline-flex; align-items: center; gap: .55rem; font-size: .96rem; color: #8db8ff;
  text-decoration: none; white-space: nowrap; }
.panel > header a .ic { width: 1.05rem; height: 1.05rem; color: inherit; stroke-width: 2; }
.panel > header a:hover { color: #fff; }
.rows { display: flex; flex-direction: column; gap: .5rem; }
.item { display: flex; align-items: center; gap: 1rem; min-height: 3.55rem; padding: .3rem .6rem .3rem .25rem; color: inherit;
  text-decoration: none; background: rgba(14, 26, 54, .62); border: 1px solid rgba(110, 150, 230, .16); border-radius: .85rem;
  transition: border-color .15s, background .15s; }
a.item:hover { border-color: rgba(130, 190, 255, .6); background: rgba(24, 42, 84, .75); }
.tile { flex: none; display: grid; place-items: center; width: 5.35rem; height: 2.95rem; color: var(--c);
  background: radial-gradient(circle at 30% 20%, color-mix(in srgb, var(--c) 42%, transparent), transparent 70%),
    linear-gradient(135deg, #1b2c58, #0d1834); border-radius: .55rem; }
.tile .ic { width: 1.6rem; height: 1.6rem; }
.tile img, .thumb img { display: block; width: 100%; height: 100%; object-fit: cover; border-radius: .55rem; }
.thumb { flex: none; display: block; width: 6rem; height: 4.5rem; background: #0d1834; border-radius: .55rem; }
.large { margin: 0 0 1rem; }
.large img { display: block; max-width: 100%; max-height: 70vh; width: auto; height: auto; margin: 0 auto; border-radius: .8rem; }
.add { margin: 0 0 1.2rem; padding: .7rem 1rem; border: 1px solid var(--line); border-radius: .85rem; }
.add summary { display: flex; align-items: center; gap: .6rem; color: var(--accent); font-weight: 600; cursor: pointer; }
.add summary .ic { width: 1.3rem; height: 1.3rem; }
.add form { margin-top: .6rem; }
.add input[type=file] { width: 100%; padding: .55rem 0; color: inherit; background: transparent; border: 0; }
.what { min-width: 0; flex: 1; }
.what strong { display: block; overflow: hidden; font-size: 1.04rem; font-weight: 600; line-height: 1.35rem; color: #fff;
  text-overflow: ellipsis; white-space: nowrap; }
.what small { display: block; overflow: hidden; font-size: .91rem; line-height: 1.2rem; color: #c3cfee; text-overflow: ellipsis;
  white-space: nowrap; }
.chip { flex: none; display: inline-flex; align-items: center; gap: .55rem; max-width: 7.4rem; padding: .32rem .75rem;
  overflow: hidden; font-size: .88rem; color: #dbe5ff; white-space: nowrap; background: rgba(46, 40, 110, .72); border-radius: 99rem; }
.chip .ic { flex: none; width: 1.15rem; height: 1.15rem; color: #c4b5fd; }
.key { padding-left: .95rem; }
.key > .ic { flex: none; width: 2.1rem; height: 2.1rem; margin-right: .35rem; color: #67e8f9; stroke-width: 1.5; }
.dots { flex: none; margin: 0 1rem; font-size: 1rem; letter-spacing: .26em; color: #e6edff; }
.key form { margin: 0; flex: none; }
.key button { display: block; width: auto; margin: 0; padding: .3rem; color: #9fc0ff; background: transparent; border: 0; }
.key button:hover { color: #fff; filter: none; }
.key button .ic { display: block; width: 1.45rem; height: 1.45rem; }
.line { position: relative; display: flex; flex-direction: column; padding: .2rem 0 0; border: 1px solid rgba(110, 150, 230, .16);
  border-radius: .85rem; background: rgba(14, 26, 54, .34); }
.line::before { content: ""; position: absolute; left: 1.42rem; top: 1.6rem; bottom: 1.6rem; width: 2px;
  background: linear-gradient(180deg, #a78bfa, #22d3ee 55%, #60a5fa); }
.event { position: relative; display: flex; gap: .95rem; min-height: 3.75rem; padding: .35rem .85rem .3rem 3.05rem; color: inherit;
  text-decoration: none; border-top: 1px solid rgba(110, 150, 230, .12); }
.event:first-child { border-top: 0; }
.event::before { content: ""; position: absolute; left: .95rem; top: .75rem; width: 1rem; height: 1rem; background: #0b1530;
  border: .22rem solid var(--c); border-radius: 50%; box-shadow: 0 0 .55rem var(--c); }
.event .when { flex: none; font-size: .92rem; color: #c3cfee; }
.empty-note { margin: .4rem .75rem; color: #c3cfee; }
.page { max-width: 62rem; padding: 2rem 0 0; }
.page > h1 { margin: 0 0 1rem; font-size: 2rem; color: #fff; }
.sheet { padding: 1.5rem; background: linear-gradient(160deg, rgba(29, 45, 74, .84), rgba(22, 35, 66, .74));
  border: 1px solid rgba(100, 150, 230, .34); border-radius: 1.3rem; }
.project { padding: 1rem 0; border-top: 1px solid var(--line); }
.project:first-of-type { border-top: 0; padding-top: 0; }
.project h2 { margin: 0 0 .2rem; font-size: 1.15rem; }
.project h2 a { color: #fff; text-decoration: none; } .project h2 a:hover { color: var(--accent); }
.project .quiet-flag { color: #fbbf24; font-weight: 600; }
.project ul { margin: .5rem 0 0; padding-left: 1.2rem; }
@media (max-width: 62rem), (max-width: 800px) {
  :root { font-size: 16px; }
  .frame { grid-template-columns: minmax(0, 1fr); }
  .side { padding: .9rem .75rem .6rem; flex-direction: row; flex-wrap: wrap; align-items: center; gap: .4rem; }
  .logo { width: 3.4rem; margin: 0 .6rem 0 0; }
  .nav { flex-direction: row; flex-wrap: wrap; gap: .2rem; }
  .nav a { height: 2.4rem; padding: 0 .7rem; gap: .45rem; font-size: .92rem; }
  .nav a .ic { width: 1.2rem; height: 1.2rem; }
  .side .leave { margin: 0 0 0 auto; } .side .leave button { padding: .4rem .5rem; font-size: .85rem; }
  .top { height: auto; flex-wrap: wrap; gap: .6rem; padding: .7rem 1rem; }
  .word { width: auto; font-size: 1.3rem; }
  .find { order: 3; width: 100%; }
  .tools { gap: 1.1rem; padding: 0; }
  main.dash { padding: 0 1rem 1.2rem; }
  .hero { display: block; min-height: 0; }
  .lead { padding: 1.2rem 0 0; }
  .lead .title { margin-left: 0; }
  .lead .sub { white-space: normal; }
  .brainbox { margin: .4rem 0 0; }
  .stats { flex-direction: column; gap: .7rem; padding: .6rem 0 0; }
  .ribbon { position: static; flex-wrap: wrap; gap: .5rem 1rem; margin-top: .9rem; font-size: .7rem; }
  .panels { grid-template-columns: minmax(0, 1fr); }
  .page { padding-top: 1.2rem; }
}
@media (hover: none) { .neuron .ring { stroke-opacity: .55; } }
@media (prefers-reduced-motion: reduce) { .neuron *, .nav a, .item { transition: none; } }
"""


def photo_image(photo_id: str, alt: str = "") -> str:
    return (f'<img src="{PHOTO_PATH}?{urlencode({"id": photo_id})}" alt="{html.escape(alt)}" loading="lazy" '
            'decoding="async">')


def icon(name: str) -> str:
    return f'<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">{_ICONS[name]}</svg>'


def number(t: dict, value: int) -> str:
    return f"{value:,}".replace(",", "." if t["lang"] == "es" else ",")


_LOGO = ('<svg viewBox="0 0 120 60" aria-hidden="true"><defs><linearGradient id="loop" x1="0" x2="1" y1="0" y2="1">'
         '<stop offset="0" stop-color="#22d3ee"/><stop offset=".5" stop-color="#6366f1"/><stop offset="1" stop-color="#d8b4fe"/>'
         '</linearGradient><filter id="shine" x="-30%" y="-40%" width="160%" height="180%"><feGaussianBlur stdDeviation="2.4" '
         'result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>'
         '<path filter="url(#shine)" fill="none" stroke="url(#loop)" stroke-width="7" stroke-linecap="round" d="M60 30C50 14 '
         '38 7 26 7 13 7 5 17 5 30s8 23 21 23c12 0 24-7 34-23s22-23 34-23c13 0 21 10 21 23s-8 23-21 23c-12 0-24-7-34-23Z"/></svg>')


def shell(t: dict, active: str, account: str, csrf: str, body: str, hidden_fields, *, pending: int = 0, query: str = "",
          to_approve: int = 0, proposed: int = 0) -> str:
    """Everything around a signed-in page. It takes the signing-out form's fields from its caller. to_approve is how
    many tasks wait for the person, shown next to Tasks, and proposed how many guidelines, next to Guidelines."""
    e = html.escape
    items = (("home", "/", "nav_home", "home"), ("vault", "/memory", "nav_vault", "lock"),
             ("memories", "/memory?view=memories", "nav_memories", "layers"),
             ("projects", "/memory?view=projects", "nav_projects", "folder"),
             ("shares", "/memory?view=shares", "nav_shares", "people"),
             ("tasks", "/memory?view=tasks", "nav_tasks", "check"),
             ("guidelines", "/memory?view=guidelines", "nav_guidelines", "bulb"),
             ("reflections", "/memory?view=reflections", "nav_reflections", "pulse"),
             ("search", "/memory?view=search", "nav_search", "search"),
             ("apps", "/memory?view=apps", "nav_apps", "network"),
             ("settings", "/memory?view=settings", "nav_settings", "gear"))

    waiting = {"tasks": to_approve, "guidelines": proposed}

    def label(key: str, name: str) -> str:
        return f'{t[name]} <b class="badge">{waiting[key]}</b>' if waiting.get(key) else t[name]

    nav = "".join(f'<a href="{href}"{" class=on aria-current=page" if key == active else ""}>{icon(name)}{label(key, text)}</a>'
                  for key, href, text, name in items)
    alert = '<i class="dot"></i>' if pending else ""
    return (f'<div class="frame"><aside class="side"><a class="logo" href="/memory" aria-label="INFINITO">{_LOGO}</a>'
            f'<nav class="nav" aria-label="{t["menu"]}">{nav}</nav>'
            f'<form class="leave" method="post" action="/memory">{hidden_fields(action="signout", csrf=csrf)}'
            f'<button type="submit">{icon("out")}{t["sign_out"]}</button></form></aside>'
            f'<div class="work"><header class="top"><a class="word" href="/memory">INFINITO</a>'
            f'<form class="find" method="get" action="/memory" role="search">{icon("search")}'
            f'<input type="search" name="q" value="{e(query)}" placeholder="{t["find"]}" aria-label="{t["find"]}"></form>'
            f'<div class="tools"><a href="/memory?kind=pending" aria-label="{t["pending_alert"]}: {pending}">{icon("bell")}{alert}</a>'
            f'<a href="/#connect" aria-label="{t["help"]}">{icon("help")}</a>'
            f'<a class="me" href="/memory?view=settings" aria-label="{t["account"]}">{e(account[:1])}</a></div></header>'
            f'<main class="dash">{body}</main></div></div>')


def neurons(t: dict, counts: Dict[str, int]) -> str:
    """One bubble of the brain for each kind: a link, with its name and count under the pointer."""
    found = []
    for kind, (x, y, radius) in NEURON_SPOTS.items():
        label, many = t["kinds"][kind], counts.get(kind, 0)
        above = y - radius - 30 if y - radius > 52 else y + radius + 22
        found.append(
            f'<a class="neuron" style="--c:{KIND_COLOURS[kind]}" href="/memory?{urlencode({"kind": kind})}" '
            f'aria-label="{html.escape(label)}: {many}"><circle class="ring" cx="{x}" cy="{y}" r="{radius}"/>'
            f'<g class="tip"><text class="name" x="{x}" y="{above}">{html.escape(label)} · {many}</text>'
            f'<text class="about" x="{x}" y="{above + 18}">{html.escape(t["kind_about"][kind])}</text></g></a>')
    return (f'<nav aria-label="{t["map"]}"><svg class="hit" viewBox="0 0 725 455" xmlns="http://www.w3.org/2000/svg">'
            f'{"".join(found)}</svg></nav>')


def dashboard(t: dict, data: dict, csrf: str, hidden_fields) -> str:
    """The vault home. data holds counts by kind, recent memories, projects, and the vault's names (None if not invited)."""
    e = html.escape
    vault: Optional[List[Dict[str, str]]] = data["vault"]
    nodes = data["total"] + data["projects"] + len(vault or [])
    lead = (f'<section class="lead"><div class="title">{icon("lock")}<h1>{t["nav_vault"]}</h1></div>'
            f'<p class="sub">{t["vault_lead"]}</p><p class="about">{t["vault_about"]}</p>'
            f'<span class="pill">{icon("shield")}{t["vault_on"] if vault is not None else t["vault_off"]}</span></section>')
    brain = f'<div class="brainbox"><img src="{BRAIN_PATH}" alt="" width="725" height="455">{neurons(t, data["counts"])}</div>'
    stats = (f'<section class="stats"><div class="stat">{icon("network")}<div><b>{number(t, nodes)}</b><span>{t["stat_nodes"]}</span></div></div>'
             f'<div class="stat">{icon("layers")}<div><b>{number(t, data["total"])}</b><span>{t["stat_memories"]}</span></div></div>'
             f'<div class="stat cipher">{icon("shield")}<div><b>256-bit</b><span>{t["stat_cipher"]}</span></div></div></section>')
    ribbon = '<div class="ribbon" aria-hidden="true">' + "<i>·</i>".join(f"<span>{word}</span>" for word in t["ribbon"]) + "</div>"

    recent = "".join(
        f'<a class="item" style="--c:{KIND_COLOURS[m["kind"]]}" href="/memory?{urlencode({"photo" if m["photo"] else "edit": m["id"]})}">'
        f'<span class="tile">{photo_image(m["photo"]) if m["photo"] else icon(KIND_ICONS[m["kind"]])}</span>'
        f'<span class="what"><strong>{e(m["text"])}</strong>'
        f'<small>{e(m["label"])} · {e(m["ago"])}</small></span>'
        + (f'<span class="chip">{icon("network")}{e(m["chip"])}</span>' if m["chip"] else "") + "</a>"
        for m in data["recent"]) or f'<p class="empty-note">{t["no_memories"]}</p>'
    recent_panel = (f'<section class="panel"><header>{icon("layers")}<h2>{t["recent"]}</h2>'
                    f'<a href="/memory?view=memories">{t["see_all"]}{icon("arrow")}</a></header><div class="rows">{recent}</div></section>')

    key_icons = ("key", "wallet", "database")
    if vault is None:
        keys = f'<p class="empty-note">{t["no_keys"]}</p>'
    elif not vault:
        keys = f'<p class="empty-note">{t["no_keys_yet"]}</p>'
    else:
        keys = "".join(
            f'<div class="item key">{icon(key_icons[number_ % 3])}<span class="what"><strong>{e(entry["name"])}</strong>'
            f'<small>{e(entry.get("note") or t["key_kind"])}</small></span><span class="dots" aria-hidden="true">••••••••</span>'
            f'<form method="post" action="/memory">{hidden_fields(action="vault", csrf=csrf, entry=entry["name"], what="reveal")}'
            f'<button type="submit" aria-label="{t["open_key"]}: {e(entry["name"])}" title="{t["open_key"]}">{icon("eye")}</button></form></div>'
            for number_, entry in enumerate(vault[:3]))
    keys_panel = (f'<section class="panel teal"><header>{icon("shield")}<h2>{t["keys"]}</h2>'
                  + (f'<a href="/memory?view=vault">{t["see_all"]}{icon("arrow")}</a>' if vault else "")
                  + f'</header><div class="rows">{keys}</div></section>')

    colours = ("#a78bfa", "#22d3ee", "#60a5fa")
    events = "".join(
        f'<a class="event" style="--c:{colours[n % 3]}" href="/memory?{urlencode({"edit": m["id"]})}"><span class="what">'
        f'<strong>{t["activity_saved"]}</strong><small>“{e(m["text"])}”</small></span><span class="when">{e(m["ago"])}</span></a>'
        for n, m in enumerate(data["recent"]))
    activity = f'<div class="line">{events}</div>' if events else f'<p class="empty-note">{t["no_memories"]}</p>'
    activity_panel = (f'<section class="panel violet"><header>{icon("pulse")}<h2>{t["activity"]}</h2>'
                      f'<a href="/memory?view=memories">{t["see_history"]}{icon("arrow")}</a></header>{activity}</section>')
    return (f'<div class="hero">{lead}{brain}{stats}{ribbon}</div>'
            f'<div class="panels">{recent_panel}{keys_panel}{activity_panel}</div>')


def projects_body(t: dict, projects: List[dict]) -> str:
    e = html.escape
    if not projects:
        return f'<p>{t["no_projects"]}</p>'
    rows = []
    for project in projects:
        quiet = f' · <span class="quiet-flag">{t["quiet"].format(n=project["quiet"])}</span>' if project["stale"] else ""
        still_open = "".join(f"<li>{e(text)}</li>" for text in project["open"])
        rows.append(
            f'<article class="project"><h2><a href="/memory?{urlencode({"project": project["name"]})}">{e(project["name"])}</a></h2>'
            f'<p class="meta">{t["last_activity"]}: {e(project["last"])}{quiet} · '
            f'{_count_line(t, project["kinds"])}</p>'
            + (f'<p class="meta" style="margin-top:.5rem">{t["still_open"]}:</p><ul>{still_open}</ul>' if still_open else "")
            + "</article>")
    return "".join(rows)


def _count_line(t: dict, kinds: Dict[str, int]) -> str:
    return " · ".join(f'{many} {html.escape(t["kinds"][kind].lower())}' for kind, many in kinds.items())


def settings_body(t: dict, account: str) -> str:
    return (f'<h2 style="margin-top:0">{t["settings_account"]}</h2><p>{html.escape(account)}</p>'
            f'<p class="note">{t["settings_sealed"]}</p>')
