"""The INFINITO Teams page, <host>/teams: where people run their organization, with no AI in between.

Everything an app may not do is done here by a person, signed in with their password: creating an organization,
inviting, making projects, giving or taking access, confirming what an org admin proposed, and choosing what each of
their apps may do in the organization. The memories of a space can be read, added and taken back here too.

The page explains itself, because the people who run a team are not the ones who built it: every screen says where
you are and lets you sign out, says what you are in that organization and what you may do, walks a new owner through
the first steps, names who sends an invitation and to what, lets people be chosen from the organization rather than
typed, and tells the record of changes in words. It shows nobody more than the permissions in remote_teams.py allow.

It has its own session, like the memory page (remote_memory.py): the password unseals the person's key, which is then
kept for a quarter of an hour wrapped under a token the server stores only as a hash, in a cookie for /teams alone that
scripts cannot read and other sites cannot send. Every form carries a token of its own against other sites' forms. A
person's own memory never appears here, and nobody in an organization can reach it from here.

Only served when the server runs with INFINITO_ENTERPRISE_ENABLED=1 (remote_teams.py).
"""
import hmac
import html
import secrets
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlencode

from . import remote_teams
from .remote_accounts import _HEADERS, NAME, _page, _texts
from .remote_teams import ADMIN, GENERAL, MANAGER, MEMBER, NOT_FOUND, OWNER, PENDING, Refused

SESSION = "teams-session"
SESSION_SECONDS = 900
COOKIE = "infinito_teams"
PATH = "/teams"

STYLE = """
main { max-width: 50rem; }
.card { display: flex; flex-direction: column; }
.top { order: -1; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
  gap: .4rem 1rem; margin: 0 0 1rem; padding-bottom: .7rem; border-bottom: 1px solid var(--line); font-size: .88rem; }
.crumbs { color: var(--soft); overflow-wrap: anywhere; }
.who { display: inline-flex; align-items: center; gap: .6rem; color: var(--soft); }
.lead { margin: .3rem 0 0; }
h2 { font-size: 1.05rem; margin: 1.8rem 0 .4rem; color: var(--accent); }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .8rem; padding: .6rem 0;
  border-top: 1px solid var(--line); }
.row > .grow { flex: 1 1 14rem; min-width: 0; overflow-wrap: anywhere; }
.meta { color: var(--soft); font-size: .85rem; }
.inline { display: inline-flex; flex-wrap: wrap; gap: .4rem; align-items: center; margin: 0; }
.inline button, .small { width: auto; margin: 0; padding: .35rem .8rem; font-size: .88rem; }
.quiet { background: transparent; color: var(--accent); border: 1px solid var(--line); }
.bad { background: transparent; color: var(--bad); border: 1px solid var(--line); }
select, textarea { font: inherit; color: inherit; background: var(--page); border: 1px solid var(--line);
  border-radius: 9px; padding: .45rem .6rem; }
textarea { width: 100%; min-height: 4.5rem; }
.ok { color: var(--accent); font-weight: 600; }
.tag { font-size: .78rem; padding: .05rem .5rem; border: 1px solid var(--line); border-radius: 99px; color: var(--soft); }
.code.link { font-size: .9rem; letter-spacing: 0; text-align: left; user-select: all; }
.steps { list-style: none; margin: 1rem 0 0; padding: .7rem 1rem; border: 1px solid var(--line); border-radius: 12px;
  background: rgba(103, 232, 249, .05); }
.steps li { display: flex; gap: .6rem; padding: .25rem 0; color: var(--soft); }
.steps li.done { color: var(--ink); text-decoration: line-through; text-decoration-color: var(--line); }
.steps .mark { flex: none; width: 1.4rem; text-align: center; color: var(--accent); font-weight: 700; }
details.panel { margin-top: .9rem; padding: .6rem .9rem; border: 1px solid var(--line); border-radius: 12px; }
details.panel > summary { color: var(--accent); font-weight: 600; font-size: .95rem; }
details.panel[open] > summary { margin-bottom: .3rem; }
.legend { margin: .4rem 0 0; }
.legend dt { font-weight: 600; margin-top: .5rem; }
.legend dd { margin: .1rem 0 0; color: var(--soft); font-size: .88rem; }
.check { display: flex; align-items: center; gap: .5rem; margin-top: .9rem; font-weight: 400; }
.check input { width: auto; margin: 0; }
.notice { margin: .9rem 0 0; padding: .6rem .8rem; border: 1px solid var(--line); border-radius: 10px;
  background: rgba(167, 139, 250, .08); }
"""

TEXT = {
    "es": {
        "title": "INFINITO Teams", "brand": "INFINITO · Teams",
        "sign_in_note": "Entra con tu cuenta de INFINITO. Tu memoria personal no aparece aquí, y nadie de tu "
                        "organización puede verla.",
        "invited_by": "{by} te invita a {org}.",
        "invite_sign_in": "Entra con tu cuenta de INFINITO para aceptar. Tu memoria personal no aparece aquí, y "
                          "nadie de la organización puede verla.",
        "no_account": '¿Aún no tienes cuenta de INFINITO? <a href="/memory">Créala gratis</a> y vuelve a abrir este '
                      "mismo enlace.",
        "name": "Tu nombre de usuario", "password": "Contraseña", "enter": "Entrar",
        "bad_password": "Ese nombre o esa contraseña no son correctos.",
        "wait": "Demasiados intentos. Espera {n} segundos.", "session_over": "La sesión ha caducado. Vuelve a entrar.",
        "bad_form": "Este formulario ya no vale. Vuelve a la página.", "back": "Volver", "signout": "Salir",
        "where": "Dónde estás",
        "orgs": "Tus organizaciones",
        "about": "INFINITO Teams es la memoria compartida de un equipo. Cada organización tiene un espacio General, "
                 "que ve todo el mundo, y proyectos, que solo ven las personas que se añaden. Las IA de cada persona "
                 "(ChatGPT, Claude…) pueden consultar y guardar ahí lo que el equipo sabe, si esa persona se lo "
                 "permite.",
        "no_orgs": "Aún no estás en ninguna organización. Si alguien te ha invitado, abre el enlace que te mandó.",
        "create_org": "Crear una organización", "org_name": "Nombre de la organización (tu empresa o tu equipo)",
        "create": "Crear",
        "roles": {OWNER: "propietario", ADMIN: "administrador", MEMBER: "miembro", MANAGER: "responsable"},
        "you_are": {OWNER: "Eres el propietario: ves todos los espacios y decides quién entra en cada uno.",
                    ADMIN: "Eres administrador: invitas, creas proyectos y das acceso a ellos, pero no lees los "
                           "proyectos en los que no estás.",
                    MEMBER: "Eres miembro: aquí ves los espacios en los que estás. Para entrar en otro proyecto, "
                            "pídeselo a {owner}, que lleva la organización."},
        "first_steps": "Primeros pasos",
        "steps": ["Crea un proyecto para cada tema, cliente o equipo.",
                  "Invita a tu gente con un enlace (más abajo, en Personas).",
                  "Entra en cada proyecto y añade a quien deba estar.",
                  "Elige qué pueden hacer tus IA aquí (al final de esta página)."],
        "spaces": "Espacios", "kinds": {GENERAL: "general", "project": "proyecto"},
        "kind_notes": {GENERAL: "Lo ve toda la organización.", "project": "Solo lo ven las personas añadidas."},
        "access": {"write": "Tú: lees y añades", "read": "Tú: solo lees", PENDING: "Tú: esperando acceso",
                   "manage_only": "Tú: das acceso, sin leer su contenido", "none": "Tú: solo ves su nombre"},
        "people_one": "1 persona", "people_many": "{n} personas",
        "new_space": "Nuevo proyecto",
        "new_space_note": "Un proyecto es un espacio aparte, cifrado con su propia llave: solo lo ven las personas "
                          "que añadas y el propietario.",
        "space_name": "Nombre del proyecto", "manager": "Responsable (opcional)",
        "manager_note": "El responsable lee, añade y decide quién entra en el proyecto.", "nobody": "Nadie por ahora",
        "manager_free": "Responsable (nombre de usuario, opcional)",
        "people": "Personas",
        "invite": "Invitar a alguien",
        "invite_note": "Se crea un enlace que sirve una sola vez y caduca en 7 días. Quien lo abra entra como "
                       "miembro y ve el espacio General; a los proyectos lo añades después.",
        "invite_go": "Crear enlace de invitación",
        "invite_link": "Enlace listo. Cópialo y mándaselo a esa persona por correo o mensaje. Por seguridad no se "
                       "vuelve a mostrar.",
        "invite_pending": "Como no tienes la llave del espacio general, quien entre con este enlace esperará a que "
                          "el propietario le dé acceso.",
        "roles_help": "¿Qué puede hacer cada papel?",
        "roles_legend": [(OWNER, "Ve y gestiona todo, también los proyectos. Hay uno por organización."),
                         (ADMIN, "Invita, crea proyectos y da acceso, pero no lee los proyectos en los que no está."),
                         (MANAGER, "En su proyecto: lee, añade y decide quién entra."),
                         (MEMBER, "Lee, y si se le permite añade, en los espacios donde está.")],
        "make_admin": "Hacer administrador", "make_member": "Dejar de ser administrador",
        "remove": "Quitar", "leave_org": "Salir de esta organización",
        "apps": "Tus IA en esta organización",
        "apps_note": "Cada IA conectada a tu memoria (ChatGPT, Claude, Grok…) empieza sin acceso a la organización. "
                     "Elige qué puede hacer en los espacios donde tú estás.",
        "no_apps": 'Aún no has conectado ninguna IA a tu memoria. <a href="/">Conéctala desde la portada</a> y '
                   "vuelve aquí para darle acceso.",
        "levels": {"none": "Nada", "read": "Solo leer", "add": "Leer y añadir"}, "save": "Guardar",
        "audit": "Registro de cambios",
        "audit_lines": {"create_org": "creó la organización", "invite": "creó un enlace de invitación",
                        "join": "entró con la invitación de {target}", "set_role": "hizo a {target} {role}",
                        "remove_member": "quitó a {target} de la organización",
                        "left_org": "salió de la organización",
                        "app_level": "cambió lo que puede hacer una de sus IA: {level}",
                        "create_space": "creó el proyecto {space}",
                        "set_seat": "cambió el papel de {who} en {space} a {role}",
                        "add_member": "añadió a {who} a {space} como {role}",
                        "confirm": "dio acceso a {who} en {space}", "remove_from_space": "quitó a {who} de {space}",
                        "left_space": "salió de {space}", "new_key": "se renovó la llave de {space}",
                        "remember": "guardó un recuerdo en {space}", "forget": "retiró un recuerdo de {space}"},
        "audit_refused": "intentó algo que no le estaba permitido ({action})",
        "space_of": {GENERAL: "Espacio general de {org}: lo ve toda la organización.",
                     "project": "Proyecto de {org}: solo lo ven las personas añadidas."},
        "memories": "Recuerdos",
        "memories_note": "Lo que el equipo sabe de este espacio. Las IA de cada persona lo consultan y lo amplían "
                         "si esa persona se lo permite.",
        "empty": "Este espacio aún no tiene recuerdos. Añade el primero o pídeselo a tu IA.",
        "add_memory": "Añadir un recuerdo", "statement": "Qué hay que recordar", "kind": "Tipo",
        "add": "Guardar recuerdo", "forget": "Retirar",
        "kinds_memory": {"fact": "hecho", "decision": "decisión", "agreement": "acuerdo", "proposal": "propuesta",
                         "pending": "pendiente", "assumption": "supuesto", "to_verify": "por verificar"},
        "space_members": "Quién está en este espacio", "owner_row": "lee y gestiona todos los espacios",
        "add_member": "Añadir a alguien a este espacio", "add_person": "Añadir", "account": "Persona",
        "account_free": "Nombre de usuario",
        "no_candidates": "Ya están aquí todos los de la organización. Para sumar a alguien nuevo, invítalo desde la "
                         "organización.",
        "role": "Papel en este espacio", "can_write": "puede añadir recuerdos", "read_only": "solo lee",
        "confirm": "Dar acceso", "pending_note": "Esperando: el propietario o un responsable tiene que darle acceso.",
        "you_pending": "Aún no tienes acceso a este espacio: el propietario o un responsable tiene que dártelo.",
        "leave_space": "Salir de este espacio",
        "invited_title": "Te han invitado a {org}", "invited_plain": "Te han invitado a una organización",
        "invited_note": "Al aceptar entrarás como miembro y verás su espacio General. Tu memoria personal sigue "
                        "siendo solo tuya.",
        "accept_as": "Aceptarás con tu cuenta {account}. Si quieres usar otra, pulsa Salir y vuelve a abrir el "
                     "enlace.",
        "accept": "Aceptar", "invite_expired": "Este enlace de invitación ya no sirve. Pide uno nuevo.",
        "joined": "Ya estás en {org}.", "done": "Hecho.",
        "refusals": {NOT_FOUND: "No existe, o no es tuyo.",
                     "You may read this space but not add to it.": "Puedes leer este espacio, pero no añadir.",
                     "You are already in this organization.": "Ya estás en esta organización.",
                     "This account may not create organizations on this server.":
                         "Tu cuenta no puede crear organizaciones en este servidor.",
                     "Everyone in the organization is in the general space already.":
                         "Todos los de la organización están ya en el espacio general.",
                     remote_teams.SELF: "Nadie puede añadirse a sí mismo a un espacio: tiene que hacerlo otra "
                                        "persona que lo lleve.",
                     remote_teams.OWNER_READS: "El propietario de la organización ya lee y gestiona todos los "
                                               "espacios: no hace falta añadirlo.",
                     remote_teams.NOBODY: "No hay nadie con ese nombre de usuario en esta organización: invítalo "
                                          "primero."},
    },
    "en": {
        "title": "INFINITO Teams", "brand": "INFINITO · Teams",
        "sign_in_note": "Sign in with your INFINITO account. Your own memory does not show here, and nobody in your "
                        "organization can see it.",
        "invited_by": "{by} invites you to {org}.",
        "invite_sign_in": "Sign in with your INFINITO account to accept. Your own memory does not show here, and "
                          "nobody in the organization can see it.",
        "no_account": 'No INFINITO account yet? <a href="/memory">Make one for free</a> and open this same link '
                      "again.",
        "name": "Your username", "password": "Password", "enter": "Sign in",
        "bad_password": "That name or password is not right.",
        "wait": "Too many tries. Wait {n} seconds.", "session_over": "Your session ended. Sign in again.",
        "bad_form": "This form no longer works. Go back to the page.", "back": "Back", "signout": "Sign out",
        "where": "Where you are",
        "orgs": "Your organizations",
        "about": "INFINITO Teams is a team's shared memory. Each organization has a General space everyone sees, and "
                 "projects only the people added to them see. Each person's AIs (ChatGPT, Claude…) can look up and "
                 "keep there what the team knows, if that person lets them.",
        "no_orgs": "You are in no organization yet. If someone invited you, open the link they sent.",
        "create_org": "Create an organization", "org_name": "Organization name (your company or your team)",
        "create": "Create",
        "roles": {OWNER: "owner", ADMIN: "admin", MEMBER: "member", MANAGER: "manager"},
        "you_are": {OWNER: "You are the owner: you see every space and decide who goes into each.",
                    ADMIN: "You are an admin: you invite, make projects and give access to them, but you do not "
                           "read the projects you are not in.",
                    MEMBER: "You are a member: here you see the spaces you are in. To join another project, ask "
                            "{owner}, who runs the organization."},
        "first_steps": "First steps",
        "steps": ["Make a project for each topic, client or team.",
                  "Invite your people with a link (below, under People).",
                  "Open each project and add whoever should be in it.",
                  "Choose what your AIs may do here (at the end of this page)."],
        "spaces": "Spaces", "kinds": {GENERAL: "general", "project": "project"},
        "kind_notes": {GENERAL: "The whole organization sees it.", "project": "Only the people added see it."},
        "access": {"write": "You: read and add", "read": "You: read only", PENDING: "You: waiting for access",
                   "manage_only": "You: give access, without reading it", "none": "You: see its name only"},
        "people_one": "1 person", "people_many": "{n} people",
        "new_space": "New project",
        "new_space_note": "A project is a space of its own, sealed with its own key: only the people you add and the "
                          "owner see it.",
        "space_name": "Project name", "manager": "Manager (optional)",
        "manager_note": "The manager reads, adds and decides who goes into the project.", "nobody": "Nobody yet",
        "manager_free": "Manager (username, optional)",
        "people": "People",
        "invite": "Invite someone",
        "invite_note": "This makes a link that works once and ends in 7 days. Whoever opens it joins as a member and "
                       "sees the General space; you add them to projects afterwards.",
        "invite_go": "Make an invitation link",
        "invite_link": "Link ready. Copy it and send it to that person by email or message. For safety it is not "
                       "shown again.",
        "invite_pending": "As you do not hold the general space's key, whoever joins with this link waits for the "
                          "owner to give them access.",
        "roles_help": "What can each role do?",
        "roles_legend": [(OWNER, "Sees and runs everything, projects included. One per organization."),
                         (ADMIN, "Invites, makes projects and gives access, but does not read the projects they are "
                                 "not in."),
                         (MANAGER, "In their project: reads, adds and decides who goes in."),
                         (MEMBER, "Reads, and if allowed adds, in the spaces they are in.")],
        "make_admin": "Make admin", "make_member": "No longer admin", "remove": "Remove",
        "leave_org": "Leave this organization",
        "apps": "Your AIs in this organization",
        "apps_note": "Each AI connected to your memory (ChatGPT, Claude, Grok…) starts with no access to the "
                     "organization. Choose what it may do in the spaces you are in.",
        "no_apps": 'You have not connected any AI to your memory yet. <a href="/">Connect one from the front page</a> '
                   "and come back here to give it access.",
        "levels": {"none": "Nothing", "read": "Read only", "add": "Read and add"}, "save": "Save",
        "audit": "Record of changes",
        "audit_lines": {"create_org": "made the organization", "invite": "made an invitation link",
                        "join": "joined with {target}'s invitation", "set_role": "made {target} {role}",
                        "remove_member": "removed {target} from the organization",
                        "left_org": "left the organization",
                        "app_level": "changed what one of their AIs may do: {level}",
                        "create_space": "made the project {space}",
                        "set_seat": "changed {who}'s role in {space} to {role}",
                        "add_member": "added {who} to {space} as {role}", "confirm": "gave {who} access to {space}",
                        "remove_from_space": "removed {who} from {space}", "left_space": "left {space}",
                        "new_key": "{space} got a new key", "remember": "kept a memory in {space}",
                        "forget": "took back a memory from {space}"},
        "audit_refused": "tried something they were not allowed to ({action})",
        "space_of": {GENERAL: "General space of {org}: the whole organization sees it.",
                     "project": "Project of {org}: only the people added see it."},
        "memories": "Memories",
        "memories_note": "What the team knows about this space. Each person's AIs look it up and add to it if that "
                         "person lets them.",
        "empty": "This space has no memories yet. Add the first one, or ask your AI to.",
        "add_memory": "Add a memory", "statement": "What to remember", "kind": "Kind", "add": "Save memory",
        "forget": "Take back",
        "kinds_memory": {"fact": "fact", "decision": "decision", "agreement": "agreement", "proposal": "proposal",
                         "pending": "pending", "assumption": "assumption", "to_verify": "to verify"},
        "space_members": "Who is in this space", "owner_row": "reads and runs every space",
        "add_member": "Add someone to this space", "add_person": "Add", "account": "Person", "account_free": "Username",
        "no_candidates": "Everyone in the organization is here already. To bring someone new, invite them from the "
                         "organization.",
        "role": "Role in this space", "can_write": "may add memories", "read_only": "reads only",
        "confirm": "Give access", "pending_note": "Waiting: the owner or a manager has to give them access.",
        "you_pending": "You have no access to this space yet: the owner or a manager has to give it to you.",
        "leave_space": "Leave this space",
        "invited_title": "You are invited to {org}", "invited_plain": "You are invited to an organization",
        "invited_note": "Accepting makes you a member and shows you its General space. Your own memory stays yours "
                        "alone.",
        "accept_as": "You will accept with your account {account}. If you want another one, sign out and open the "
                     "link again.",
        "accept": "Accept", "invite_expired": "This invitation link no longer works. Ask for a new one.",
        "joined": "You are in {org} now.", "done": "Done.",
        "refusals": {NOT_FOUND: "It does not exist, or it is not yours."},
    },
}

_LEVELS = ("none", "read", "add")
_KINDS = ("fact", "decision", "agreement", "proposal", "pending", "to_verify")


def _hidden(**fields) -> str:
    return "".join(f'<input type="hidden" name="{name}" value="{html.escape(str(value))}">'
                   for name, value in fields.items() if value is not None)


def _said(t: dict, refused: Refused) -> str:
    return t["refusals"].get(str(refused), str(refused))


def _button(t: dict, label: str, css: str = "small", **fields) -> str:
    return (f'<form method="post" action="{PATH}" class="inline">{_hidden(**fields)}'
            f'<button class="{css}" type="submit">{html.escape(label)}</button></form>')


def _link(label: str, **where) -> str:
    return f'<a href="{PATH}?{urlencode(where)}">{html.escape(label)}</a>' if where else \
        f'<a href="{PATH}">{html.escape(label)}</a>'


def _top(t: dict, account: str, csrf: str, crumbs: List[str]) -> str:
    """Where the person is, and who they are with a way out: on every page once signed in. crumbs are ready HTML."""
    return (f'<div class="top"><nav class="crumbs" aria-label="{t["where"]}">{" › ".join(crumbs)}</nav>'
            f'<div class="who">{html.escape(account)}'
            f'{_button(t, t["signout"], "small quiet", action="signout", csrf=csrf)}</div></div>')


def _when(at: str, lang: str) -> str:
    """A moment of the record as people read it: day and time, in Spain for the Spanish page, else in UTC."""
    try:
        moment = datetime.fromisoformat(at)
    except (TypeError, ValueError):
        return str(at)
    if lang == "es":
        try:
            from zoneinfo import ZoneInfo

            return moment.astimezone(ZoneInfo("Europe/Madrid")).strftime("%d/%m %H:%M")
        except (ImportError, KeyError, ValueError):  # no time zone data on this machine
            pass
    return moment.astimezone(timezone.utc).strftime("%d/%m %H:%M UTC")


def _audit_line(t: dict, item: dict, spaces: Dict[str, str]) -> str:
    """One change in words: who did what to whom, with names instead of ids. Never the words of a memory."""
    action, _, detail = str(item.get("action", "")).partition(":")
    target, actor = str(item.get("target", "")), str(item.get("actor", ""))
    space_id, _, who = target.partition("/")
    if item.get("outcome") == "refused":
        text = t["audit_refused"].format(action=action)
    else:
        key = ("left_org" if action == "remove_member" and target == actor else
               "left_space" if action == "remove_from_space" and who == actor else action)
        template = t["audit_lines"].get(key)
        text = template.format(target=target, who=who, space=spaces.get(space_id, space_id),
                               role=t["roles"].get(detail, detail), level=t["levels"].get(detail, detail)) \
            if template else f"{item.get('action', '')} {target}".strip()
    return f'{_when(item.get("at", ""), t["lang"])} · {actor} {text}'


def sign_in_body(t: dict, *, invite: str = "", name: str = "", error: str = "",
                 invitation: Optional[dict] = None) -> str:
    e = html.escape
    wrong = f'<p class="error" role="alert">{e(error)}</p>' if error else ""
    if invitation:
        intro = (f'<p class="lead">{e(t["invited_by"].format(by=invitation["by"], org=invitation["org"]))}</p>'
                 f'<p class="note">{t["invite_sign_in"]}</p>')
    else:
        intro = f'<p class="note">{t["sign_in_note"]}</p>'
    return (f'{intro}<form method="post" action="{PATH}">'
            f'{_hidden(action="signin", invite=invite or None)}'
            f'<label for="name">{t["name"]}</label><input id="name" name="name" type="text" value="{e(name)}" '
            'autocomplete="username" autocapitalize="none" spellcheck="false" required>'
            f'<label for="password">{t["password"]}</label><input id="password" name="password" type="password" '
            f'autocomplete="current-password" required>{wrong}<button type="submit">{t["enter"]}</button></form>'
            + (f'<p class="note">{t["no_account"]}</p>' if invitation else ""))


def home_body(t: dict, orgs: List[dict], csrf: str, *, can_create: bool, account: str = "", message: str = "",
              error: str = "") -> str:
    e = html.escape
    rows = "".join(f'<div class="row"><div class="grow">{_link(o["name"], org=o["id"])}</div>'
                   f'<span class="tag">{t["roles"][o["role"]]}</span></div>' for o in orgs)
    create = (f'<details class="panel"{"" if orgs else " open"}><summary>{t["create_org"]}</summary>'
              f'<form method="post" action="{PATH}">{_hidden(action="create_org", csrf=csrf)}'
              f'<label for="org_name">{t["org_name"]}</label><input id="org_name" name="name" type="text" '
              f'maxlength="60" required><button type="submit">{t["create"]}</button></form></details>'
              ) if can_create else ""
    return (_top(t, account, csrf, [e(t["orgs"])]) + _notice(message, error)
            + (f'<p class="note">{t["about"]}</p>' if not orgs else "")
            + f'<h2>{t["orgs"]}</h2>' + (rows or f'<p class="note">{t["no_orgs"]}</p>') + create)


def _notice(message: str = "", error: str = "") -> str:
    e = html.escape
    return ((f'<p class="ok" role="status">{e(message)}</p>' if message else "")
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else ""))


def _access(t: dict, space: dict) -> str:
    return t["access"]["write" if space["write"] else "read" if space["read"] else
                       PENDING if space["state"] == PENDING else "manage_only" if space["add"] else "none"]


def _people(t: dict, org: dict, space: dict) -> str:
    """How many are in a space, the owner included, when this person may see who is in it."""
    if not (space["read"] or space["add"]):
        return ""
    n = len(space["members"]) + (0 if org["owner"] in space["members"] else 1)
    return " · " + (t["people_one"] if n == 1 else t["people_many"].format(n=n))


def _first_steps(t: dict, org: dict, levels: Dict[str, str]) -> str:
    """For whoever runs the organization, until each step is done: what to do, and in what order."""
    projects = [s for s in org["spaces"] if s["kind"] != GENERAL]
    done = [bool(projects), len(org["members"]) > 1, any(s["members"] for s in projects),
            any(level != "none" for level in levels.values())]
    if all(done):
        return ""
    items = "".join(f'<li class="{"done" if ok else ""}"><span class="mark">{"✓" if ok else n}</span>'
                    f'<span>{step}</span></li>' for n, (step, ok) in enumerate(zip(t["steps"], done), 1))
    return f'<h2>{t["first_steps"]}</h2><ol class="steps">{items}</ol>'


def _member_choice(t: dict, field: str, label: str, candidates: Optional[List[str]], *, optional: bool) -> str:
    """Someone of the organization, picked from a list when this person may see its members; typed otherwise."""
    e = html.escape
    if candidates is None:
        return (f'<label for="{field}">{label}</label><input id="{field}" name="{field}" type="text" '
                f'autocapitalize="none" spellcheck="false"{"" if optional else " required"}>')
    options = (f'<option value="">{t["nobody"]}</option>' if optional else "") + "".join(
        f'<option value="{e(name)}">{e(name)}</option>' for name in candidates)
    return f'<label for="{field}">{label}</label><select id="{field}" name="{field}">{options}</select>'


def org_body(t: dict, org: dict, account: str, csrf: str, apps: List[dict], levels: Dict[str, str], *,
             link: str = "", link_pending: bool = False, message: str = "", error: str = "") -> str:
    e = html.escape
    org_id, role = org["id"], org["role"]
    runs = role in (OWNER, ADMIN)
    out = [_top(t, account, csrf, [_link(t["orgs"]), e(org["name"])]), _notice(message, error),
           f'<p class="lead"><span class="tag">{t["roles"][role]}</span> '
           f'{e(t["you_are"][role].format(owner=org["owner"]))}</p>']
    if link:  # right at the top, where the person who just asked for it looks
        out.append(f'<div class="notice"><p class="ok" role="status">{t["invite_link"]}</p>'
                   f'<div class="code link">{e(link)}</div>'
                   + (f'<p class="note">{t["invite_pending"]}</p>' if link_pending else "") + '</div>')
    if runs:
        out.append(_first_steps(t, org, levels))

    out.append(f'<h2>{t["spaces"]}</h2>')
    for space in org["spaces"]:
        name = _link(space["name"], org=org_id, space=space["id"]) if space["read"] or space["add"] \
            else e(space["name"])
        out.append(f'<div class="row"><div class="grow">{name} <span class="tag">{t["kinds"][space["kind"]]}</span>'
                   f'<div class="meta">{t["kind_notes"][space["kind"]]}{_people(t, org, space)}</div></div>'
                   f'<span class="meta">{_access(t, space)}</span></div>')
    if runs:
        candidates = [name for name in org["members"] if name != org["owner"]]
        projects = [s for s in org["spaces"] if s["kind"] != GENERAL]
        out.append(f'<details class="panel"{"" if projects else " open"}><summary>{t["new_space"]}</summary>'
                   f'<p class="note">{t["new_space_note"]}</p><form method="post" action="{PATH}">'
                   f'{_hidden(action="create_space", csrf=csrf, org=org_id)}<label for="space_name">{t["space_name"]}'
                   f'</label><input id="space_name" name="name" type="text" maxlength="60" required>'
                   + _member_choice(t, "manager", t["manager"], candidates, optional=True)
                   + f'<p class="hint">{t["manager_note"]}</p><button type="submit">{t["create"]}</button></form>'
                   '</details>')

    if org["members"]:
        out.append(f'<h2>{t["people"]}</h2>')
        for name, their_role in org["members"].items():
            actions = ""
            if role == OWNER and their_role != OWNER:
                new = MEMBER if their_role == ADMIN else ADMIN
                actions += _button(t, t["make_member" if their_role == ADMIN else "make_admin"], "small quiet",
                                   action="set_role", csrf=csrf, org=org_id, target=name, role=new)
            if their_role != OWNER and name != account and (role == OWNER or their_role != ADMIN):
                actions += _button(t, t["remove"], "small bad", action="remove_member", csrf=csrf, org=org_id,
                                   target=name)
            out.append(f'<div class="row"><div class="grow">{e(name)}</div><span class="tag">'
                       f'{t["roles"][their_role]}</span>{actions}</div>')
    if runs:
        out.append(f'<details class="panel"><summary>{t["invite"]}</summary><p class="note">{t["invite_note"]}</p>'
                   + _button(t, t["invite_go"], "small quiet", action="invite", csrf=csrf, org=org_id) + '</details>')
        legend = "".join(f'<dt>{t["roles"][r]}</dt><dd>{text}</dd>' for r, text in t["roles_legend"])
        out.append(f'<details class="panel"><summary>{t["roles_help"]}</summary><dl class="legend">{legend}</dl>'
                   '</details>')

    out.append(f'<h2>{t["apps"]}</h2><p class="note">{t["apps_note"]}</p>')
    for app in apps:
        level = levels.get(app["client_id"], "none")
        options = "".join(f'<option value="{name}"{" selected" if name == level else ""}>{t["levels"][name]}</option>'
                          for name in _LEVELS)
        out.append(f'<div class="row"><div class="grow">{e(app["name"])} <span class="meta">{e(app["host"])}</span>'
                   f'</div><form method="post" action="{PATH}" class="inline">'
                   f'{_hidden(action="app_level", csrf=csrf, org=org_id, client=app["client_id"])}'
                   f'<select name="level" aria-label="{e(app["name"])}">{options}</select>'
                   f'<button class="small quiet" type="submit">{t["save"]}</button></form></div>')
    if not apps:
        out.append(f'<p class="note">{t["no_apps"]}</p>')

    if org["audit"]:
        names = {s["id"]: s["name"] for s in org["spaces"]}
        out.append(f'<details class="panel"><summary>{t["audit"]}</summary>' + "".join(
            f'<div class="row meta"><div class="grow">{e(_audit_line(t, item, names))}</div></div>'
            for item in reversed(org["audit"])) + '</details>')
    if role != OWNER:
        leave = _button(t, t["leave_org"], "small bad", action="remove_member", csrf=csrf, org=org_id, target=account)
        out.append(f'<div class="links">{leave}</div>')
    return "".join(out)


def space_body(t: dict, org: dict, space: dict, memories: Optional[List[dict]], account: str, csrf: str, *,
               message: str = "", error: str = "") -> str:
    e = html.escape
    ids = {"org": org["id"], "space": space["id"]}
    out = [_top(t, account, csrf, [_link(t["orgs"]), _link(org["name"], org=org["id"]), e(space["name"])]),
           _notice(message, error),
           f'<p class="lead">{e(t["space_of"][space["kind"]].format(org=org["name"]))}</p>',
           f'<p class="meta">{_access(t, space)}</p>']
    if space["state"] == PENDING and not space["read"]:
        out.append(f'<p class="notice">{t["you_pending"]}</p>')
    if memories is not None:
        out.append(f'<h2>{t["memories"]}</h2><p class="note">{t["memories_note"]}</p>')
        for memory in memories:
            kind = t["kinds_memory"].get(memory["kind"], memory["kind"])
            mine = memory["by"] == account or memory["by"].startswith(f"{account} · ")
            take_back = (_button(t, t["forget"], "small bad", action="forget", csrf=csrf, memory=memory["id"], **ids)
                         if mine or space["manage"] else "")
            out.append(f'<div class="row"><div class="grow">{e(memory["text"])}<div class="meta">{e(memory["day"])}'
                       f'{" · " + e(kind) if kind else ""} · {e(memory["by"])}</div></div>{take_back}</div>')
        if not memories:
            out.append(f'<p class="note">{t["empty"]}</p>')
    if space["write"]:
        options = "".join(f'<option value="{k}">{t["kinds_memory"][k]}</option>' for k in _KINDS)
        out.append(f'<details class="panel" open><summary>{t["add_memory"]}</summary><form method="post" '
                   f'action="{PATH}">{_hidden(action="remember", csrf=csrf, **ids)}<label for="statement">'
                   f'{t["statement"]}</label><textarea id="statement" name="statement" maxlength="2000" required>'
                   f'</textarea><label for="kind">{t["kind"]}</label><select id="kind" name="kind">{options}</select>'
                   f'<button type="submit">{t["add"]}</button></form></details>')
    if space["read"] or space["add"]:
        out.append(f'<h2>{t["space_members"]}</h2>')
        if org["owner"] not in space["members"]:
            out.append(f'<div class="row"><div class="grow">{e(org["owner"])} <span class="tag">{t["roles"][OWNER]}'
                       f'</span><div class="meta">{t["owner_row"]}</div></div></div>')
        for name, seat in space["members"].items():
            actions = ""
            if seat["state"] == PENDING and space["manage"] and name != account:
                actions += _button(t, t["confirm"], "small", action="confirm", csrf=csrf, target=name, **ids)
            if space["kind"] != GENERAL and space["add"] and name != account:
                actions += _button(t, t["remove"], "small bad", action="remove_from_space", csrf=csrf, target=name,
                                   **ids)
            state = t["pending_note"] if seat["state"] == PENDING else (
                t["can_write"] if seat["write"] else t["read_only"])
            out.append(f'<div class="row"><div class="grow">{e(name)} <span class="tag">{t["roles"][seat["role"]]}'
                       f'</span><div class="meta">{state}</div></div>{actions}</div>')
    if space["add"] and space["kind"] != GENERAL:
        # From the organization's people when this person may see them (owner and admins); typed otherwise.
        candidates = ([name for name in org["members"] if name != org["owner"] and name not in space["members"]]
                      if org["members"] else None)
        if candidates == []:
            out.append(f'<details class="panel"><summary>{t["add_member"]}</summary>'
                       f'<p class="note">{t["no_candidates"]}</p></details>')
        else:
            roles = "".join(f'<option value="{r}">{t["roles"][r]}</option>' for r in (MEMBER, MANAGER))
            out.append(f'<details class="panel"><summary>{t["add_member"]}</summary><form method="post" '
                       f'action="{PATH}">{_hidden(action="add_member", csrf=csrf, **ids)}'
                       + _member_choice(t, "target", t["account"] if candidates is not None else t["account_free"],
                                        candidates, optional=False)
                       + f'<label for="role">{t["role"]}</label><select id="role" name="role">{roles}</select>'
                       f'<label class="check"><input type="checkbox" name="write" value="1" checked> '
                       f'{t["can_write"]}</label><button type="submit">{t["add_person"]}</button></form></details>')
    if space["kind"] != GENERAL and account in space["members"]:
        leave = _button(t, t["leave_space"], "small bad", action="remove_from_space", csrf=csrf, target=account, **ids)
        out.append(f'<div class="links">{leave}</div>')
    return "".join(out)


def routes(accounts, grants, teams, limiter, secure: bool, issuer: str = "") -> list:
    """GET and POST /teams. teams is the server's one remote_teams.Teams, shared with the company_* tools."""
    import anyio
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    policy = f"{_HEADERS['Content-Security-Policy']}; form-action 'self'; base-uri 'none'"
    # An invitation works once, even for two people who open it at the same moment.
    accepting = threading.Lock()

    def take_invitation(token: str, account: str, key: bytes) -> Optional[str]:
        """The organization's name once account has joined with token, or None if the token no longer works."""
        with accepting:
            offer = grants.find(remote_teams.INVITE, token)
            if offer is None:
                return None
            name = teams.accept(offer, account, key)
            grants.drop(token)
            return name

    def texts(request) -> dict:
        base = _texts(request)
        return {**base, **TEXT[base["lang"]]}

    def view(t: dict, title: str, body: str, status: int = 200):
        response = _page(t, title, body, status, style=STYLE)
        response.headers["Content-Security-Policy"] = policy
        return response

    def signed_in(request) -> Optional[dict]:
        token = request.cookies.get(COOKIE, "")
        found = grants.find(SESSION, token)
        if found is None or found["key"] is None or not accounts.exists(found["account"]):
            return None
        return {**found, "token": token}

    async def run(work, *args):
        return await anyio.to_thread.run_sync(work, *args)

    async def invitation_of(token: str) -> Optional[dict]:
        """Who invites, and to what, while the invitation still works."""
        offer = grants.find(remote_teams.INVITE, token) if token else None
        return await run(teams.invitation, offer) if offer else None

    def back_to(**where) -> RedirectResponse:
        wanted = {name: value for name, value in where.items() if value}
        return RedirectResponse(f"{PATH}?{urlencode(wanted)}" if wanted else PATH, status_code=303, headers=_HEADERS)

    async def sign_in_page(t: dict, invite: str = "", status: int = 200, **fields):
        invitation = await invitation_of(invite)
        title = t["invited_title"].format(org=invitation["org"]) if invitation else t["title"]
        return view(t, title, sign_in_body(t, invite=invite, invitation=invitation, **fields), status)

    async def home(t: dict, session: dict, **notices):
        orgs = await run(teams.orgs_of, session["account"])
        return view(t, t["title"], home_body(t, orgs, session["data"]["csrf"], account=session["account"],
                                             can_create=accounts.teams_allowed(session["account"]), **notices))

    async def org_page(t: dict, session: dict, org_id: str, **extra):
        account = session["account"]
        try:
            org = await run(teams.org, org_id, account)
        except Refused as refused:
            return await home(t, session, error=_said(t, refused))
        apps = await run(grants.apps_of, account)
        levels = {app["client_id"]: await run(teams.app_level, org_id, account, app["client_id"]) for app in apps}
        return view(t, org["name"], org_body(t, org, account, session["data"]["csrf"], apps, levels, **extra))

    async def space_page(t: dict, session: dict, org_id: str, space_id: str, **notices):
        account = session["account"]
        try:
            org = await run(teams.org, org_id, account)
            space = next((s for s in org["spaces"] if s["id"] == space_id), None)
            if space is None:
                raise Refused(NOT_FOUND)
            memories = (await run(teams.memories, org_id, space_id, account, session["key"]) if space["read"]
                        else None)
        except Refused as refused:
            return await home(t, session, error=_said(t, refused))
        return view(t, space["name"], space_body(t, org, space, memories, account, session["data"]["csrf"],
                                                 **notices))

    async def invited(t: dict, session: dict, token: str):
        invitation = await invitation_of(token)
        if invitation is None:
            return view(t, t["title"], f'<p class="error">{t["invite_expired"]}</p>', status=400)
        e = html.escape
        account, csrf = session["account"], session["data"]["csrf"]
        # Who is signed in, with a way out: whoever opens the link on a shared browser must not accept as someone else.
        return view(t, t["invited_title"].format(org=invitation["org"]),
                    _top(t, account, csrf, [_link(t["orgs"]), e(invitation["org"])])
                    + f'<p class="lead">{e(t["invited_by"].format(by=invitation["by"], org=invitation["org"]))}</p>'
                    f'<p class="note">{t["invited_note"]}</p>'
                    f'<p class="note">{e(t["accept_as"].format(account=account))}</p>'
                    + _button(t, t["accept"], "", action="accept", csrf=csrf, invite=token))

    async def page(request):
        t, params = texts(request), request.query_params
        session = signed_in(request)
        if session is None:
            return await sign_in_page(t, invite=params.get("invite", ""))
        if params.get("invite"):
            return await invited(t, session, params["invite"])
        if params.get("org") and params.get("space"):
            return await space_page(t, session, params["org"], params["space"])
        if params.get("org"):
            return await org_page(t, session, params["org"])
        return await home(t, session)

    async def sign_in(t: dict, form):
        name, password = str(form.get("name", "")).strip().lower(), str(form.get("password", ""))
        invite = str(form.get("invite", ""))
        if not NAME.fullmatch(name):
            return await sign_in_page(t, invite=invite, status=400, name=name, error=t["bad_password"])
        if accounts.waiting(name):
            error = t["wait"].format(n=accounts.waiting(name))
            return await sign_in_page(t, invite=invite, status=429, name=name, error=error)
        key = await anyio.to_thread.run_sync(accounts.open, name, password, limiter=limiter)
        if key is None:
            return await sign_in_page(t, invite=invite, status=400, name=name, error=t["bad_password"])
        # From now on this person can be handed a space's key while they are away (remote_keys.py).
        await run(accounts.key_pair, name, key)
        session = grants.issue(SESSION, client_id="", seconds=SESSION_SECONDS, account=name, key=key,
                               data={"csrf": secrets.token_urlsafe(24)})
        response = back_to(invite=invite)
        response.set_cookie(COOKIE, session, max_age=SESSION_SECONDS, path=PATH, secure=secure, httponly=True,
                            samesite="strict")
        return response

    async def answer(request):
        t = texts(request)
        form = await request.form()
        action = str(form.get("action", ""))
        if action == "signin":
            return await sign_in(t, form)
        session = signed_in(request)
        if session is None:
            return await sign_in_page(t, invite=str(form.get("invite", "")), status=401, error=t["session_over"])
        if not hmac.compare_digest(str(form.get("csrf", "")), session["data"]["csrf"]):
            return view(t, t["title"], f'<p class="error">{t["bad_form"]}</p><p><a href="{PATH}">{t["back"]}</a></p>',
                        403)
        account, key = session["account"], session["key"]
        org_id, space_id, target = str(form.get("org", "")), str(form.get("space", "")), str(form.get("target", ""))
        try:
            if action == "signout":
                grants.drop(session["token"])
                response = back_to()
                response.delete_cookie(COOKIE, path=PATH)
                return response
            if action == "create_org":
                return back_to(org=await run(teams.create_org, account, key, str(form.get("name", ""))))
            if action == "accept":
                name = await run(take_invitation, str(form.get("invite", "")), account, key)
                if name is None:
                    return view(t, t["title"], f'<p class="error">{t["invite_expired"]}</p>', status=400)
                return await home(t, session, message=t["joined"].format(org=name))
            if action == "invite":
                carried = {}

                def issue(space_key, data):
                    carried["key"] = space_key
                    return grants.issue(remote_teams.INVITE, client_id="", seconds=remote_teams.INVITE_SECONDS,
                                        account=account, key=space_key, data=data)

                token = await run(teams.invite, org_id, account, key, issue)
                return await org_page(t, session, org_id, link=f"{issuer}{PATH}?{urlencode({'invite': token})}",
                                      link_pending=carried.get("key") is None)
            if action == "create_space":
                manager = str(form.get("manager", "")).strip().lower() or None
                space = await run(teams.create_space, org_id, account, key, str(form.get("name", "")), manager)
                return back_to(org=org_id, space=space)
            if action == "set_role":
                await run(teams.set_role, org_id, account, target, str(form.get("role", "")))
                return back_to(org=org_id)
            if action == "remove_member":
                await run(teams.remove_member, org_id, account, target)
                return back_to(org=None if target == account else org_id)
            if action == "app_level":
                await run(teams.set_app_level, org_id, account, str(form.get("client", "")),
                          str(form.get("level", "")))
                return back_to(org=org_id)
            if action == "add_member":
                await run(teams.add, org_id, space_id, account, key, target.strip().lower(),
                          str(form.get("role", MEMBER)), bool(form.get("write")))
                return back_to(org=org_id, space=space_id)
            if action == "confirm":
                await run(teams.confirm, org_id, space_id, account, key, target)
                return back_to(org=org_id, space=space_id)
            if action == "remove_from_space":
                await run(teams.remove, org_id, space_id, account, target)
                return back_to(org=org_id, space=None if target == account else space_id)
            if action == "remember":
                said = await run(lambda: teams.remember(org_id, space_id, account, key, str(form.get("statement", "")),
                                                        kind=str(form.get("kind", "")) or None, source="Teams"))
                if said.startswith("Remembered") or said.startswith("Already"):
                    return back_to(org=org_id, space=space_id)
                return await space_page(t, session, org_id, space_id, error=said)
            if action == "forget":
                await run(teams.forget, org_id, space_id, account, key, str(form.get("memory", "")))
                return back_to(org=org_id, space=space_id)
        except Refused as refused:
            if space_id:
                return await space_page(t, session, org_id, space_id, error=_said(t, refused))
            if org_id:
                return await org_page(t, session, org_id, error=_said(t, refused))
            return await home(t, session, error=_said(t, refused))
        return back_to()

    return [Route(PATH, page, methods=["GET"]), Route(PATH, answer, methods=["POST"])]
