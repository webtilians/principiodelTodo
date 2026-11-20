"""The INFINITO Teams page, <host>/teams: where people run their organization, with no AI in between.

Everything an app may not do is done here by a person, signed in with their password: creating an organization,
inviting, making projects, giving or taking access, confirming what an org admin proposed, and choosing what each of
their apps may do in the organization. The memories of a space can be read, added and taken back here too.

It has its own session, like the memory page (remote_memory.py): the password unseals the person's key, which is then
kept for a quarter of an hour wrapped under a token the server stores only as a hash, in a cookie for /teams alone that
scripts cannot read and other sites cannot send. Every form carries a token of its own against other sites' forms. A
person's own memory never appears here, and nobody in an organization can reach it from here.

Only served when the server runs with INFINITO_ENTERPRISE_ENABLED=1 (remote_teams.py).
"""
import hmac
import html
import secrets
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
main { max-width: 48rem; }
h2 { font-size: 1.05rem; margin: 1.6rem 0 .5rem; color: var(--accent); }
.row { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem .8rem; padding: .55rem 0;
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
.code.link { font-size: .9rem; letter-spacing: 0; text-align: left; }
"""

TEXT = {
    "es": {
        "title": "INFINITO Teams", "brand": "INFINITO · Teams",
        "sign_in_note": "Entra con tu cuenta de INFINITO. Tu memoria personal no aparece aquí, y nadie de tu "
                        "organización puede verla.",
        "name": "Tu nombre de usuario", "password": "Contraseña", "enter": "Entrar",
        "bad_password": "Ese nombre o esa contraseña no son correctos.",
        "wait": "Demasiados intentos. Espera {n} segundos.", "session_over": "La sesión ha caducado. Vuelve a entrar.",
        "bad_form": "Este formulario ya no vale. Vuelve a la página.", "back": "Volver", "signout": "Salir",
        "orgs": "Tus organizaciones", "no_orgs": "Aún no estás en ninguna organización. Para entrar en una, pide un "
                                                "enlace de invitación a quien la lleve.",
        "create_org": "Crear una organización", "org_name": "Nombre de la organización", "create": "Crear",
        "roles": {OWNER: "propietario", ADMIN: "administrador", MEMBER: "miembro", MANAGER: "responsable"},
        "spaces": "Espacios", "kinds": {GENERAL: "general", "project": "proyecto"},
        "access": {"write": "lees y añades", "read": "lees", PENDING: "pendiente de acceso", "none": "ves su nombre"},
        "new_space": "Nuevo proyecto", "space_name": "Nombre del proyecto",
        "manager": "Responsable (nombre de usuario, opcional)",
        "invite": "Invitar a alguien", "invite_go": "Crear un enlace de invitación",
        "invite_link": "Enlace de invitación. Sirve una vez y caduca en 7 días: mándaselo a esa persona.",
        "invite_pending": "Como no tienes la llave del espacio general, quien entre con este enlace esperará a que "
                          "el propietario le dé acceso.",
        "members": "Miembros", "make_admin": "Hacer administrador", "make_member": "Dejar de ser administrador",
        "remove": "Quitar", "leave_org": "Salir de esta organización",
        "apps": "Tus apps en esta organización",
        "apps_note": "Una app conectada a tu memoria no recibe nada de la organización hasta que tú lo eliges aquí.",
        "no_apps": "No tienes apps conectadas a tu memoria.",
        "levels": {"none": "Nada", "read": "Solo leer", "add": "Leer y añadir"}, "save": "Guardar",
        "audit": "Registro de cambios", "owner_reads": "El propietario de la organización puede leer este espacio.",
        "memories": "Recuerdos", "empty": "Este espacio aún no tiene recuerdos.", "add_memory": "Añadir un recuerdo",
        "statement": "Qué hay que recordar", "kind": "Tipo", "add": "Guardar", "forget": "Retirar",
        "kinds_memory": {"fact": "hecho", "decision": "decisión", "agreement": "acuerdo", "proposal": "propuesta",
                         "pending": "pendiente", "assumption": "supuesto", "to_verify": "por verificar"},
        "space_members": "Quién está en este espacio", "add_member": "Añadir a alguien", "add_person": "Añadir",
        "account": "Nombre de usuario",
        "role": "Papel", "can_write": "puede añadir", "read_only": "solo lee", "confirm": "Dar acceso",
        "pending_note": "Pendiente: el propietario o un responsable tiene que darle acceso.",
        "leave_space": "Salir de este espacio", "invited_title": "Te han invitado a una organización",
        "invited_note": "Al aceptar entrarás en ella como miembro. Tu memoria personal sigue siendo solo tuya.",
        "accept": "Aceptar", "invite_expired": "Este enlace de invitación ya no sirve. Pide uno nuevo.",
        "joined": "Ya estás en {org}.", "done": "Hecho.",
        "refusals": {NOT_FOUND: "No existe, o no es tuyo.",
                     "You may read this space but not add to it.": "Puedes leer este espacio, pero no añadir.",
                     "You are already in this organization.": "Ya estás en esta organización.",
                     "This account may not create organizations on this server.":
                         "Tu cuenta no puede crear organizaciones en este servidor.",
                     "Everyone in the organization is in the general space already.":
                         "Todos los de la organización están ya en el espacio general."},
    },
    "en": {
        "title": "INFINITO Teams", "brand": "INFINITO · Teams",
        "sign_in_note": "Sign in with your INFINITO account. Your own memory does not show here, and nobody in your "
                        "organization can see it.",
        "name": "Your username", "password": "Password", "enter": "Sign in",
        "bad_password": "That name or password is not right.",
        "wait": "Too many tries. Wait {n} seconds.", "session_over": "Your session ended. Sign in again.",
        "bad_form": "This form no longer works. Go back to the page.", "back": "Back", "signout": "Sign out",
        "orgs": "Your organizations", "no_orgs": "You are in no organization yet. To join one, ask whoever runs it "
                                                 "for an invitation link.",
        "create_org": "Create an organization", "org_name": "Organization name", "create": "Create",
        "roles": {OWNER: "owner", ADMIN: "admin", MEMBER: "member", MANAGER: "manager"},
        "spaces": "Spaces", "kinds": {GENERAL: "general", "project": "project"},
        "access": {"write": "you read and add", "read": "you read", PENDING: "waiting for access",
                   "none": "you see its name"},
        "new_space": "New project", "space_name": "Project name", "manager": "Manager (username, optional)",
        "invite": "Invite someone", "invite_go": "Make an invitation link",
        "invite_link": "Invitation link. It works once and ends in 7 days: send it to that person.",
        "invite_pending": "As you do not hold the general space's key, whoever joins with this link waits for the "
                          "owner to give them access.",
        "members": "Members", "make_admin": "Make admin", "make_member": "No longer admin", "remove": "Remove",
        "leave_org": "Leave this organization",
        "apps": "Your apps in this organization",
        "apps_note": "An app connected to your memory gets nothing of the organization until you choose it here.",
        "no_apps": "You have no apps connected to your memory.",
        "levels": {"none": "Nothing", "read": "Read only", "add": "Read and add"}, "save": "Save",
        "audit": "Record of changes", "owner_reads": "The organization's owner can read this space.",
        "memories": "Memories", "empty": "This space has no memories yet.", "add_memory": "Add a memory",
        "statement": "What to remember", "kind": "Kind", "add": "Save", "forget": "Take back",
        "kinds_memory": {"fact": "fact", "decision": "decision", "agreement": "agreement", "proposal": "proposal",
                         "pending": "pending", "assumption": "assumption", "to_verify": "to verify"},
        "space_members": "Who is in this space", "add_member": "Add someone", "add_person": "Add", "account": "Username",
        "role": "Role",
        "can_write": "may add", "read_only": "reads only", "confirm": "Give access",
        "pending_note": "Waiting: the owner or a manager has to give them access.",
        "leave_space": "Leave this space", "invited_title": "You are invited to an organization",
        "invited_note": "Accepting makes you a member of it. Your own memory stays yours alone.",
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


def sign_in_body(t: dict, *, invite: str = "", name: str = "", error: str = "") -> str:
    e = html.escape
    wrong = f'<p class="error" role="alert">{e(error)}</p>' if error else ""
    return (f'<p class="note">{t["sign_in_note"]}</p><form method="post" action="{PATH}">'
            f'{_hidden(action="signin", invite=invite or None)}'
            f'<label for="name">{t["name"]}</label><input id="name" name="name" type="text" value="{e(name)}" '
            'autocomplete="username" autocapitalize="none" spellcheck="false" required>'
            f'<label for="password">{t["password"]}</label><input id="password" name="password" type="password" '
            f'autocomplete="current-password" required>{wrong}<button type="submit">{t["enter"]}</button></form>')


def home_body(t: dict, orgs: List[dict], csrf: str, *, can_create: bool, message: str = "", error: str = "") -> str:
    e = html.escape
    rows = "".join(f'<div class="row"><div class="grow"><a href="{PATH}?{urlencode({"org": o["id"]})}">'
                   f'{e(o["name"])}</a></div><span class="tag">{t["roles"][o["role"]]}</span></div>' for o in orgs)
    create = (f'<h2>{t["create_org"]}</h2><form method="post" action="{PATH}">{_hidden(action="create_org", csrf=csrf)}'
              f'<label for="org_name">{t["org_name"]}</label><input id="org_name" name="name" type="text" '
              f'maxlength="60" required><button type="submit">{t["create"]}</button></form>') if can_create else ""
    return (_notice(message, error) + f'<h2>{t["orgs"]}</h2>' + (rows or f'<p class="note">{t["no_orgs"]}</p>')
            + create + _signout(t, csrf))


def _notice(message: str = "", error: str = "") -> str:
    e = html.escape
    return ((f'<p class="ok" role="status">{e(message)}</p>' if message else "")
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else ""))


def _signout(t: dict, csrf: str) -> str:
    return f'<div class="links">{_button(t, t["signout"], "small quiet", action="signout", csrf=csrf)}</div>'


def _access(t: dict, space: dict) -> str:
    return t["access"]["write" if space["write"] else "read" if space["read"] else
                       PENDING if space["state"] == PENDING else "none"]


def org_body(t: dict, org: dict, account: str, csrf: str, apps: List[dict], levels: Dict[str, str], *,
             link: str = "", link_pending: bool = False, message: str = "", error: str = "") -> str:
    e = html.escape
    org_id, role = org["id"], org["role"]
    out = [_notice(message, error), f'<p class="meta">{t["roles"][role]}</p>']
    if link:
        out.append(f'<p class="note">{t["invite_link"]}</p><div class="code link">{e(link)}</div>'
                   + (f'<p class="note">{t["invite_pending"]}</p>' if link_pending else ""))
    out.append(f'<h2>{t["spaces"]}</h2>')
    for space in org["spaces"]:
        target = f'{PATH}?{urlencode({"org": org_id, "space": space["id"]})}'
        name = f'<a href="{target}">{e(space["name"])}</a>' if space["read"] or space["add"] else e(space["name"])
        out.append(f'<div class="row"><div class="grow">{name} <span class="tag">{t["kinds"][space["kind"]]}</span>'
                   f'</div><span class="meta">{_access(t, space)}</span></div>')
    if role in (OWNER, ADMIN):
        out.append(f'<h2>{t["new_space"]}</h2><form method="post" action="{PATH}">'
                   f'{_hidden(action="create_space", csrf=csrf, org=org_id)}<label for="space_name">{t["space_name"]}'
                   f'</label><input id="space_name" name="name" type="text" maxlength="60" required>'
                   f'<label for="manager">{t["manager"]}</label><input id="manager" name="manager" type="text" '
                   f'autocapitalize="none" spellcheck="false"><button type="submit">{t["create"]}</button></form>')
        out.append(f'<h2>{t["invite"]}</h2>'
                   + _button(t, t["invite_go"], "small quiet", action="invite", csrf=csrf, org=org_id))
    if org["members"]:
        out.append(f'<h2>{t["members"]}</h2>')
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
        out.append(f'<details><summary>{t["audit"]}</summary>' + "".join(
            f'<div class="row meta"><div class="grow">{e(item["at"])} · {e(item["actor"])} · {e(item["action"])} '
            f'{e(item["target"])}</div><span>{e(item["outcome"])}</span></div>' for item in reversed(org["audit"]))
                   + '</details>')
    if role != OWNER:
        leave = _button(t, t["leave_org"], "small bad", action="remove_member", csrf=csrf, org=org_id, target=account)
        out.append(f'<div class="links">{leave}</div>')
    out.append(f'<div class="links"><a href="{PATH}">{t["back"]}</a></div>')
    return "".join(out)


def space_body(t: dict, org: dict, space: dict, memories: Optional[List[dict]], account: str, csrf: str, *,
               message: str = "", error: str = "") -> str:
    e = html.escape
    ids = {"org": org["id"], "space": space["id"]}
    out = [_notice(message, error),
           f'<p class="meta">{e(org["name"])} · {t["kinds"][space["kind"]]} · {_access(t, space)}</p>',
           f'<p class="note">{t["owner_reads"]}</p>']
    if memories is not None:
        out.append(f'<h2>{t["memories"]}</h2>')
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
        out.append(f'<h2>{t["add_memory"]}</h2><form method="post" action="{PATH}">'
                   f'{_hidden(action="remember", csrf=csrf, **ids)}<label for="statement">{t["statement"]}</label>'
                   f'<textarea id="statement" name="statement" maxlength="2000" required></textarea>'
                   f'<label for="kind">{t["kind"]}</label><select id="kind" name="kind">{options}</select>'
                   f'<button type="submit">{t["add"]}</button></form>')
    if space["members"]:
        out.append(f'<h2>{t["space_members"]}</h2>')
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
        roles = "".join(f'<option value="{r}">{t["roles"][r]}</option>' for r in (MEMBER, MANAGER))
        out.append(f'<h2>{t["add_member"]}</h2><form method="post" action="{PATH}">'
                   f'{_hidden(action="add_member", csrf=csrf, **ids)}<label for="target">{t["account"]}</label>'
                   f'<input id="target" name="target" type="text" autocapitalize="none" spellcheck="false" required>'
                   f'<label for="role">{t["role"]}</label><select id="role" name="role">{roles}</select>'
                   f'<label><input type="checkbox" name="write" value="1" checked> {t["can_write"]}</label>'
                   f'<button type="submit">{t["add_person"]}</button></form>')
    if space["kind"] != GENERAL and account in space["members"]:
        leave = _button(t, t["leave_space"], "small bad", action="remove_from_space", csrf=csrf, target=account, **ids)
        out.append(f'<div class="links">{leave}</div>')
    out.append(f'<div class="links"><a href="{PATH}?{urlencode({"org": org["id"]})}">{t["back"]}</a></div>')
    return "".join(out)


def routes(accounts, grants, teams, limiter, secure: bool, issuer: str = "") -> list:
    """GET and POST /teams. teams is the server's one remote_teams.Teams, shared with the company_* tools."""
    import anyio
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    policy = f"{_HEADERS['Content-Security-Policy']}; form-action 'self'; base-uri 'none'"

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

    def back_to(**where) -> RedirectResponse:
        wanted = {name: value for name, value in where.items() if value}
        return RedirectResponse(f"{PATH}?{urlencode(wanted)}" if wanted else PATH, status_code=303, headers=_HEADERS)

    async def home(t: dict, session: dict, **notices):
        orgs = await run(teams.orgs_of, session["account"])
        return view(t, t["title"], home_body(t, orgs, session["data"]["csrf"],
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
        if grants.find(remote_teams.INVITE, token) is None:
            return view(t, t["title"], f'<p class="error">{t["invite_expired"]}</p>', status=400)
        return view(t, t["invited_title"], f'<p class="note">{t["invited_note"]}</p>' + _button(
            t, t["accept"], "", action="accept", csrf=session["data"]["csrf"], invite=token))

    async def page(request):
        t, params = texts(request), request.query_params
        session = signed_in(request)
        if session is None:
            return view(t, t["title"], sign_in_body(t, invite=params.get("invite", "")))
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
            return view(t, t["title"], sign_in_body(t, invite=invite, name=name, error=t["bad_password"]), 400)
        if accounts.waiting(name):
            error = t["wait"].format(n=accounts.waiting(name))
            return view(t, t["title"], sign_in_body(t, invite=invite, name=name, error=error), 429)
        key = await anyio.to_thread.run_sync(accounts.open, name, password, limiter=limiter)
        if key is None:
            return view(t, t["title"], sign_in_body(t, invite=invite, name=name, error=t["bad_password"]), 400)
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
            return view(t, t["title"], sign_in_body(t, invite=str(form.get("invite", "")), error=t["session_over"]),
                        401)
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
                token = str(form.get("invite", ""))
                offer = grants.find(remote_teams.INVITE, token)
                if offer is None:
                    return view(t, t["title"], f'<p class="error">{t["invite_expired"]}</p>', status=400)
                name = await run(teams.accept, offer, account, key)
                grants.drop(token)
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
