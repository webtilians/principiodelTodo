# INFINITO Desktop (phase 1)

INFINITO's nervous system on a Windows PC, built phase by phase from
[the dossier](../docs/INFINITO_DESKTOP_DOSSIER_2026-11-27.md). Phase 1 is an icon in the tray. It says at a glance
whether INFINITO works on this PC and warns you when something stops working. The memory itself stays in your account.

- **Green, amber or red.** Green means everything is well, amber means a warning or something the app could not
  read, and red means a failure. Click the icon for the state of each piece, with a link to where it is fixed.
  Right-click it for the menu: *Estado*, *Qué recojo y qué envío*, *Pausa*, *Abrir mi memoria*, *Ajustes* and
  *Salir de INFINITO Desktop*.
- **Six checks:**

  | Check | What it does |
  |---|---|
  | The app's own session | Renews it |
  | The server | Answers its OAuth metadata |
  | Claude Code's hooks | `infinito-remote context` and `checkpoint`, and whether they have a session |
  | The cognitive engine | Read through its own summary, `cognition/status.py` |
  | The backups | The task «INFINITO copias de seguridad» and its `pull.log`, with the day of the last good copy |
  | The scheduled tasks | The «INFINITO» ones |

  What a check cannot read is «no lo sé», never «bien».
- **A notification only when something gets worse,** at most once a day for each check. Clicking it opens the page
  or the folder where it is fixed.
- **A device's sign-in.** The app signs in as «INFINITO Desktop en <equipo>» with the scope `infinito:device`. The
  sign-in page says it is a device, and its token opens no part of the memory, not even reading it
  (`src/infinito3/remote_device_api.py`). The session is sealed with Windows' DPAPI in
  `~/.infinito/desktop/session.dpapi`: only you, on this PC, can open it.

## What it reads and what it sends

| What | Where | Leaves the PC? |
|---|---|---|
| Whether the engine is paused, its last review and whether it was judged, today's spending, proposals waiting, whether its diary adds up | `~/.infinito/cognition/`, through `cognition.status.summary`. It never opens `session.json`, only checks it is there | No |
| The last lines of the backups' log and the names of the copies | `~/.infinito/copias-servidor/` | No |
| The «INFINITO» scheduled tasks: next run, last result, on or off | `schtasks /query` | No |
| Whether the hooks are in Claude Code's settings, and whether their session file is there (never opened) | `~/.claude/settings.json`, `~/.infinito/remote-session.json` | No |
| Whether the server answers | `GET https://infinitomemory.com/.well-known/oauth-authorization-server` | Only the request |
| Its own sign-in: register, sign in, renew, end | `/register`, `/authorize`, `/token`, `/revoke` | Only to infinitomemory.com |

Nothing goes to your memory in phase 1. One module, `server.py`, talks to the server, from a closed list of paths,
and a test fails if any other module reaches the network.

## Run it

From a checkout, with Python 3.10+ on Windows:

```powershell
pythonw desktop\infinito-desktop.pyw
```

`python desktop\infinito-desktop.pyw --check` prints every check as JSON and exits. `--login` and `--logout` sign
the app in and out.

The executable is built on every PR by `.github/workflows/desktop.yml`, with PyInstaller `--onedir` on
`windows-2022`. Download the `infinito-desktop-windows` artifact, check its SHA-256 against the one in the run's
summary, unzip it and open `INFINITO Desktop.exe`. It is not signed, so Windows SmartScreen asks first: *More info →
Run anyway*.

Its own folder is `~/.infinito/desktop/` (or `INFINITO_DESKTOP_HOME`). It holds:

- `session.dpapi`, the session;
- `state.json`, which keeps the settings, what each check last said, the day each was last notified and the device id.
