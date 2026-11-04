# INFINITO on a server: one memory for every agent

An agent on your computer can use a memory file on your computer; ChatGPT, Claude on the web or on
your phone, and an agent on another machine cannot. This is INFINITO's memory as a server on the
internet, `src/infinito3/remote.py`: the same four tools as the local server (remember, recall,
list_memories, forget), with one memory per person that all of that person's apps share. What you
tell ChatGPT, Claude remembers, and the other way round.

## Connect an app

Every app gets the same public address, `https://<host>/mcp`. The first time, the app opens
INFINITO's page: choose a username and a password (or sign in, if you already have a memory) and
that app is connected. Each other app is one more sign-in. To tell someone, send them
`https://<host>/`: that page shows the address, with a copy button, and the steps below.

- **Claude** (desktop, web, phone and Claude Code in the app): Settings → Connectors → **Add custom
  connector**, name `INFINITO`, URL `https://<host>/mcp`, then **Connect**.
- **ChatGPT** (Plus or Pro): Settings → Apps & Connectors → Advanced settings, turn on **Developer
  mode**; create a connector with that URL and authentication **OAuth**.
- **Codex** (app and terminal): `codex mcp add infinito --url https://<host>/mcp`, then
  `codex mcp login infinito` to sign in.
- **Claude Code in a terminal**: `claude mcp add --transport http --scope user infinito https://<host>/mcp`,
  then `/mcp` to sign in.

The menu names change now and then; the steps stay the same.

**When you create your memory you get a recovery code.** Keep it on paper or in your password
manager: with it you can set a new password if you forget yours, and keep your memory. Changing the
password signs every app out, so a lost phone stops working too.

## What the person should know

- **Nobody else can read your memory, whoever runs the server included.** It is sealed on disk
  (AES-256-GCM) with a random key that is stored only locked under your password, under your
  recovery code and under each app's sign-in, of which only a hash is kept. The files and their
  backups are unreadable without one of those. The memory is opened only while one of your
  requests is served. No access log is written; the service log notes only how far an app got
  signing in (the step and its status, never a name, an address or a token).
- What sealing cannot stop is someone who changes the server's code to capture passwords as they
  arrive. Use a server run by someone you trust, as with any service.
- Lose both the password and the recovery code and the memory cannot be opened again, by you or by
  anyone.
- Forgetting works as in the local version: deleted for good, with the history of each fact.
- There is no vault here: its windows can only open on your own screen. Keep the local server for it
  (below).

## Keep the vault on your computer

The local server (`infinito-mcp`) can serve only the vault, so the memory lives on the server and the
secrets stay on your computer. Start it with `INFINITO_LOCAL_MEMORY=off`: in Claude Code,

```bash
claude mcp add --scope user --env INFINITO_LOCAL_MEMORY=off infinito-vault -- infinito-mcp
```

and in Claude Desktop's `claude_desktop_config.json`, `"env": {"INFINITO_LOCAL_MEMORY": "off"}` in the
server's entry (edit it with Claude Desktop closed; it rewrites the file while it runs).

## Private addresses, the older way

Before sign-in, each person got a private address, `https://<host>/u/<secret>/mcp`, whose secret was
the only key. Those addresses keep working. To move such a memory into an account, open "Did you use
INFINITO with a private address?" when creating your memory and paste the address there: the memory
comes along and the address stops working. `infinito-remote copy-local` copies a computer's local
memory into an address's memory.

## Run it on a server

```bash
pip install "infinito-memory[mcp] @ git+https://github.com/webtilians/principiodelTodo"
infinito-remote serve --port 8500 --url https://memory.example.org   # behind an HTTPS proxy
```

`--url` (or `INFINITO_REMOTE_URL`) is the server's public address; without it there is no sign-in,
only private addresses. `deploy/vps/` installs it on a Debian or Ubuntu server with Caddy:
`bash remote/deploy/vps/setup.sh` as root, then `infinito-remote-person list` or `remove <name>` and
`infinito-remote-status`. It follows its GitHub branch every 2 minutes, its commands and service
included.
