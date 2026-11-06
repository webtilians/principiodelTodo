# INFINITO on a server: one memory for every agent

An agent on your computer can use a memory file on your computer; ChatGPT, Claude on the web or on
your phone, and an agent on another machine cannot. This is INFINITO's memory as a server on the
internet, `src/infinito3/remote.py`: the same memory tools as the local server (remember, remember_many,
recall, list_memories, forget), with one memory per person that all of that person's apps share. What you
tell ChatGPT, Claude remembers, and the other way round.

## Connect an app

Every app gets the same public address, `https://<host>/mcp`. The first time, the app opens
INFINITO's page: choose a username and a password (or sign in, if you already have a memory) and
that app is connected. Each other app is one more sign-in. To tell someone, send them
`https://<host>/`: that page shows the address, with a copy button, and the steps below.

- **Claude** (desktop, web, phone and Claude Code in the app): Settings → Connectors → **Add custom
  connector**, name `INFINITO`, URL `https://<host>/mcp`, then **Connect**.
- **ChatGPT** (a paid plan, on the web): on `chatgpt.com/plugins`, **Add** → Add custom MCP server, with that
  URL and authentication **OAuth**, then **Create as a plugin**. Where that is missing, turn on **Developer
  mode** first in Settings → Apps & Connectors → Advanced settings. Without custom instructions it needs `@INFINITO`
  in each message; the front page gives a text to paste once so it looks by itself. A connector made before a server update keeps its old tools until **Refresh tools** in its
  settings. OpenAI's help lists writing for Business and Enterprise/Edu and read/fetch for Pro, and this changes
  often: if ChatGPT reads the memory but does not save, connect Claude as well.
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
- Forgetting works as in the local version: deleted for good, with the history of each fact. Older backups
  keep it, sealed, until they age out (see Backups).

## A vault from any app

ChatGPT or a phone cannot open a window on your computer, so on the server `vault_add` and `vault_reveal` answer
with a link to a page here. The master password and the values are typed and shown on that page: no tool can
receive a value or returns one, so they never reach the agent, its model or the chat. The agent sees only each entry's name and a note on what it
is for, which you can correct on the page; a note it shows is one typed there, never the Notes of an entry made
elsewhere.

- **It is a standard KeePass file** (`accounts/<name>/vault.kdbx`): a copy opens with KeePassXC or KeePassDX.
- **It is by invitation.** On the server: `infinito-remote-person vault-allow <name>` (`vault-deny` stops it). This
  server's owner keeps other people's locked secrets, which should be a choice.
- **Choose a master password you will not lose:** at least 14 characters, not your INFINITO password. There is no
  recovery code: without it, nobody can open the vault.
- **What it cannot stop:** the master password reaches the server to open the file, like your INFINITO password does,
  so someone who changes the server's code can capture it. A copy alone gives only a locked file, whose strength is
  your master password's. Check the address before typing: a link in a chat can be forged. Old copies keep the secrets
  they had, under the master password of that day, until they age out (see Backups).
- **Until the server has its own domain and copies off it, keep nothing critical there.**

Keep the local server (below) if you want your secrets on your own computer only.

## A memory for work

For someone who works with several clients, a memory also says what it is, where it came from and whose
it is. An agent can give each memory:

- `kind`: `fact`, `decision`, `agreement`, `pending` or `proposal`, so a proposal is never read as a decision;
- `source`: where it came from, such as "email from Ana" or "meeting with Marta";
- `project`: the client or project. The same sentence about two clients stays two memories, and one client's
  facts never replace another's;
- `date`: when it was said, so an old email takes its place among the rest.

`remember_many` saves everything a meeting or a document established in one call, so apps that ask before
each write ask once. `recall` and `list_memories` take `project` and `kind`, which answer "where were we with
Acme?" and "what's pending?". Secrets and contact details are refused here too.

INFINITO is the memory, not the assistant: it does not read your email, send reminders or move files. Whatever
does (ChatGPT, n8n, another agent) can share this memory instead of each keeping its own.

## The memory in every message

Agents look at the memory when they think to, which is not always. Claude Code and Codex can run a
hook before each message that brings the related memories in by themselves.

- **Codex** already has the memory connected, so its hook calls the server's recall directly. In
  `~/.codex/hooks.json`:
  `{"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "mcp_tool", "server": "infinito", "tool": "recall", "input": {"query": "${prompt}"}}]}]}}`
- **Claude Code**: sign the computer in once with `infinito-remote login https://<host>` (the server's page
  opens in the browser), then add to `~/.claude/settings.json`:
  `{"hooks": {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "infinito-remote", "args": ["context"], "timeout": 10}]}]}}`.
  The session is kept in `~/.infinito/remote-session.json`; `infinito-remote logout` deletes it. Without
  a session, or when the server does not answer within a few seconds, the hook adds nothing and the
  message goes on. A session's first message also brings what was left pending. Past 9,000 characters (Claude Code
  keeps only a preview over 10,000) the longest memories are shortened, and recall brings them whole.

## Where the work stands

An agent remembers facts about you and rarely notes where the work stopped, so "where were we?" finds nothing.
In Claude Code, a Stop hook takes care of it. In `~/.claude/settings.json`:
`{"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "infinito-remote", "args": ["checkpoint"], "timeout": 5}]}]}}`.

When the agent stops, the hook looks at the session's own transcript, on your computer, and sends nothing
anywhere. If files were edited since the last write to the memory, and at least 30 minutes have passed since that
write, or a commit or push was made, it asks the agent to save what is left as `pending` and to close what is done.
It asks once each time and never keeps a session from stopping. The agent writes the words, because only it knows
what is left. ChatGPT has no such hook: there, ask it to save.

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

## Backups

A copy holds only what is already sealed: each memory, each vault locked under its master password, and each
person's key locked under their password or address. Without those a copy opens nothing, so it can be kept anywhere;
the usernames and the size of each memory and vault are all that can be read in one. A person who loses their password and recovery code can't be saved
by a backup either: the copy is as locked as the original.

**Forgetting and copies.** `forget` deletes a memory for good from the live memory at once, but it stays,
sealed, in the copies made before, until they age out: 14 days on the server and 30 on the computer that
brings them. Restoring an old copy brings back what was forgotten since; ask again to forget it.

- `infinito-remote backup <folder> --keep 14` writes one more copy and checks it; `verify-backup <file>`
  checks one; `restore <file>` puts one back, moving what was there aside. Apps sign in again after a restore.
- `deploy/vps/` makes a copy every night into `/var/lib/infinito-remote-backups`. On that server,
  `infinito-remote-restore` lists them and `infinito-remote-restore <file>` restores one.
- Copies on the server do not survive losing the server. `deploy/windows/infinito-backup-pull.ps1 -Server
  root@<ip> -Install` schedules a Windows computer to bring them every day, check them and warn on screen if
  the newest is over 3 days old.

## Run it on a server

```bash
pip install "infinito-memory[mcp,semantic] @ git+https://github.com/webtilians/principiodelTodo"
pip install model2vec && python -m infinito3.semantic save /opt/infinito-models/potion-multilingual-int8   # once
INFINITO_EMBEDDINGS=/opt/infinito-models/potion-multilingual-int8 \
  infinito-remote serve --port 8500 --url https://memory.example.org   # behind an HTTPS proxy
```

`--url` (or `INFINITO_REMOTE_URL`) is the server's public address; without it there is no sign-in,
only private addresses. With a model in `INFINITO_EMBEDDINGS`, recall finds memories by meaning
("what music do I like?" finds "my favourite songs are…"): a static multilingual model, about
100 MB of RAM with its vectors memory-mapped, no GPU. Without it, recall matches words. Saving the
model needs about 1.5 GB of RAM; on a small server, save it on another computer and copy the folder.
`deploy/vps/` installs it on a Debian or Ubuntu server with Caddy:
`bash remote/deploy/vps/setup.sh` as root, then `infinito-remote-person list` or `remove <name>` and
`infinito-remote-status`. It follows its GitHub branch every 2 minutes, its commands and service
included.
