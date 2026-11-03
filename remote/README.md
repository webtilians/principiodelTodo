# INFINITO on a server: one memory for every agent

An agent on your computer can use a memory file on your computer; ChatGPT, Claude on the web or on
your phone, and an agent on another machine cannot. This is INFINITO's memory as a server on the
internet, `src/infinito3/remote.py`: the same four tools as the local server (remember, recall,
list_memories, forget), with one private memory per person that all of that person's agents share.
What you tell ChatGPT, Claude Code remembers, and the other way round.

**What the person should know before using it**

- Each person has a private address, `https://<host>/u/<secret>/mcp`. Whoever has it can read and change
  that memory, so it is given privately and never posted. Each app you connect keeps a copy of it.
- **Nobody else can read the memory, whoever runs the server included.** The server keeps only a hash of
  each address, and each memory is sealed on disk (AES-256-GCM) with a key derived from the address, so
  the files and their backups are unreadable without it. A memory is opened in the server's memory only
  while one of its owner's requests is served, and no access log is written.
- What sealing cannot stop is someone who changes the server's code to capture addresses as requests
  arrive. Use a server run by someone you trust, as with any service.
- Lose the address and the memory cannot be opened again, by you or by anyone.
- Forgetting works as in the local version: deleted for good, with the history of each fact.
- There is no vault here: its windows can only open on your own screen. Keep the local server for it
  (below).

## Connect your agents

Every agent gets the same address. Type or paste it yourself; don't send it to an agent in a chat.

**ChatGPT** (Plus or Pro): Settings → Apps & Connectors → Advanced settings, turn on **Developer mode**.
Create a connector: name `INFINITO`, MCP server URL = your address, authentication **No authentication**
(the address is the key), and accept the warning about custom connectors. In a chat, pick INFINITO from
the tools (+ → Developer mode).

**Claude** (desktop, web and phone): Settings → Connectors → **Add custom connector**, name `INFINITO`,
URL = your address. It then works in every Claude app signed in to your account.

**Claude Code**: in your own terminal,

```bash
claude mcp add --transport http --scope user infinito-memory <your address>
```

ChatGPT's and Claude's menu names change now and then; the steps stay the same.

## Keep the vault on your computer

The local server (`infinito-mcp`) can serve only the vault, so the memory lives on the server and the
secrets stay on your computer. Start it with `INFINITO_LOCAL_MEMORY=off`: in Claude Code,

```bash
claude mcp add --scope user --env INFINITO_LOCAL_MEMORY=off infinito -- infinito-mcp
```

and in Claude Desktop's `claude_desktop_config.json`, `"env": {"INFINITO_LOCAL_MEMORY": "off"}` in the
server's entry (edit it with Claude Desktop closed; it rewrites the file while it runs).

To bring what the local memory already knows, run `infinito-remote copy-local` on that computer. It asks
for your address without showing it and copies every memory; secrets and duplicates are refused as
usual. The local file stays where it was until you delete it.

## Run it on a server

```bash
pip install "infinito-memory[mcp] @ git+https://github.com/webtilians/principiodelTodo"
INFINITO_REMOTE_URL=https://memory.example.org infinito-remote add ana   # prints Ana's address once
infinito-remote serve --port 8500                                         # behind an HTTPS proxy
```

`deploy/vps/` installs it on a Debian or Ubuntu server with Caddy: `bash remote/deploy/vps/setup.sh`
as root, then `infinito-remote-person add <name>`, `list` or `remove <name>`, and
`infinito-remote-status`. It follows its GitHub branch every 2 minutes. A memory written before sealing
existed is sealed, and its plain file overwritten, the first time its owner uses it.
