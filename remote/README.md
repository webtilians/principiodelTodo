# INFINITO for ChatGPT (remote memory)

ChatGPT only connects to MCP servers on the internet, not to programs on your computer. This is
INFINITO's memory as such a server: `src/infinito3/remote.py`, the same four tools as the local
server (remember, recall, list_memories, forget), with one private memory per person.

**What the person should know before using it**

- Each person has a private address, `https://<host>/u/<secret>/mcp`. Whoever has it can read and change
  that memory, so it is given privately and never posted.
- Their memories live on the server, not on their computer. The server keeps only a hash of each
  address and writes no access log, but whoever runs the server can read the memory files.
- Forgetting works as in the local version: deleted for good, with the history of each fact.
- There is no vault here: its windows can only open on the computer that runs the server.

## Connect it in ChatGPT

1. Settings → Apps & Connectors → Advanced settings: turn on **Developer mode** (Plus or Pro plan).
2. Create a connector: name `INFINITO`, MCP server URL = your private address, authentication
   **No authentication** (the address is the key). Accept the warning about custom connectors.
3. In a new chat, pick INFINITO from the tools (+ → Developer mode). Say "remember that…" or
   "what do you know about me?" to try it.

ChatGPT's menu names change now and then; the steps stay the same.

## Run it on a server

```bash
pip install "infinito-memory[mcp] @ git+https://github.com/webtilians/principiodelTodo"
INFINITO_REMOTE_URL=https://memory.example.org infinito-remote add ana   # prints Ana's address once
infinito-remote serve --port 8500                                         # behind an HTTPS proxy
```

`deploy/vps/` installs it on a Debian or Ubuntu server with Caddy: `bash remote/deploy/vps/setup.sh`
as root, then `infinito-remote-person add <name>`, `list` or `remove <name>`, and
`infinito-remote-status`. It follows its GitHub branch every 2 minutes.
