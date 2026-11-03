# INFINITO demo

A public page where anyone can talk to INFINITO and to a plain memory at once, and see what
each one keeps: tell it something, give it a password, ask it to forget, ask a trick question.

- Both memories answer with the same model and only from what they remember; the chat history
  is not sent, so something forgotten cannot come back through it.
- INFINITO is the real core (`src/infinito3`) with an in-memory SQLite store and the LLM forget
  planner. The plain memory keeps every statement and retrieves the most similar ones.
- Secrets are blocked before anything sees them. The plain memory keeps them, as a plain memory
  would, but the demo never sends them to the model.
- Each visitor's memories live only in the server process, are never written to disk and are
  dropped after an idle hour or when the visitor erases them. Message text is never logged.
- Limits: messages per visitor and per connection, and an estimated daily spend.

## Run it locally

```bash
pip install -r demo/requirements.txt
python -m demo                       # http://127.0.0.1:8400, offline: simulated answers, real deletion
OPENAI_API_KEY=... python -m demo    # with gpt-5.6-luna and text-embedding-3-small
python -m pytest demo/tests          # needs pytest and httpx too
```

Settings are `DEMO_*` variables, read in `demo/service.py` (`Settings`).

## On the server

`demo/deploy/vps/setup.sh` installs it on the LAIN server as the `infinito-demo` service on
127.0.0.1:8400, behind Caddy at `demo.<ip>.sslip.io`, and updates it from GitHub every two
minutes. The OpenAI key goes in from the PC with `demo/deploy/vps/clave-openai.ps1`; on the
server, `infinito-demo-status` shows the address, version, key (set or not) and today's spend.
