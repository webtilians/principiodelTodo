# INFINITO Security Arena

Real attacks against a throwaway clone of the INFINITO server, recorded step by step so a page can replay them.
For each step the page shows what reached the attacker, how it read it, what it decided, what it sent, what came
back, and the rule the referee applied.

## What it is, and what it is not

- **The server is the real one.** `recorder.py` starts `remote.build_app` on a free loopback port, in a temporary
  folder, with made-up accounts. Connections that would leave the machine are refused.
- **The attackers follow fixed rules, not a model.** Each one reads the answer it just got, by a rule on its words,
  and that reading decides its next move. Two moves are only made after a success: asking the vault to show an
  entry it has just seen named, and reading through a session a dead token managed to open.
- **The referee has no opinion.** It applies one of the rules in `RULES` to what really came back, or to Ana's
  memory read again with her own app. Ana's memory holds a canary, a made-up phrase nobody else should read.
- **A planted fault is not a flaw of the product.** The same attackers run against clones with one check taken out
  on purpose. That measures the attackers and the referee: a fault they do not notice is a blind spot of the arena.
- **Fifteen attacks failing says those fifteen ideas do not work**, not that no other exists.

## Record a run

From the repository root, with the server's dependencies installed:

```bash
python -m pip install -e ".[mcp,vault,dev]" uvicorn
python -m security_lab.arena.recorder
```

It writes `security_lab/arena/viewer/run.json` and prints one line per round:

```
current: 15/15 held, through: none
levels-off: 12/16 held, through: A10, A11, A13, A14, caught
disconnect-off: 14/16 held, through: A15, A15b, caught
wrong-account: 15/15 held, through: none, contained
```

`caught` means attacks got through, the referee proved it, and the same attacks fail again on a new clone with
the original code. `contained` means the fault was there and still no attack got through: with a token that
opens the wrong account, the other account's key does not open the sealed memory.

## Watch it

The page is one file with no dependencies. It reads `run.json` beside it, so serve the folder:

```bash
python -m http.server 8791 --bind 127.0.0.1 --directory security_lab/arena/viewer
```

and open <http://127.0.0.1:8791/>. It follows the browser's language, Spanish or English.

## Add to it

- **An attack** is one row in `MOVES`: who makes it, the door it knocks on, the defence expected to hold, a title
  in both languages, how it is made, and optionally the condition on that attacker's last answer.
- **A referee rule** is a function returning the rule's name, whether the defence held and what it saw, plus its
  entry in `RULES` so the page can show what it looked at.
- **A planted fault** is one branch in `planted()` and its entry in `FAULTS`.

The recording's fields are meant to carry a model's prompt and output unchanged when an attacker or a defender
is a model: `arrived` is what it was given, `decided` what it chose and why.

## Tests

```bash
python -m pytest tests/test_security_arena_v3.py -q
```
