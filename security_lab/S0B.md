# INFINITO Security S0-B — MCP adversarial on synthetic OAuth clone

**Branch:** `feat/security-s0b-mcp-agent-20261109`  
**Depends on:** S0-A draft PR #136. See [frozen S0-B protocol](../docs/INFINITO_SECURITY_S0B_PROTOCOL_2026-11-09.md).

S0-B adds a **feedback-driven, bounded deterministic adversarial agent**, real OAuth 2.1 sign-in (test client), **real MCP tool calls** to the same `build_app` HTTP ASGI app used by INFINITO, and a **real KeePass vault created through the synthetic account's web page**. It does **not** use an LLM, perform automatic vulnerability exploitation or learn policies yet.

Only cloned temporary accounts and markers are used. The agent cannot choose its target URL: its MCP call adapter is bound to the test fixture, which serves only `127.0.0.1` in a Docker networkless container. Its tool catalog and budgets are enforced by code, not by prompts. URLs containing OAuth tokens and vault links never appear in the success summary.

## Running

From the repository root:

~~~bash
docker build -f security_lab/Dockerfile.s0b -t infinito-security-s0b .
docker run --rm   --network none --read-only --cap-drop ALL   --security-opt no-new-privileges   --pids-limit 128 --memory 1024m --cpus 2   --tmpfs /tmp:rw,nosuid,size=128m,mode=1777   infinito-security-s0b
~~~

Do not publish host ports, mount volumes, pass INFINITO variables or secrets, or execute against any production endpoint. The networkless container allows **internal loopback** for MCP/OAuth and denies external networking. Tests also intercept outbound sockets and refuse any non-loopback destination.

## Gates

1. Unauthenticated and fake OAuth bearer requests receive 401.
2. Two synthetic accounts sign in through real OAuth and invoke real MCP tools.
3. A bounded attacker attempts cross-account recall/list/forget; independent asserts detect disclosure and unauthorized deletion.
4. The actual server denies writes after the app's permission is changed to read-only.
5. A memory containing instructions hostile to the defender is labelled as **untrusted data** and never appears in the other account; **not a claim that an LLM resists prompt injection**.
6. A synthetic secret goes through the real vault web page, never through an MCP tool argument. Vault list/reveal responses exclude its value and the reveal page requires a master password.
7. Uninvited account cannot use vault tools; invitations do not grant access to someone else's vault.
8. Agent tool-call cap and allowlist prevent scope expansion.
9. Network guard prohibits non-loopback TCP endpoints and binds.

Read the [GitHub Actions workflow](../.github/workflows/infinito-security-s0b.yml) for the reproducible run. Existing tests already cover additional OAuth and vault cases, but S0-B does not establish security of the deployed server.

## Next: S0-C

A separate, preregistered study can compare an authenticated, sandboxed **real LLM adversarial agent** in two conditions: no memory and INFINITO episodic memory, **same tools, models, action/time/token budgets and cloned worlds**. Do not use these S0-B tests as confirmation data. Evaluate only confirmed, reproducible defects with an independent oracle. Approval and manual review are mandatory before deploying any defense.
