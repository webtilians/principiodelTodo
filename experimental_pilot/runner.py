"""P1 isolated, one-shot API runner. Offline unless explicitly authorized."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from urllib import request

from experimental_pilot.pilot import (
    MAX_ACTIONS, TOOL_NAMES, evaluate_actions, export_packets, seed_experiences,
)
from src.infinito3.persistent_memory import SQLiteCognitiveMemoryStore

PROVIDERS = ("openai", "anthropic")
VARIANTS = ("A_no_memory", "B_episodes", "C_verified_effects", "D_verified_plus_reflection")
INPUT_BYTES = 4096
OUTPUT_TOKENS = 256


class PilotError(Exception):
    """Only fixed, non-sensitive error codes may reach reports."""


def prompts():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        packets = export_packets(store)
    finally:
        store.close()
    # Allowlist, never serialize the full packet, referee or positive control.
    return {v: json.dumps({"task": packets["task"]["instruction"],
                           "observations": sorted(packets["variants"][v])},
                          ensure_ascii=True, sort_keys=True) for v in VARIANTS}


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PilotError("http_error")


def post(url, headers, body):
    req = request.Request(url, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **headers})
    # No retries, redirects, cookies, persisted sessions or response logging.
    try:
        with request.build_opener(NoRedirect).open(req, timeout=60) as response:
            raw = response.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError()
        return json.loads(raw)
    except Exception:
        raise PilotError("transport_error") from None


class APIClient:
    def __init__(self, provider, model, env):
        self.provider, self.model = provider, model
        self.key = env.get(provider.upper() + "_API_KEY", "")
        if not self.key:
            raise PilotError("missing_api_key")

    def complete(self, prompt, max_tokens):
        if self.provider == "openai":
            data = post("https://api.openai.com/v1/responses",
                        {"Authorization": "Bearer " + self.key},
                        {"model": self.model, "input": [{"role": "user", "content": prompt}],
                         "max_output_tokens": max_tokens, "store": False})
            if data.get("status") != "completed":
                raise PilotError("incomplete_response")
            text = "".join(c["text"] for item in data["output"]
                           if item.get("type") == "message" for c in item["content"]
                           if c.get("type") == "output_text")
        else:
            data = post("https://api.anthropic.com/v1/messages",
                        {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                        {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens})
            if data.get("stop_reason") != "end_turn":
                raise PilotError("incomplete_response")
            text = "".join(c["text"] for c in data["content"] if c.get("type") == "text")
        usage = data["usage"]
        counts = {k: usage[k] for k in ("input_tokens", "output_tokens")}
        if any(type(n) is not int or n < 0 for n in counts.values()):
            raise PilotError("invalid_usage")
        return text, counts


class FakeClient:
    def complete(self, prompt, max_tokens):
        # Deliberate abstention, not a hidden solver or simulated scientific result.
        return '{"steps":[]}', {"input_tokens": 0, "output_tokens": 0}


def positive_number(env, name):
    try:
        value = float(env[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError()
        return value
    except (KeyError, ValueError, TypeError):
        raise PilotError("invalid_budget_configuration") from None


def live_config(env, authorized):
    if not authorized:
        raise PilotError("live_not_authorized")
    limit = positive_number(env, "PILOT_MAX_USD")
    models, reserve = {}, 0.0
    for provider in PROVIDERS:
        prefix = provider.upper()
        model = env.get(prefix + "_MODEL", "")
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._:-]{0,127}", model):
            raise PilotError("missing_or_invalid_model")
        if not env.get(prefix + "_API_KEY"):
            raise PilotError("missing_api_key")
        models[provider] = model
        # Conservative byte-token envelope plus protocol overhead. Operator must
        # supply upper-bound prices; reserve ALL eight attempts before any call.
        rate_in = positive_number(env, prefix + "_INPUT_USD_PER_MILLION")
        rate_out = positive_number(env, prefix + "_OUTPUT_USD_PER_MILLION")
        reserve += 4 * ((INPUT_BYTES + 1024) * rate_in + OUTPUT_TOKENS * rate_out) / 1e6
    if reserve > limit:
        raise PilotError("budget_exceeded")
    return models, {"max_usd": limit, "reserved_usd": reserve}


def run(*, dry_run=True, authorized=False, env=None, client_factory=None):
    env = os.environ if env is None else env
    models, budget = ({p: "fake-v1" for p in PROVIDERS}, None)
    if not dry_run:
        models, budget = live_config(env, authorized)
    packets = prompts()
    if any(len(p.encode()) > INPUT_BYTES for p in packets.values()):
        raise PilotError("input_budget_exceeded")
    rows = []
    for provider in PROVIDERS:
        for variant, prompt in packets.items():
            row = {"provider": provider, "model": models[provider], "variant": variant,
                   "session": f"{provider}/{variant}", "status": "error", "steps": None,
                   "usage": None, "referee": None, "error": None,
                   "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                   "input_bytes": len(prompt.encode())}
            try:
                # A fresh client and single user message for every condition.
                client = (client_factory(provider, models[provider]) if client_factory else
                          FakeClient() if dry_run else APIClient(provider, models[provider], env))
                answer, row["usage"] = client.complete(prompt, OUTPUT_TOKENS)
                parsed = json.loads(answer)
                steps = parsed.get("steps") if isinstance(parsed, dict) else None
                if (not isinstance(steps, list) or len(steps) > MAX_ACTIONS
                        or any(not isinstance(s, str) or s not in TOOL_NAMES for s in steps)
                        or set(parsed) != {"steps"}):
                    raise PilotError("invalid_actions")
                row.update(status="ok", steps=steps, referee=evaluate_actions(steps))
            except Exception as exc:
                # Never persist raw replies, exception messages, headers or keys.
                safe = {"transport_error", "incomplete_response", "invalid_usage", "invalid_actions"}
                row["error"] = str(exc) if isinstance(exc, PilotError) and str(exc) in safe else "invalid_response"
            rows.append(row)
    return {"type": "fake-client-dry-run" if dry_run else "live-api-pilot",
            "model_calls": 0 if dry_run else 8, "budget": budget,
            "limits": {"input_bytes": INPUT_BYTES, "output_tokens": OUTPUT_TOKENS,
                       "actions": MAX_ACTIONS, "requests_per_condition": 1},
            "results": rows}


def markdown(report):
    lines = ["# P1 synthetic pilot", "", f"Mode: {report['type']}. No claim of learning.", "",
             "| Provider | Model | Variant | Status | Success | Input/output tokens |",
             "|---|---|---|---|---|---|"]
    for row in report["results"]:
        usage = row["usage"] or {}
        lines.append(f"| {row['provider']} | {row['model']} | {row['variant']} | "
                     f"{row['error'] or row['status']} | {(row['referee'] or {}).get('success', False)} | "
                     f"{usage.get('input_tokens', '?')}/{usage.get('output_tokens', '?')} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    parser.add_argument("--authorize-paid-api", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("work/pilot-p1"))
    args = parser.parse_args()
    try:
        report = run(dry_run=args.dry_run, authorized=args.authorize_paid_api)
    except PilotError as exc:
        parser.exit(2, f"P1 refused: {exc}\n")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown(report), encoding="utf-8")
    print("P1 reports written; " + report["type"])
    if any(row["status"] == "error" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
