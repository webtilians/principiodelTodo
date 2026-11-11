"""python -m security_lab.analyst --out <new-directory> --model <local-model>"""
from __future__ import annotations

import argparse
import difflib
import inspect
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from .core import SYSTEM, digest, normalize, read_json, retrieve, snapshot, validate
from .memory import IncidentMemory
from .verifier import verify
from src.infinito3 import remote_apps

ROOT = Path(__file__).resolve().parents[2]
FAULTY = 'def allowed(level: Optional[str], tool: str) -> bool:\n    return True\n'


def local_request(path, data=None):
    # Fixed numeric loopback endpoint; no user/model URL, proxy, redirect or credentials.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError("redirect refused")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    request = urllib.request.Request("http://127.0.0.1:11434/api/" + path,
                                     data=None if data is None else json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=600) as response:
        raw = response.read(256_001)
    if len(raw) > 256_000:
        raise ValueError("model response too large")
    return json.loads(raw)


def infer(model, packet):
    # Raw Qwen chat framing is needed for the installed model's raw template.
    # JSON escapes control characters but framing strings in untrusted text must
    # also be neutralized. This is hygiene, not a semantic injection guarantee.
    data = json.dumps(packet, ensure_ascii=True).replace("<|", "<\\u007c").replace("|>", "\\u007c>")
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nUNTRUSTED DATA:\n{data}<|im_end|>\n<|im_start|>assistant\n"
    started = time.monotonic()
    result = local_request("generate", {"model": model, "prompt": prompt, "raw": True, "stream": False,
                            "format": "json", "options": {"temperature": 0, "seed": 1729,
                            "num_predict": 1400, "num_ctx": 8192, "stop": ["<|im_end|>"]}})
    return {"prompt": prompt, "response": result["response"], "seconds": time.monotonic() - started,
            "prompt_tokens": result.get("prompt_eval_count"), "output_tokens": result.get("eval_count"),
            "done_reason": result.get("done_reason"), "prompt_sha256": digest(prompt)}


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True, help="installed local Qwen-compatible Ollama model")
    parser.add_argument("--trace", type=Path, default=ROOT / "security_lab/arena/viewer/run.json")
    parser.add_argument("--round", default="levels-off")
    parser.add_argument("--demo-levels", action="store_true", help="verify only the bundled known planted fault")
    parser.add_argument("--synthetic", action="store_true", required=True,
                        help="operator attests input contains synthetic/public data only")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    trace_doc = read_json(args.trace)
    if args.demo_levels and (args.round != "levels-off" or
            digest(trace_doc) != digest(read_json(ROOT / "security_lab/arena/viewer/run.json"))):
        raise ValueError("demo verifier requires the bundled Arena recording and levels-off round")
    trace = normalize(trace_doc, args.round)
    code = snapshot(ROOT, ("src/infinito3/remote_apps.py",))
    original = code[0]["text"]
    trusted = inspect.getsource(remote_apps.allowed)
    if original.count(trusted) != 1:
        raise ValueError("trusted source mismatch")
    # Deliberate defect in an in-memory source snapshot, never product files.
    code[0]["text"] = original.replace(trusted, FAULTY) if args.demo_levels else original
    code[0]["sha256"] = digest(code[0]["text"])
    code[0]["scope"] = "synthetic defective snapshot" if args.demo_levels else "current product snapshot"
    sources = retrieve(json.dumps(trace))
    packet = {"trace": trace, "code": code, "sources": sources, "memory": ""}
    manifest = {"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "trace_sha256": digest(trace_doc), "catalog_sha256": digest(read_json(Path(__file__).with_name("catalog.json"))),
                "model": args.model, "model_inventory": local_request("tags"), "seed": 1729,
                "max_output_tokens": 1400, "scope": "engineering demo, no learning claim"}
    write(args.out / "manifest.json", manifest)
    memory = IncidentMemory(args.out / "memory")
    arms = {}
    try:
        memory.record("incident", digest(trace), "observed")
        recalled = memory.recall()  # frozen before either model outcome
        for arm in ("baseline", "memory"):
            packet["memory"] = "" if arm == "baseline" else recalled
            write(args.out / f"{arm}-input.json", packet)
            execution = infer(args.model, packet)
            write(args.out / f"{arm}-model.json", execution)
            try:
                answer = validate(json.loads(execution["response"]), packet)
                evaluation = verify(answer) if args.demo_levels else {
                    "status": "unverified", "reason": "imported trace has no independent lab reproduction"}
                if evaluation["status"] == "verified":
                    diff = "".join(difflib.unified_diff(FAULTY.splitlines(True), answer["replacement"].splitlines(True),
                                    fromfile="synthetic/remote_apps.allowed", tofile="proposed/remote_apps.allowed"))
                    (args.out / f"{arm}-proposal.diff").write_text(diff, encoding="utf-8")
            except (ValueError, SyntaxError) as exc:
                evaluation = {"status": "rejected", "reason": str(exc)}
            write(args.out / f"{arm}-evaluation.json", evaluation)
            arms[arm] = {"status": evaluation["status"], "prompt_tokens": execution["prompt_tokens"],
                         "output_tokens": execution["output_tokens"], "seconds": execution["seconds"]}
            memory.record("hypothesis", digest(execution), "proposed")
            category = "verified_defence" if evaluation["status"] == "verified" else "failure"
            memory.record(category, digest(evaluation),
                          "proposed" if evaluation["status"] == "unverified" else evaluation["status"])
    finally:
        memory.close()
    report = {"arms": arms, "learning_demonstrated": False,
              "limits": ["One known planted fault; no held-out study.", "Equal generation cap; input lengths differ.",
                         "Citation existence is checked, semantic support requires review.",
                         "Only exact trusted restoration is executable; proposed tests are text.",
                         "No deployment, arbitrary patch execution or personal memory access."]}
    write(args.out / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return 0 if all(a["status"] == "verified" for a in arms.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
