"""Verify one externally reviewed proposal against the known synthetic fixture.

This is a supervised positive control, never a memory-comparison arm.
"""
import argparse
import difflib
import json
from pathlib import Path

from .core import digest, normalize, read_json, retrieve, snapshot, validate
from .verifier import verify
from .memory import IncidentMemory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    document = read_json(root / "security_lab/arena/viewer/run.json")
    trace = normalize(document, "levels-off")
    packet = {"trace": trace, "code": snapshot(root, ("src/infinito3/remote_apps.py",)),
              "sources": retrieve(json.dumps(trace))}
    response = read_json(args.response)
    result = verify(validate(response, packet))
    report = {"origin": "externally reviewed proposal; supervised positive control",
              "blinded": False, "learning_demonstrated": False,
              "trace_sha256": digest(document), "response_sha256": digest(response),
              "response": response, "sources": packet["sources"], "evaluation": result}
    (args.out / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if result["status"] == "verified":
        faulty = 'def allowed(level: Optional[str], tool: str) -> bool:\n    return True\n'
        diff = "".join(difflib.unified_diff(faulty.splitlines(True), response["replacement"].splitlines(True),
                       fromfile="synthetic/remote_apps.allowed", tofile="proposed/remote_apps.allowed"))
        (args.out / "proposal.diff").write_text(diff, encoding="utf-8")
    memory = IncidentMemory(args.out / "memory")
    try:
        memory.record("incident", digest(document), "observed")
        memory.record("hypothesis", digest(response), "proposed")
        memory.record("verified_defence" if result["status"] == "verified" else "failure",
                      digest(report), result["status"])
    finally:
        memory.close()
    print(json.dumps({"status": result["status"], "origin": report["origin"]}))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
