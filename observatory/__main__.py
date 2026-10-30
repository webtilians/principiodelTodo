"""Run with: python -m observatory {demo,blind,analyze}."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .blinding import blind_runs
from .adapters import export_jsonl, parse_legacy_log, read_scalar_csv
from .core import analyze, load_jsonl, save_json


def generate_demo(directory: Path, seed: int) -> dict:
    """Synthetic verification fixture: three planted transitions, three null runs.

    This fixture validates the observer, not any claim about INFINITO itself.
    """
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / "raw.jsonl"
    if raw.exists():
        raise FileExistsError("demo raw observations already exist")
    with raw.open("x", encoding="utf-8") as out:
        for index in range(6):
            rng = random.Random(seed + index)
            planted = index < 3
            for step in range(60):
                value = rng.gauss(0, 0.2) + (4.0 if planted and step >= 30 else 0.0)
                out.write(json.dumps({
                    "run_id": ("planted" if planted else "null") + "_" + str(index),
                    "step": step,
                    "metrics": {"signal": value},
                    # Deliberately private; blind_runs strips it completely.
                    "condition": "planted_shift" if planted else "noise_control",
                    "seed": seed + index,
                }) + "\n")
    info = blind_runs(
        load_jsonl(raw), output_path=directory / "blinded.jsonl",
        key_file=directory / "PRIVATE_mapping.json",
    )
    report = analyze(load_jsonl(directory / "blinded.jsonl"), seed=seed)
    save_json(directory / "report.json", report)
    return {"directory": str(directory), "blind": info, "flagged": sum(c["flagged"] for c in report["candidates"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind time-series observatory; never a consciousness test")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic shifted and noise-only runs")
    demo.add_argument("--output-dir", type=Path, default=Path("observatory_results/demo"))
    demo.add_argument("--seed", type=int, default=42)
    blind = commands.add_parser("blind", help="Strip metadata and pseudonymize run IDs")
    blind.add_argument("--input", type=Path, required=True)
    blind.add_argument("--output", type=Path, required=True)
    blind.add_argument("--key-file", type=Path, required=True, help="PRIVATE; never commit")
    scan = commands.add_parser("analyze", help="Find statistically unusual temporal shifts")
    scan.add_argument("--input", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    scan.add_argument("--window", type=int, default=6)
    scan.add_argument("--permutations", type=int, default=199)
    scan.add_argument("--alpha", type=float, default=0.05)
    scan.add_argument("--min-effect", type=float, default=1.0)
    scan.add_argument("--seed", type=int, default=42)
    ingest_log = commands.add_parser("import-log", help="Import original INFINITO console logs")
    ingest_log.add_argument("--input", type=Path, required=True)
    ingest_log.add_argument("--output", type=Path, required=True)
    ingest_log.add_argument("--run-id", required=True)
    ingest_csv = commands.add_parser("import-csv", help="Import a scalar time series")
    ingest_csv.add_argument("--input", type=Path, required=True)
    ingest_csv.add_argument("--output", type=Path, required=True)
    ingest_csv.add_argument("--run-id", required=True)
    ingest_csv.add_argument("--step-column", default="")
    args = parser.parse_args()
    if args.command == "import-log":
        content = args.input.read_text(encoding="utf-8-sig")
        rows = parse_legacy_log(content, run_id=args.run_id)
        export_jsonl(args.output, rows)
        result = {"output": str(args.output), "observations": len(rows)}
    elif args.command == "import-csv":
        rows = read_scalar_csv(args.input, run_id=args.run_id, step_column=args.step_column)
        export_jsonl(args.output, rows)
        result = {"output": str(args.output), "observations": len(rows)}
    elif args.command == "demo":
        result = generate_demo(args.output_dir, args.seed)
    elif args.command == "blind":
        result = blind_runs(load_jsonl(args.input), output_path=args.output, key_file=args.key_file)
    else:
        result = analyze(
            load_jsonl(args.input), window=args.window, permutations=args.permutations,
            alpha=args.alpha, min_effect=args.min_effect, seed=args.seed,
        )
        save_json(args.output, result)
        result = {
            "report": str(args.output), "runs": result["run_count"],
            "flagged": sum(c["flagged"] for c in result["candidates"]),
            "recurrences": len(result["recurrences"]),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
