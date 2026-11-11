"""CLI for a synthetic, reproducible single-day cognitive audit."""
import argparse
import json
from pathlib import Path

from .fixture import fictional_day
from .journal import Journal, canonical


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Write fixed synthetic journal as JSON and Markdown")
    mode.add_argument("--replay", action="store_true", help="Validate journal and compare it to the fixture")
    mode.add_argument("--export", action="store_true", help="Print deterministic JSON (or Markdown) to stdout")
    p.add_argument("--output-dir", type=Path, default=Path("work/cognitive-diary-p0"))
    p.add_argument("--format", choices=("json", "markdown"), default="json")
    args = p.parse_args()
    expected = fictional_day()
    if args.run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "journal.json").write_text(json.dumps(expected.snapshot(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.output_dir / "diary.md").write_text(expected.markdown(), encoding="utf-8")
        print(f"P0 synthetic fixture written: {args.output_dir}")
    elif args.replay:
        saved = json.loads((args.output_dir / "journal.json").read_text(encoding="utf-8"))
        restored = Journal.load(saved)
        if canonical(restored.snapshot()) != canonical(expected.snapshot()):
            p.error("the supplied day is valid but does not match the sealed synthetic fixture")
        print(f"REPLAY PASS: {len(restored.entries)} events; hash={restored.entries[-1]['hash']}; model_calls=0; applied_policies=0")
    else:
        if args.format == "markdown":
            print(expected.markdown(), end="")
        else:
            print(json.dumps(expected.snapshot(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
