#!/usr/bin/env python3
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "1060dcc9acd0e8d37ae0c6ff0d4d26552815a99f"


def run(args):
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    run(["git", "fetch", "--no-tags", "--depth=1", "origin", FROZEN])
    run(["git", "checkout", "--force", "--detach", FROZEN])
    python = "/usr/bin/python3.12"
    run([python, "-m", "pip", "install", "openai==3.14.0", "pytest==9.1.1"])
    run([python, "scripts/run_infinito3_trajectory_holdout_v10.py"])
    run([
        python,
        "scripts/run_infinito3_trajectory_holdout_v10.py",
        "--authorize-live-v10",
        "--expected-revision", FROZEN,
        "--output-dir", "trajectory-results/v10",
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
