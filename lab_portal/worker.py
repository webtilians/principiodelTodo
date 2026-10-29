"""Detached single-run supervisor. A portal restart does not destroy experiment logs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from lab_portal.catalog import EXPERIMENTS, command


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_status(folder: Path, changes: dict):
    target = folder / "job.json"
    data = json.loads(target.read_text(encoding="utf8"))
    data.update(changes)
    tmp = folder / ("job." + str(os.getpid()) + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
    os.replace(tmp, target)


def execute(folder: Path, root: Path, timeout: int):
    data = json.loads((folder / "job.json").read_text(encoding="utf8"))
    experiment = data["experiment"]
    if experiment not in EXPERIMENTS:
        raise ValueError("invalid experiment")
    outdir = folder / "output"
    outdir.mkdir(exist_ok=True)
    interpreter = (os.environ.get("LAB_F9_PYTHON", "")
                   if experiment.startswith("f9_") else sys.executable)
    if not interpreter or not Path(interpreter).is_file():
        raise RuntimeError("Intérprete del experimento no disponible")
    call = [interpreter, *command(experiment, data["params"], str(outdir))]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
    write_status(folder, {"status": "running", "started_at": timestamp()})
    code = -1
    status = "failed"
    try:
        with (folder / "stdout.log").open("ab", buffering=0) as log:
            log.write(("$ " + " ".join(call[:4]) + " [allowed arguments]\n").encode())
            process = subprocess.Popen(call, stdout=log, stderr=subprocess.STDOUT,
                                       cwd=root, env=environment)
            try:
                code = process.wait(timeout=timeout)
                status = "completed" if code == 0 and (outdir / "report.json").exists() else "failed"
            except subprocess.TimeoutExpired:
                log.write(f"\nLAB: límite de {timeout}s alcanzado. Finalizando.\n".encode())
                process.terminate()
                try:
                    process.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                status = "timeout"
            except BaseException:
                process.terminate()
                raise
    except BaseException as err:
        with (folder / "stdout.log").open("a", encoding="utf8") as log:
            log.write("\nLAB ERROR: " + repr(err) + "\n")
        status = "failed"
    finally:
        if (folder / "cancel.request").exists():
            status = "cancelled"
        write_status(folder, {"status": status, "finished_at": timestamp(), "exit_code": code})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    execute(args.job_dir.resolve(), args.repo_root.resolve(), args.timeout)


if __name__ == "__main__":
    main()
