"""Authenticated local INFINITO lab API. No GitHub token and no remote shell."""
from __future__ import annotations

import hmac
import json
import os
import re
import signal
import subprocess
import time
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict

from lab_portal.catalog import EXPERIMENTS, ParameterError, describe, normalize
from lab_portal.summaries import summarize

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
JOB_ID = re.compile(r"^[a-f0-9]{32}$")
START_LOCK = Lock()
TERMINAL = {"completed", "failed", "cancelled", "timeout", "interrupted"}
app = FastAPI(title="INFINITO Laboratory", docs_url=None, redoc_url=None, openapi_url=None)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str
    params: dict = {}


def workspace():
    folder = Path(os.environ.get("LAB_DATA_DIR", str(ROOT / "observatory_results" / "portal_runs"))).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def now():
    return datetime.now(timezone.utc).isoformat()


def require_auth(x_lab_token: str | None = Header(default=None)):
    expected = os.environ.get("LAB_API_TOKEN", "")
    if not expected or expected.startswith("REEMPLAZAR_"):
        raise HTTPException(503, detail="Configura una clave privada real en LAB_API_TOKEN")
    if not x_lab_token or not hmac.compare_digest(x_lab_token, expected):
        raise HTTPException(401, detail="Clave de acceso incorrecta")


def process_alive(pid: int):
    if os.name != "nt":
        os.kill(pid, 0)
        return True
    # os.kill(pid, 0) terminates processes on Windows; query a handle instead.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        if ctypes.get_last_error() == 5:
            return True  # Access denied is not proof of termination.
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def revision():
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"],
            cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=3).strip()
    except (OSError, subprocess.SubprocessError):
        return None


# Capture loaded code, rather than a checkout that an updater may later change.
CODE_REVISION = revision()


def read_job(folder: Path):
    try:
        job = json.loads((folder / "job.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        raise HTTPException(404, "Ejecución no encontrada")
    # A cancelled job must remain cancelled even if the worker was terminated
    # before writing final metadata.
    if (folder / "cancel.request").exists():
        job["status"] = "cancelled"
    elif job["status"] in ("queued", "running"):
        try:
            pid = int((folder / "worker.pid").read_text())
            if not process_alive(pid):
                job["status"] = "interrupted"
        except (OSError, ValueError):
            # Allow the worker time to spawn immediately after writing metadata.
            if ((folder / "worker.pid").exists() or
                    time.time() - (folder / "job.json").stat().st_mtime > 30):
                job["status"] = "interrupted"
    return job


def find_job(job_id: str):
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "ID inválido")
    folder = workspace() / job_id
    if not folder.is_dir():
        raise HTTPException(404, "Ejecución no encontrada")
    return folder


def list_jobs():
    folders = sorted((p for p in workspace().iterdir()
                      if p.is_dir() and JOB_ID.fullmatch(p.name)), reverse=True)
    result = []
    for p in folders:
        try:
            result.append(read_job(p))
        except HTTPException:
            pass
    result.sort(key=lambda job: job.get("created_at", ""), reverse=True)
    return result


def pending_job():
    return next((job for job in list_jobs() if job["status"] in ("running", "queued")), None)


@app.get("/")
def homepage():
    return FileResponse(STATIC / "index.html", media_type="text/html",
                        headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/assets/{asset}")
def asset(asset: str):
    if asset not in ("lab.css", "lab.js"):
        raise HTTPException(404)
    mime = "text/css" if asset.endswith(".css") else "text/javascript"
    return FileResponse(STATIC / asset, media_type=mime,
                        headers={"Cache-Control": "no-store"})


@app.get("/api/ping", dependencies=[Depends(require_auth)])
def ping():
    return {"ok": True, "workspace": str(workspace()), "experiments": len(EXPERIMENTS),
            "revision": CODE_REVISION}


@app.get("/api/catalog", dependencies=[Depends(require_auth)])
def catalog():
    return {"experiments": describe(), "max_concurrent": 1,
            "timeout_seconds": int(os.environ.get("LAB_MAX_SECONDS", "14400"))}


@app.get("/api/jobs", dependencies=[Depends(require_auth)])
def jobs():
    return {"jobs": list_jobs()}


@app.post("/api/jobs", status_code=201, dependencies=[Depends(require_auth)])
def start(submission: Submission):
    try:
        params = normalize(submission.experiment, submission.params)
    except ParameterError as error:
        raise HTTPException(422, detail=str(error))
    if submission.experiment.startswith("f9_"):
        # Heavy Torch/MiniGrid stacks are never imported into the web process.
        # Admin must explicitly provision a separate F9 interpreter and RAM.
        if os.environ.get("LAB_F9_ENABLED") != "1":
            raise HTTPException(503, "F9 requiere habilitación del administrador y entorno Python independiente")
        interpreter = os.environ.get("LAB_F9_PYTHON", "")
        if not interpreter or not Path(interpreter).is_file():
            raise HTTPException(503, "LAB_F9_PYTHON no apunta a un intérprete F9 instalado")
    else:
        for name in EXPERIMENTS[submission.experiment].dependencies:
            try:
                __import__(name)
            except ImportError:
                raise HTTPException(422, f"Dependencia {name} no disponible")
    with START_LOCK:
        active = pending_job()
        if active:
            raise HTTPException(409, f"Ya se ejecuta {active['id'][:8]}; espera o cancélala")
        job_id = uuid.uuid4().hex
        folder = workspace() / job_id
        folder.mkdir(mode=0o700)
        info = {"id": job_id, "experiment": submission.experiment,
                "params": params, "status": "queued", "created_at": now(),
                "started_at": None, "finished_at": None, "exit_code": None,
                "note": EXPERIMENTS[submission.experiment].warning,
                "revision": CODE_REVISION}
        (folder / "job.json").write_text(json.dumps(info, indent=2), encoding="utf8")
        seconds = min(max(int(os.environ.get("LAB_MAX_SECONDS", "14400")), 60), 86400)
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "lab_portal.worker", "--job-dir", str(folder),
                 "--repo-root", str(ROOT), "--timeout", str(seconds)],
                cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=(os.name != "nt"),
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
        except OSError as error:
            info.update(status="failed", finished_at=now(), error=type(error).__name__)
            (folder / "job.json").write_text(json.dumps(info, indent=2), encoding="utf8")
            raise HTTPException(503, "No se pudo iniciar el trabajador") from error
        (folder / "worker.pid").write_text(str(proc.pid))
        Thread(target=proc.wait, daemon=True).start()
        return info


@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_auth)])
def job_detail(job_id: str):
    return read_job(find_job(job_id))


@app.get("/api/jobs/{job_id}/log", dependencies=[Depends(require_auth)])
def log(job_id: str):
    file = find_job(job_id) / "stdout.log"
    if not file.exists():
        return {"log": "Iniciando…"}
    with file.open("rb") as fp:
        fp.seek(0, 2)
        fp.seek(max(0, fp.tell() - 25000), 0)
        tail = fp.read()
    return {"log": tail.decode("utf8", errors="replace")}


@app.get("/api/jobs/{job_id}/summary", dependencies=[Depends(require_auth)])
def summary(job_id: str):
    folder = find_job(job_id)
    path = folder / "output" / "report.json"
    if not path.is_file():
        raise HTTPException(404, "Informe aún no disponible")
    try:
        contents = json.loads(path.read_text(encoding="utf8"))
        return summarize(read_job(folder)["experiment"], contents)
    except (ValueError, KeyError, TypeError):
        raise HTTPException(422, "Formato de informe no reconocido")


@app.get("/api/jobs/{job_id}/report", dependencies=[Depends(require_auth)])
def report(job_id: str):
    path = find_job(job_id) / "output" / "report.json"
    if not path.is_file():
        raise HTTPException(404, "Informe aún no disponible")
    return FileResponse(path, filename=f"infinito_{job_id[:8]}_report.json",
                        media_type="application/json")


@app.get("/api/jobs/{job_id}/download", dependencies=[Depends(require_auth)])
def download(job_id: str):
    folder = find_job(job_id)
    archive = folder / "results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [folder / "job.json", folder / "stdout.log"]:
            if p.is_file():
                z.write(p, p.name)
        output = folder / "output"
        if output.exists():
            for p in output.rglob("*"):
                relative = p.relative_to(output)
                regular = p.suffix in (".json", ".jsonl", ".log", ".csv", ".txt", ".npy")
                checkpoint = (read_job(folder)["experiment"].startswith("f9_")
                              and relative.parts[0] == "checkpoints"
                              and p.suffix == ".zip" and p.stat().st_size < 50_000_000)
                if (p.is_file() and not p.is_symlink()
                    and (regular or checkpoint)
                    and p.stat().st_size < 250_000_000):
                    z.write(p, str(Path("output") / relative))
    return FileResponse(archive, filename=f"infinito_lab_{job_id[:8]}.zip",
                        media_type="application/zip")


@app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_auth)])
def cancel(job_id: str):
    folder = find_job(job_id)
    job = read_job(folder)
    if job["status"] in TERMINAL:
        raise HTTPException(409, "Ejecución ya terminada")
    (folder / "cancel.request").write_text(now())
    try:
        pid = int((folder / "worker.pid").read_text())
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=8, check=False)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return {"status": "cancelled", "id": job_id}
