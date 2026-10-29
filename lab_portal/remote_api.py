"""Authenticated, allowlisted *pull* queue for a Windows F9 worker.

Only predefined F9 development jobs are distributed. Workers never expose
inbound ports, receive shell commands or see LAB_API_TOKEN. VPS holds its
own independent LAB_WORKER_TOKEN; transport requires HTTPS.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/worker", tags=["remote-execution"], include_in_schema=False)
ALLOWED = frozenset(("f9_memory", "f9_doorkey", "f9_dynamic", "f9b_memory_baselines"))
MAX_BUNDLE = 64 * 1024 * 1024
MAX_EXTRACTED = 90 * 1024 * 1024
MAX_FILES = 45
MAX_LOG_CHARS = 1800
MAX_QUEUED_SECONDS = 1800
MAX_HEARTBEAT_AGE = 300
WORKER_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
CHECKPOINT = re.compile(r"^checkpoints/[a-zA-Z0-9_-]+\.zip$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def authenticated_worker(x_lab_worker_token: str | None = Header(default=None)):
    expected = os.environ.get("LAB_WORKER_TOKEN", "")
    if (os.environ.get("LAB_F9_REMOTE_ENABLED") != "1"
            or len(expected) < 32 or not x_lab_worker_token
            or not hmac.compare_digest(x_lab_worker_token, expected)):
        raise HTTPException(status_code=401, detail="Worker no autorizado")


class WorkerInfo(BaseModel):
    worker_id: str = Field(min_length=1, max_length=64)
    revision: str = Field(min_length=40, max_length=40)


class Progress(BaseModel):
    claim_id: str = Field(min_length=32, max_length=32)
    log_tail: str = Field(default="", max_length=MAX_LOG_CHARS)


def validate_worker_id(worker: WorkerInfo):
    if not WORKER_ID.fullmatch(worker.worker_id):
        raise HTTPException(422, "Worker ID inválido")
    if not re.fullmatch(r"[0-9a-f]{40}", worker.revision):
        raise HTTPException(422, "SHA inválido")


def _server():
    # Lazily imported to avoid a circular dependency at module initialization.
    from lab_portal import server
    return server


def remote_job_state(folder: Path, job: dict):
    """Derive status from recorded lease without changing the running worker."""
    if job.get("executor") != "remote_windows":
        return job
    if job.get("status") not in ("queued", "running"):
        return job
    started = job.get("heartbeat_epoch") if job["status"] == "running" else job.get("created_epoch")
    if started is not None:
        age = time.time() - float(started)
        maximum = MAX_HEARTBEAT_AGE if job["status"] == "running" else MAX_QUEUED_SECONDS
        if age > maximum:
            job["status"] = "interrupted"
            job["error"] = "Worker sin conexión o cola caducada. Repite la ejecución."
    return job


def _write_job(folder: Path, record: dict):
    file = folder / "job.json"
    temp = folder / "job.remote.tmp"
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, file)


def _job_under_claim(folder: Path, claim: str):
    sv = _server()
    job = sv.read_job(folder)
    if (job.get("executor") != "remote_windows" or
            job.get("status") != "running" or
            not job.get("claim_id") or
            not hmac.compare_digest(claim, job["claim_id"])):
        raise HTTPException(409, "Trabajo no asignado o lease caducado")
    return job


@router.post("/ping", dependencies=[Depends(authenticated_worker)])
def ping(worker: WorkerInfo):
    validate_worker_id(worker)
    sv = _server()
    (sv.workspace() / "f9_worker_heartbeat.json").write_text(
        json.dumps({"worker_id": worker.worker_id, "revision": worker.revision,
                    "last_seen": utc_now(), "last_seen_epoch": time.time()}),
        encoding="utf-8")
    available = []
    for job in sv.list_jobs():
        if (job.get("executor") == "remote_windows"
                and job.get("status") == "queued"
                and job.get("revision") == worker.revision
                and job.get("experiment") in ALLOWED):
            available.append({"id": job["id"], "experiment": job["experiment"],
                              "params": job["params"], "revision": job["revision"]})
    return {"jobs": available[:1], "server_revision": sv.CODE_REVISION}


@router.post("/jobs/{job_id}/claim", dependencies=[Depends(authenticated_worker)])
def claim(job_id: str, worker: WorkerInfo):
    validate_worker_id(worker)
    sv = _server()
    with sv.START_LOCK:
        folder = sv.find_job(job_id)
        record = sv.read_job(folder)
        if (record.get("executor") != "remote_windows" or
                record.get("experiment") not in ALLOWED or
                record["status"] != "queued" or
                record["revision"] != worker.revision):
            raise HTTPException(409, "Trabajo no asignable a este worker/revisión")
        token = secrets.token_hex(16)
        record.update(status="running", worker_id=worker.worker_id,
                      started_at=sv.now(), heartbeat_epoch=time.time(),
                      claim_id=token)
        _write_job(folder, record)
    return {"id": job_id, "claim_id": token,
            "experiment": record["experiment"], "params": record["params"],
            "revision": record["revision"]}


@router.post("/jobs/{job_id}/heartbeat", dependencies=[Depends(authenticated_worker)])
def heartbeat(job_id: str, progress: Progress):
    sv = _server()
    folder = sv.find_job(job_id)
    if (folder / "cancel.request").exists():
        return {"cancel": True}
    with sv.START_LOCK:
        record = _job_under_claim(folder, progress.claim_id)
        record["heartbeat_epoch"] = time.time()
        _write_job(folder, record)
        if progress.log_tail:
            with (folder / "stdout.log").open("a", encoding="utf-8") as output:
                output.write(progress.log_tail[-MAX_LOG_CHARS:])
    return {"cancel": False}


def validate_bundle(archive: Path, destination: Path):
    """Reject zip-slip, links, duplicates, bombs and non-allowlisted files."""
    if archive.stat().st_size > MAX_BUNDLE:
        raise ValueError("Paquete demasiado grande")
    total = 0
    records = []
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if not 1 <= len(infos) <= MAX_FILES:
            raise ValueError("Número de archivos no permitido")
        names = set()
        for info in infos:
            name = info.filename
            pure = PurePosixPath(name)
            if (info.is_dir() or name in names or pure.is_absolute() or
                ".." in pure.parts or "\\" in name or
                not (name in ("report.json", "frozen_manifest.json", "stdout.log")
                     or CHECKPOINT.fullmatch(name))):
                raise ValueError("Ruta de archivo prohibida")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Enlace simbólico prohibido")
            names.add(name)
            total += info.file_size
            if total > MAX_EXTRACTED or info.file_size > 40 * 1024 * 1024:
                raise ValueError("Paquete descomprimido demasiado grande")
            if info.compress_size > MAX_BUNDLE:
                raise ValueError("ZIP corrupto")
            records.append((name, info))
        if "report.json" not in names or "frozen_manifest.json" not in names:
            raise ValueError("Falta informe o manifiesto")
        # Validate all content before touching the final output directory.
        parsed = json.loads(z.read("report.json"))
        manifest = json.loads(z.read("frozen_manifest.json"))
        if not isinstance(parsed, dict) or parsed.get("phase") != "development":
            raise ValueError("Solo se aceptan informes de desarrollo F9")
        if not isinstance(manifest, dict) or manifest.get("stage") not in (
                "development_complete", "f9b_baseline_development"):
            raise ValueError("Manifiesto F9 inválido")
        for model in manifest.get("models", {}).values():
            if not isinstance(model, dict):
                raise ValueError("Metadatos de checkpoint inválidos")
            checkpoint = "checkpoints/" + str(model.get("model_path", ""))
            expected = str(model.get("model_sha256", ""))
            if not CHECKPOINT.fullmatch(checkpoint) or checkpoint not in names:
                raise ValueError("Falta un checkpoint declarado")
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Checksum de checkpoint inválido")
            digest = hashlib.sha256()
            with z.open(checkpoint) as payload:
                while chunk := payload.read(1 << 20):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("Checksum de checkpoint no coincide")
        destination.mkdir(exist_ok=True)
        for name, info in records:
            target = destination.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(exist_ok=True)
            # Stream to disk; no arbitrary extraction.
            with z.open(info) as source, target.open("wb") as sink:
                copied = 0
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > info.file_size:
                        raise ValueError("Archivo excede tamaño declarado")
                    sink.write(chunk)
    return parsed, manifest


@router.post("/jobs/{job_id}/finish", dependencies=[Depends(authenticated_worker)])
async def finish(job_id: str, request: Request,
                 x_lab_claim_id: str | None = Header(default=None),
                 x_lab_exit_code: int = Header(default=0)):
    sv = _server()
    folder = sv.find_job(job_id)
    with sv.START_LOCK:
        if (folder / "cancel.request").exists():
            raise HTTPException(409, "Ejecución cancelada")
        _job_under_claim(folder, x_lab_claim_id or "")
    if x_lab_exit_code != 0:
        with sv.START_LOCK:
            job = _job_under_claim(folder, x_lab_claim_id or "")
            job.update(status="failed", exit_code=x_lab_exit_code, finished_at=sv.now())
            _write_job(folder, job)
        return {"status": "failed"}
    upload = folder / ("remote-upload-" + secrets.token_hex(8) + ".part")
    copied = 0
    try:
        with upload.open("wb") as sink:
            async for chunk in request.stream():
                copied += len(chunk)
                if copied > MAX_BUNDLE:
                    raise HTTPException(413, "Paquete demasiado grande")
                sink.write(chunk)
        if copied == 0:
            raise HTTPException(422, "Falta paquete de resultados")
        with sv.START_LOCK:
            job = _job_under_claim(folder, x_lab_claim_id or "")
            staging = folder / "remote-validated"
            if staging.exists():
                shutil.rmtree(staging)
            try:
                parsed, manifest = validate_bundle(upload, staging)
            except (ValueError, OSError, zipfile.BadZipFile, KeyError, UnicodeError) as exc:
                shutil.rmtree(staging, ignore_errors=True)
                raise HTTPException(422, "Resultados inválidos: " + type(exc).__name__) from exc
            expected_task = {"f9_memory": "memory", "f9_doorkey": "doorkey",
                             "f9_dynamic": "dynamic",
                             "f9b_memory_baselines": "memory"}[job["experiment"]]
            expected_stage = ("f9b_baseline_development"
                              if job["experiment"] == "f9b_memory_baselines"
                              else "development_complete")
            if (parsed.get("task") != expected_task or
                    manifest.get("stage") != expected_stage or
                    (job["experiment"] == "f9b_memory_baselines" and
                     parsed.get("stage") != "baselines")):
                shutil.rmtree(staging, ignore_errors=True)
                raise HTTPException(422, "Informe o manifiesto de otro protocolo")
            # stdout is also the live log visible in the web UI.
            uploaded_log = folder / "remote-validated" / "stdout.log"
            if uploaded_log.is_file():
                with uploaded_log.open("rb") as source, (folder / "stdout.log").open("wb") as target:
                    while chunk := source.read(1 << 16):
                        target.write(chunk)
                uploaded_log.unlink()
            (folder / "remote-validated").replace(folder / "output")
            job.update(status="completed", finished_at=sv.now(), exit_code=0,
                       claim_id=None)
            _write_job(folder, job)
        return {"status": "completed"}
    finally:
        upload.unlink(missing_ok=True)


def worker_status():
    from lab_portal.server import workspace
    path = workspace() / "f9_worker_heartbeat.json"
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
        online = time.time() - float(item["last_seen_epoch"]) < 45
        return {"online": online, "worker_id": item["worker_id"] if online else None,
                "last_seen": item.get("last_seen")}
    except (OSError, ValueError, KeyError, TypeError):
        return {"online": False, "worker_id": None, "last_seen": None}
