"""Windows-compatible F9 compute agent. Pull-only; never accepts shell commands.

This process runs on the user's PC. It contacts a trusted HTTPS laboratory,
claims *allowlisted* F9 development experiments, checks the exact commit SHA,
executes the local reviewed F9 module and uploads validated results.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

from lab_portal.catalog import EXPERIMENTS, command, normalize

ROOT = Path(__file__).resolve().parents[1]
TASKS = frozenset(("f9_memory", "f9_doorkey", "f9_dynamic"))
MAX_UPLOAD = 64 * 1024 * 1024


def current_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        text=True, stderr=subprocess.DEVNULL, timeout=5).strip()


def server_url(raw: str, allow_http_local: bool = False) -> str:
    parsed = urllib.parse.urlsplit(raw.strip())
    local = parsed.hostname in ("127.0.0.1", "localhost")
    if (parsed.scheme != "https" and not (allow_http_local and
                                             parsed.scheme == "http" and local)):
        raise ValueError("Se requiere HTTPS; HTTP permitido solo con --test-local en localhost")
    if (not parsed.hostname or parsed.username or parsed.password or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("Usa únicamente la URL raíz del laboratorio, sin rutas ni credenciales")
    return raw.rstrip("/")


class LabClient:
    def __init__(self, url: str, token: str, *, allow_http_local=False):
        self.base = server_url(url, allow_http_local)
        if len(token) < 32:
            raise ValueError("Token del trabajador ausente o demasiado corto")
        self.token = token

    def call(self, endpoint: str, payload=None, *, raw=False, extra_headers=None):
        if not endpoint.startswith("/api/worker/"):
            raise ValueError("Ruta de API no permitida")
        data = (payload if raw else
                json.dumps(payload or {}, separators=(",", ":")).encode("utf-8"))
        headers = {"X-Lab-Worker-Token": self.token,
                   "Content-Type": "application/zip" if raw else "application/json",
                   "User-Agent": "INFINITO-F9-WindowsWorker/1"}
        if extra_headers:
            headers.update(extra_headers)
        request = urllib.request.Request(self.base + endpoint, data=data,
                                         headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=75 if raw else 20) as answer:
            return json.loads(answer.read())


def _output_zip(folder: Path) -> bytes:
    stream = io.BytesIO()
    includes = []
    for basename in ("report.json", "frozen_manifest.json"):
        file = folder / basename
        if not file.is_file():
            raise ValueError("No existe " + basename + "; no se puede informar éxito")
        includes.append(file)
    for checkpoint in (folder / "checkpoints").glob("*.zip"):
        if checkpoint.is_file() and not checkpoint.is_symlink():
            includes.append(checkpoint)
    stdout = folder / "stdout.log"
    if stdout.is_file():
        includes.append(stdout)
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for file in includes:
            relative = file.relative_to(folder).as_posix()
            if file.stat().st_size > 40 * 1024 * 1024:
                raise ValueError("Archivo individual demasiado grande")
            z.write(file, relative)
            if stream.tell() > MAX_UPLOAD:
                raise ValueError("Resultados exceden límite de subida de 64 MiB")
    content = stream.getvalue()
    if len(content) > MAX_UPLOAD:
        raise ValueError("Resultados demasiado grandes")
    return content


def _kill_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, timeout=12, check=False)
    else:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def execute_job(client: LabClient, claimed: dict, *, python: str,
                max_seconds: int, runs_root: Path) -> dict:
    job_id = claimed["id"]
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("ID de ejecución inválido")
    exp = claimed["experiment"]
    if exp not in TASKS:
        raise ValueError("El servidor no puede ejecutar programas ajenos al catálogo F9")
    if claimed["revision"] != current_revision():
        raise ValueError("El código local difiere de la revisión del servidor")
    params = normalize(exp, claimed["params"])
    # Independently reconstruct the exact allowlisted command, not server argv.
    output = runs_root / job_id / "output"
    output.mkdir(parents=True, exist_ok=True)
    local_log = output / "stdout.log"
    argv = [python, *command(exp, params, str(output))]
    print("EJECUTANDO", exp, job_id[:8], params, flush=True)
    started = time.monotonic()
    claim_token = claimed["claim_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", claim_token):
        raise ValueError("Lease inválido")
    with local_log.open("wb") as out:
        proc = subprocess.Popen(
            argv, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=out, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"},
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        last_log_pos = 0
        cancelled = False
        timed_out = False
        try:
            while proc.poll() is None:
                time.sleep(3)
                with local_log.open("rb") as src:
                    src.seek(last_log_pos)
                    fragment = src.read(1500)
                    last_log_pos = src.tell()
                response = client.call(
                    f"/api/worker/jobs/{job_id}/heartbeat",
                    {"claim_id": claim_token, "log_tail": fragment.decode("utf-8", "replace")})
                if response.get("cancel"):
                    cancelled = True
                    break
                if time.monotonic() - started > max_seconds:
                    timed_out = True
                    break
        finally:
            if proc.poll() is None:
                _kill_process_tree(proc)
            proc.wait()
    status = proc.returncode
    if cancelled or timed_out:
        status = 124 if timed_out else 130
    if status == 0:
        try:
            bundle = _output_zip(output)
            reply = client.call(
                f"/api/worker/jobs/{job_id}/finish", bundle, raw=True,
                extra_headers={"X-Lab-Claim-ID": claim_token,
                               "X-Lab-Exit-Code": "0"})
        except Exception:
            # Do not delete checkpoints. On upload failure the exact output
            # remains on this computer for recovery.
            print("No se pudo subir el resultado: se conserva en", output, flush=True)
            raise
    else:
        reply = client.call(
            f"/api/worker/jobs/{job_id}/finish", b"", raw=True,
            extra_headers={"X-Lab-Claim-ID": claim_token,
                           "X-Lab-Exit-Code": str(status)})
    print("RESULTADO", job_id[:8], reply, "archivos locales:", output, flush=True)
    return reply


def loop(server: str, *, once=False, interval=8, python=sys.executable,
         max_seconds=14_400, test_local=False, output_dir=None):
    token = os.environ.get("INFINITO_WORKER_TOKEN", "")
    client = LabClient(server, token, allow_http_local=test_local)
    revision = current_revision()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("No hay un commit de Git válido")
    worker_id = re.sub(r"[^A-Za-z0-9_.-]", "-", platform.node())[:45] or "windows"
    worker_id += "-" + uuid.uuid4().hex[:8]
    runs_root = (output_dir or (ROOT / "observatory_results" / "windows_worker")).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    print(f"INFINITO F9 worker: {worker_id} · revisión {revision[:12]}", flush=True)
    print("Conectando por HTTPS a", client.base, flush=True)
    while True:
        try:
            reply = client.call("/api/worker/ping",
                                {"worker_id": worker_id, "revision": revision})
            server_rev = reply.get("server_revision")
            if server_rev != revision:
                print("CÓDIGO DISTINTO: servidor", str(server_rev)[:12],
                      "local", revision[:12],
                      "→ actualiza Git en ambos lados antes de lanzar F9", flush=True)
            elif reply.get("jobs"):
                item = reply["jobs"][0]
                claimed = client.call(f"/api/worker/jobs/{item['id']}/claim",
                                      {"worker_id": worker_id, "revision": revision})
                execute_job(client, claimed, python=python, max_seconds=max_seconds,
                            runs_root=runs_root)
                if once:
                    break
            elif once:
                print("No hay ningún F9 pendiente en el servidor", flush=True)
                break
        except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError,
                subprocess.SubprocessError) as error:
            print("ERROR:", type(error).__name__, str(error)[:250], flush=True)
            if once:
                raise
        time.sleep(interval)


def main():
    p = argparse.ArgumentParser(description="INFINITO F9 Windows remote worker")
    p.add_argument("--server", default="https://lab.178-105-103-4.sslip.io")
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=8)
    p.add_argument("--max-seconds", type=int, default=14_400)
    p.add_argument("--test-local", action="store_true", help="Solo tests HTTP localhost")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if args.interval < 2 or not 60 <= args.max_seconds <= 86_400:
        p.error("interval >=2 y max-seconds 60..86400")
    try:
        loop(args.server, once=args.once, interval=args.interval,
             max_seconds=args.max_seconds, test_local=args.test_local,
             output_dir=args.output_dir)
    except KeyboardInterrupt:
        print("Worker detenido. Si hay una prueba en curso consulta su estado.",flush=True)


if __name__ == "__main__":
    main()
