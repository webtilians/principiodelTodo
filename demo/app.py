"""HTTP layer of the INFINITO demo: the page, and a small JSON API for it."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .service import Demo, DemoError

STATIC = Path(__file__).resolve().parent / "static"


class SessionBody(BaseModel):
    session: str = ""


class ChatBody(BaseModel):
    session: str
    message: str


def client_ip(request: Request) -> str:
    # The service listens on 127.0.0.1 behind Caddy, which sets X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "unknown")


def create_app(demo: Optional[Demo] = None, *, sweep: bool = True) -> FastAPI:
    demo = demo or Demo()
    if sweep:
        demo.start_sweeper()
    app = FastAPI(title="INFINITO demo", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.demo = demo
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(DemoError)
    def refused(_: Request, error: DemoError) -> JSONResponse:
        return JSONResponse({"error": error.code}, status_code=error.status)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/healthz")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/status")
    def status() -> dict:
        return demo.status()

    @app.post("/api/session")
    def session() -> dict:
        created = demo.new_session()
        return {"session": created.id, "messages_left": demo.settings.messages_per_visitor}

    @app.post("/api/chat")
    def chat(body: ChatBody, request: Request) -> dict:
        return demo.chat(body.session, body.message, client_ip(request))

    @app.post("/api/forget-everything")
    def forget_everything(body: SessionBody) -> dict:
        demo.drop(body.session)
        return {"ok": True}

    return app
