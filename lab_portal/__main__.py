"""Run locally: LAB_API_TOKEN='long-random-secret' python -m lab_portal."""
from __future__ import annotations

import os

import uvicorn


if __name__ == "__main__":
    if (len(os.environ.get("LAB_API_TOKEN", "")) < 12 or
            os.environ.get("LAB_API_TOKEN", "").startswith("REEMPLAZAR_")):
        raise SystemExit("Define LAB_API_TOKEN con al menos 12 caracteres antes de iniciar el portal")
    host = os.getenv("LAB_HOST", "127.0.0.1")
    port = int(os.getenv("LAB_PORT", "8765"))
    print(f"INFINITO Laboratory: http://{host}:{port}")
    uvicorn.run("lab_portal.server:app", host=host, port=port,
                workers=1, log_level="info", proxy_headers=False)
