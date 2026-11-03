"""Run the demo: python -m demo (DEMO_HOST and DEMO_PORT, default 127.0.0.1:8400)."""
import logging
import os

import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "demo.app:create_app",
        factory=True,
        host=os.environ.get("DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("DEMO_PORT", "8400")),
        # No access log: the demo keeps no record of who sent what.
        access_log=False,
    )
