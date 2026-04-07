"""Allow ``python -m universal_memory`` to start the service."""

import uvicorn

from .config import load_config
from .main import app  # noqa: F401 — imported so uvicorn can resolve the reference

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "universal_memory.main:app",
        host=cfg.service.host,
        port=cfg.service.port,
        reload=False,
    )
