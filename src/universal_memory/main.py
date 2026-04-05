"""Universal Memory Service — main entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="Universal Memory Service",
    description="Hybrid memory search and write for AI agents",
    version="0.1.0",
)


@app.get("/api/v1/status")
async def status():
    return {"status": "starting", "version": "0.1.0"}
