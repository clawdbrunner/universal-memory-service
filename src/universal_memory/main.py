"""Universal Memory Service — main entry point."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config import get_config
from .db import init_db
from .indexer import Indexer
from .retrieval.embeddings import EmbeddingService
from .retrieval.pipeline import RetrievalPipeline
from .retrieval.vector_store import VectorStore
from .sync.sync_engine import SyncEngine
from .watcher import FileWatcher
from .writers.file_writer import FileWriter
from .writers.graphiti_writer import GraphitiWriter

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle."""
    cfg = get_config()

    # --- Logging ---
    log_level = getattr(logging, cfg.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    if cfg.logging.file:
        log_path = Path(cfg.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(log_level)
        logging.getLogger().addHandler(file_handler)

    # --- Database ---
    await init_db()

    # --- Services ---
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    pipeline = RetrievalPipeline(vector_store=vector_store, embeddings=embedding_service)
    file_writer = FileWriter()
    graphiti_writer = GraphitiWriter()
    indexer = Indexer(embedding_service, vector_store)
    sync_engine = SyncEngine()

    # --- File watcher callback ---
    async def _on_change(file_path: str, change_type: str) -> None:
        if change_type == "deleted":
            await indexer.remove_file(file_path)
        else:
            await indexer.index_file(file_path)
            await sync_engine.sync_file(file_path)

    watcher = FileWatcher(on_change=_on_change)

    # --- Attach to app state ---
    application.state.config = cfg
    application.state.pipeline = pipeline
    application.state.file_writer = file_writer
    application.state.graphiti_writer = graphiti_writer
    application.state.indexer = indexer
    application.state.sync_engine = sync_engine
    application.state.watcher = watcher
    application.state.start_time = time.time()

    # --- Start watcher & initial sync ---
    await watcher.start()
    await sync_engine.sync_all()

    logger.info(
        "Universal Memory Service started on %s:%s",
        cfg.service.host,
        cfg.service.port,
    )

    yield

    # --- Shutdown ---
    await watcher.stop()
    logger.info("Universal Memory Service stopped")


app = FastAPI(
    title="Universal Memory Service",
    description="Hybrid memory search and write for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
