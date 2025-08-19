# app/main.py
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (parent of app/)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ⬅️ NEW
from .config import settings
from .logging import setup_logging
from . import db
from .routes.message import router as message_router

setup_logging(settings.LOG_LEVEL)
app = FastAPI(title="Incentives Backend", version="0.1.0")

# 🔓 CORS: allow all origins, methods, headers (no credentials with "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,  # must be False when using "*"
)

@app.on_event("startup")
def _startup():
    db.init_pool()

@app.on_event("shutdown")
def _shutdown():
    db.close_pool()

app.include_router(message_router)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": settings.APP_ENV}

@app.get("/readyz")
def readyz():
    ok = db.db_ready()
    return {"db": "up" if ok else "down"}

@app.get("/version")
def version():
    return {"version": app.version, "model": settings.EMBED_MODEL}
