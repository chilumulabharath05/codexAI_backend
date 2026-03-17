from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from core.config import settings
from core.database import init_db
from api.routes import health, repositories, analysis, chat, search


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="CodexAI API",
    description="AI-powered codebase analysis using Google Gemini (FREE)",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router,       prefix="",          tags=["health"])
app.include_router(repositories.router, prefix="/repos",    tags=["repositories"])
app.include_router(analysis.router,     prefix="/analysis", tags=["analysis"])
app.include_router(chat.router,         prefix="/chat",     tags=["chat"])
app.include_router(search.router,       prefix="/search",   tags=["search"])
