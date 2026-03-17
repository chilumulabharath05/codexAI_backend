from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db
from core.config import settings

router = APIRouter()


@router.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    gemini_ok = bool(settings.GEMINI_API_KEY)

    return {
        "status":   "ok" if db_ok else "degraded",
        "database": "ok" if db_ok else "error",
        "gemini":   "configured" if gemini_ok else "missing - add GEMINI_API_KEY to .env",
        "version":  "2.0.0",
    }
