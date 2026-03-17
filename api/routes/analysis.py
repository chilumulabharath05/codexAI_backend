from typing import Dict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import settings
from core.database import Analysis, RepoFile, get_db
from pipeline.rag import RAGPipeline

router = APIRouter()
rag    = RAGPipeline()


def _check_key():
    if not settings.GEMINI_API_KEY:
        raise HTTPException(400, "GEMINI_API_KEY not set. Add it to your .env file and restart the server.")


@router.get("/{repo_id}")
async def list_analyses(repo_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Analysis).where(Analysis.repo_id == repo_id).order_by(Analysis.created_at.desc()))
    return [{"id": a.id, "type": a.type, "title": a.title, "content": a.content,
             "target_path": a.target_path, "created_at": a.created_at.isoformat()}
            for a in result.scalars().all()]


@router.post("/{repo_id}/file")
async def analyze_file(repo_id: str, payload: Dict, db: AsyncSession = Depends(get_db)):
    _check_key()
    path = (payload.get("path") or "").strip()
    if not path:
        raise HTTPException(400, "path is required")

    cached = await db.execute(
        select(Analysis).where(Analysis.repo_id == repo_id,
                                Analysis.type == "file", Analysis.target_path == path))
    hit = cached.scalar_one_or_none()
    if hit:
        return {"content": hit.content, "cached": True}

    fr = await db.execute(select(RepoFile).where(RepoFile.repo_id == repo_id, RepoFile.path == path))
    f  = fr.scalar_one_or_none()
    if not f:
        raise HTTPException(404, f"File not found: {path}")
    if not f.content:
        raise HTTPException(400, "File has no content (binary or too large)")

    try:
        content = await rag.summarize_file(f.path, f.language or "text", f.content, f.line_count)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    db.add(Analysis(repo_id=repo_id, type="file", target_path=path, title=f"File: {f.name}", content=content))
    f.summary = content
    await db.commit()
    return {"content": content, "cached": False}


@router.post("/{repo_id}/function")
async def analyze_function(repo_id: str, payload: Dict, db: AsyncSession = Depends(get_db)):
    _check_key()
    file_path = payload.get("file_path", "")
    fname     = payload.get("function_name", "")
    code      = payload.get("function_code", "")
    ctx       = payload.get("context", "")
    if not file_path or not fname:
        raise HTTPException(400, "file_path and function_name are required")
    fr   = await db.execute(select(RepoFile).where(RepoFile.repo_id == repo_id, RepoFile.path == file_path))
    f    = fr.scalar_one_or_none()
    lang = f.language if f else "text"
    try:
        content = await rag.explain_function(file_path, fname, lang or "text", code, ctx)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")
    return {"content": content, "function": fname}


@router.post("/{repo_id}/security")
async def security_scan(repo_id: str, payload: Dict, db: AsyncSession = Depends(get_db)):
    _check_key()
    path = (payload.get("path") or "").strip()
    if not path:
        raise HTTPException(400, "path is required")
    fr = await db.execute(select(RepoFile).where(RepoFile.repo_id == repo_id, RepoFile.path == path))
    f  = fr.scalar_one_or_none()
    if not f or not f.content:
        raise HTTPException(404, "File not found or no content")
    try:
        content = await rag.audit_security(path, f.language or "text", f.content)
    except Exception as e:
        raise HTTPException(500, f"Security scan failed: {str(e)}")
    db.add(Analysis(repo_id=repo_id, type="security", target_path=path, title=f"Security: {f.name}", content=content))
    await db.commit()
    return {"content": content}
