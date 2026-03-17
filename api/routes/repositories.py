from datetime import datetime
from typing import Dict
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.config import settings
from core.database import Repository, RepoFile, get_db
from pipeline.embedding import EmbeddingService
from services.processor import process_repository

router = APIRouter()


def _repo_dict(r: Repository) -> Dict:
    return {
        "id": r.id, "name": r.name, "source": r.source, "url": r.url,
        "branch": r.branch, "status": r.status, "error": r.error_message,
        "language": r.language, "languages": r.languages or {},
        "tech_stack": r.tech_stack or [], "total_files": r.total_files,
        "total_lines": r.total_lines, "architecture": r.architecture,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "analyzed_at": r.analyzed_at.isoformat() if r.analyzed_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


async def _get_or_404(repo_id: str, db: AsyncSession) -> Repository:
    result = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = result.scalar_one_or_none()
    if not repo:
        raise HTTPException(404, "Repository not found")
    return repo


@router.get("/")
async def list_repos(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Repository).order_by(Repository.created_at.desc()))
    return [_repo_dict(r) for r in result.scalars().all()]


@router.get("/{repo_id}")
async def get_repo(repo_id: str, db: AsyncSession = Depends(get_db)):
    return _repo_dict(await _get_or_404(repo_id, db))


@router.get("/{repo_id}/status")
async def get_status(repo_id: str, db: AsyncSession = Depends(get_db)):
    repo = await _get_or_404(repo_id, db)
    return {
        "status": repo.status, "error": repo.error_message,
        "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
    }


@router.post("/import/git", status_code=status.HTTP_202_ACCEPTED)
async def import_git(payload: Dict, bg: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    url = (payload.get("url") or "").strip()
    if not url:
        raise HTTPException(400, "url is required")
    branch = payload.get("branch", "main") or "main"
    name   = payload.get("name") or url.rstrip("/").split("/")[-1].replace(".git", "")
    source = ("github" if "github.com" in url else
              "gitlab" if "gitlab.com" in url else
              "bitbucket" if "bitbucket.org" in url else "url")
    repo = Repository(name=name, source=source, url=url, branch=branch)
    db.add(repo)
    await db.commit()
    await db.refresh(repo)
    bg.add_task(process_repository, repo.id, "git", {"url": url, "branch": branch})
    return {"repo_id": repo.id, "status": "queued"}


@router.post("/import/zip", status_code=status.HTTP_202_ACCEPTED)
async def import_zip(bg: BackgroundTasks, file: UploadFile = File(...),
                     name: str = Form(...), db: AsyncSession = Depends(get_db)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(400, "Only .zip files accepted")
    zip_bytes = await file.read()
    if len(zip_bytes) > settings.MAX_REPO_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"ZIP exceeds {settings.MAX_REPO_SIZE_MB} MB")
    repo = Repository(name=name or file.filename.replace(".zip", ""), source="zip")
    db.add(repo)
    await db.commit()
    await db.refresh(repo)
    bg.add_task(process_repository, repo.id, "zip", {"zip_bytes": zip_bytes})
    return {"repo_id": repo.id, "status": "queued"}


@router.get("/{repo_id}/files")
async def get_file_tree(repo_id: str, db: AsyncSession = Depends(get_db)):
    repo = await _get_or_404(repo_id, db)
    return {"file_tree": repo.file_tree, "total_files": repo.total_files}


@router.get("/{repo_id}/files/{file_path:path}")
async def get_file(repo_id: str, file_path: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(RepoFile).where(RepoFile.repo_id == repo_id, RepoFile.path == file_path))
    f = result.scalar_one_or_none()
    if not f:
        raise HTTPException(404, "File not found")
    return {
        "path": f.path, "name": f.name, "language": f.language,
        "line_count": f.line_count, "content": f.content, "summary": f.summary,
        "functions": f.functions or [], "classes": f.classes or [], "imports": f.imports or [],
    }


@router.delete("/{repo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repo(repo_id: str, db: AsyncSession = Depends(get_db)):
    repo = await _get_or_404(repo_id, db)
    EmbeddingService().delete_namespace(f"repo_{repo_id}")
    await db.delete(repo)
    await db.commit()
