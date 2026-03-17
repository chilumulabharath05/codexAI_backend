from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import Repository, RepoFile, get_db
from pipeline.embedding import EmbeddingService
from pipeline.rag import RAGPipeline

router   = APIRouter()
embedder = EmbeddingService()
rag      = RAGPipeline()


@router.get("/{repo_id}")
async def semantic_search(
    repo_id: str,
    q:        str  = Query(..., min_length=2),
    top_k:    int  = Query(10, ge=1, le=30),
    language: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    r = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = r.scalar_one_or_none()
    if not repo:
        raise HTTPException(404, "Repository not found")
    if repo.status != "ready":
        raise HTTPException(400, "Repository not indexed yet")
    hits = await embedder.query(q, f"repo_{repo_id}", top_k=top_k, language=language)
    return {
        "query": q,
        "results": [
            {
                "file_path":  h["metadata"].get("file_path", ""),
                "start_line": h["metadata"].get("start_line", 0),
                "end_line":   h["metadata"].get("end_line", 0),
                "language":   h["metadata"].get("language", ""),
                "snippet":    h["content"][:500],
                "score":      round(1 - h.get("score", 0), 3),
            }
            for h in hits
        ],
    }


@router.get("/{repo_id}/readme")
async def generate_readme(repo_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = r.scalar_one_or_none()
    if not repo:
        raise HTTPException(404, "Repository not found")
    prompt = (
        "Generate a professional README.md for this repository.\n\n"
        "Name: " + repo.name + "\n"
        "Language: " + str(repo.language) + "\n"
        "Tech stack: " + ", ".join(repo.tech_stack or []) + "\n"
        "Files: " + str(repo.total_files) + " | Lines: " + str(repo.total_lines) + "\n\n"
        "Architecture:\n" + (repo.architecture or "Not available")[:2000] + "\n\n"
        "Generate a complete README with: title, badges, description, features, "
        "tech stack, installation, usage, contributing, license."
    )
    try:
        content = await rag._complete(prompt)
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")
    return {"content": content}


@router.get("/{repo_id}/apidocs")
async def generate_api_docs(repo_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import or_
    r = await db.execute(
        select(RepoFile).where(
            RepoFile.repo_id == repo_id,
            or_(RepoFile.path.ilike("%route%"), RepoFile.path.ilike("%controller%"),
                RepoFile.path.ilike("%handler%"), RepoFile.path.ilike("%endpoint%"))
        ).limit(12))
    files = r.scalars().all()
    combined = "\n\n".join(
        "### " + f.path + "\n```" + (f.language or "") + "\n" + (f.content or "")[:1500] + "\n```"
        for f in files
    ) if files else "No route files found."
    try:
        content = await rag._complete(
            "Generate API documentation in Markdown from these files:\n\n" + combined +
            "\n\nInclude: method, path, description, params, request body, response, errors."
        )
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")
    return {"content": content}
