from typing import Dict
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import ChatMessage, ChatSession, Repository, get_db
from pipeline.rag import RAGPipeline

router = APIRouter()
rag    = RAGPipeline()
NL     = "\n"
ESC_NL = "\\n"


async def _repo_meta(repo_id: str, db: AsyncSession) -> Dict:
    r = await db.execute(select(Repository).where(Repository.id == repo_id))
    repo = r.scalar_one_or_none()
    if not repo:
        raise HTTPException(404, "Repository not found")
    if repo.status != "ready":
        raise HTTPException(400, "Repository not indexed yet. Wait for status = ready.")
    return {"name": repo.name, "language": repo.language or "", "tech_stack": repo.tech_stack or []}


@router.get("/{repo_id}/sessions")
async def list_sessions(repo_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(
        select(ChatSession).where(ChatSession.repo_id == repo_id).order_by(ChatSession.updated_at.desc()))
    return [{"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat()} for s in r.scalars()]


@router.post("/{repo_id}/sessions")
async def create_session(repo_id: str, db: AsyncSession = Depends(get_db)):
    s = ChatSession(repo_id=repo_id)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return {"id": s.id, "title": s.title}


@router.delete("/{repo_id}/sessions/{session_id}", status_code=204)
async def delete_session(repo_id: str, session_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    s = r.scalar_one_or_none()
    if s:
        await db.delete(s)
        await db.commit()


@router.get("/{repo_id}/sessions/{session_id}/messages")
async def get_messages(repo_id: str, session_id: str, db: AsyncSession = Depends(get_db)):
    r = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at))
    return [{"id": m.id, "role": m.role, "content": m.content,
             "sources": m.sources or [], "created_at": m.created_at.isoformat()} for m in r.scalars()]


@router.post("/{repo_id}/sessions/{session_id}/messages")
async def send_message(repo_id: str, session_id: str, payload: Dict,
                       db: AsyncSession = Depends(get_db)):
    message = (payload.get("message") or "").strip()
    stream  = payload.get("stream", False)
    if not message:
        raise HTTPException(400, "message is required")

    meta      = await _repo_meta(repo_id, db)
    namespace = f"repo_{repo_id}"

    hr = await db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at).limit(16))
    history = [{"role": m.role, "content": m.content} for m in hr.scalars()]

    db.add(ChatMessage(session_id=session_id, role="user", content=message))
    await db.commit()

    sr = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    session = sr.scalar_one_or_none()
    if session and session.title == "New conversation":
        session.title = message[:60]
        await db.commit()

    if stream:
        async def event_gen():
            full = ""
            try:
                async for chunk in rag.stream_answer(message, namespace, meta, history):
                    full += chunk
                    safe = chunk.replace(NL, ESC_NL)
                    yield "data: " + safe + "\n\n"
            except Exception as e:
                yield "data: [ERROR] " + str(e) + "\n\n"
            finally:
                db.add(ChatMessage(session_id=session_id, role="assistant", content=full, sources=[]))
                await db.commit()
                yield "data: [DONE]\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    result = await rag.answer(message, namespace, meta, history)
    ai_msg = ChatMessage(session_id=session_id, role="assistant",
                         content=result["answer"], sources=result["sources"])
    db.add(ai_msg)
    await db.commit()
    await db.refresh(ai_msg)
    return {"id": ai_msg.id, "role": "assistant", "content": result["answer"],
            "sources": result["sources"], "session_id": session_id}
