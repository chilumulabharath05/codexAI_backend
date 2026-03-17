import traceback
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import select, update

from core.database import Analysis, AsyncSessionLocal, Repository, RepoFile
from pipeline.chunking import CodeChunker
from pipeline.embedding import EmbeddingService
from pipeline.ingestion import IngestionPipeline
from pipeline.rag import RAGPipeline

_chunker  = CodeChunker()
_embedder = EmbeddingService()
_rag      = RAGPipeline()

SUMMARY_EXTS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs",
                ".java", ".kt", ".rb", ".cs", ".php"}
DEP_FILES    = {"requirements.txt", "Pipfile", "pyproject.toml", "package.json",
                "go.mod", "Cargo.toml", "pom.xml", "Gemfile"}


async def _set_status(repo_id: str, status: str, error: Optional[str] = None):
    async with AsyncSessionLocal() as db:
        vals: Dict = {"status": status, "updated_at": datetime.utcnow()}
        if error:
            vals["error_message"] = error[:500]
        await db.execute(update(Repository).where(Repository.id == repo_id).values(**vals))
        await db.commit()


async def process_repository(repo_id: str, source: str, params: Dict):
    ingestion = IngestionPipeline(repo_id)
    print(f"[CodexAI] Starting pipeline for {repo_id}, source={source}", flush=True)

    try:
        # 1. Acquire
        await _set_status(repo_id, "cloning")
        if source == "git":
            repo_dir = await ingestion.clone_git(params["url"], params.get("branch", "main"))
        elif source == "zip":
            repo_dir = await ingestion.extract_zip(params["zip_bytes"])
        else:
            raise ValueError(f"Unknown source: {source}")
        print(f"[CodexAI] Acquired repo at {repo_dir}", flush=True)

        # 2. Walk + persist
        await _set_status(repo_id, "parsing")
        all_files: List[Dict] = []
        dep_contents: List[str] = []

        async with AsyncSessionLocal() as db:
            async for fm in ingestion.walk_repository(repo_dir):
                db.add(RepoFile(
                    repo_id    = repo_id,
                    path       = fm["path"],
                    name       = fm["name"],
                    extension  = fm["extension"],
                    language   = fm["language"],
                    size_bytes = fm["size_bytes"],
                    line_count = fm["line_count"],
                    content    = fm["content"] if not fm["is_binary"] else None,
                    is_binary  = fm["is_binary"],
                ))
                all_files.append(fm)
                if fm["name"] in DEP_FILES and fm["content"]:
                    dep_contents.append(f"=== {fm['path']} ===\n{fm['content'][:2000]}")

            lang_stats   = ingestion.language_stats(all_files)
            primary_lang = max(lang_stats, key=lang_stats.get) if lang_stats else None
            file_tree    = ingestion.build_file_tree([f["path"] for f in all_files])
            namespace    = f"repo_{repo_id}"

            await db.execute(update(Repository).where(Repository.id == repo_id).values(
                languages=lang_stats, language=primary_lang,
                total_files=len(all_files),
                total_lines=sum(f["line_count"] for f in all_files),
                file_tree=file_tree, vector_namespace=namespace,
                updated_at=datetime.utcnow(),
            ))
            await db.commit()

        print(f"[CodexAI] Parsed {len(all_files)} files, lang={primary_lang}", flush=True)

        # 3. AI summaries
        key_files: List[str] = []
        for fm in all_files:
            if (fm.get("extension", "") in SUMMARY_EXTS
                    and fm["content"] and fm["line_count"] > 5
                    and len(key_files) < 8):
                try:
                    summary = await _rag.summarize_file(
                        fm["path"], fm["language"] or "text",
                        fm["content"], fm["line_count"],
                    )
                    async with AsyncSessionLocal() as db:
                        await db.execute(update(RepoFile)
                            .where(RepoFile.repo_id == repo_id, RepoFile.path == fm["path"])
                            .values(summary=summary))
                        await db.commit()
                    key_files.append(f"=== {fm['path']} ===\n{fm['content'][:1000]}")
                    print(f"[CodexAI] Summarized {fm['path']}", flush=True)
                except Exception as e:
                    print(f"[CodexAI] Summary warning {fm['path']}: {e}", flush=True)

        # 4. Architecture
        try:
            async with AsyncSessionLocal() as db:
                r    = await db.execute(select(Repository).where(Repository.id == repo_id))
                repo = r.scalar_one()
                arch = await _rag.analyze_architecture(
                    repo_meta={
                        "name": repo.name, "language": primary_lang or "",
                        "tech_stack": _detect_stack(all_files),
                        "total_files": len(all_files),
                        "total_lines": sum(f["line_count"] for f in all_files),
                        "file_tree": file_tree,
                    },
                    key_files="\n\n".join(key_files[:6]),
                )
                await db.execute(update(Repository).where(Repository.id == repo_id).values(architecture=arch))
                db.add(Analysis(repo_id=repo_id, type="architecture", title="Architecture Analysis", content=arch))
                await db.commit()
            print("[CodexAI] Architecture analysis done", flush=True)
        except Exception as e:
            print(f"[CodexAI] Architecture warning: {e}", flush=True)

        # 5. Dependencies
        if dep_contents:
            try:
                dep = await _rag.analyze_dependencies("\n\n".join(dep_contents))
                async with AsyncSessionLocal() as db:
                    db.add(Analysis(repo_id=repo_id, type="dependency", title="Dependency Analysis", content=dep))
                    await db.commit()
            except Exception as e:
                print(f"[CodexAI] Dependency warning: {e}", flush=True)

        # 6. Embed
        await _set_status(repo_id, "embedding")
        all_chunks = []
        for fm in all_files:
            if fm["content"] and not fm["is_binary"]:
                all_chunks.extend(_chunker.chunk_file(
                    repo_id, fm["path"], fm["content"], fm["language"]))
        if all_chunks:
            count = await _embedder.index_chunks(all_chunks, f"repo_{repo_id}")
            print(f"[CodexAI] Indexed {count} chunks", flush=True)

        # 7. Ready
        async with AsyncSessionLocal() as db:
            await db.execute(update(Repository).where(Repository.id == repo_id).values(
                status="ready", analyzed_at=datetime.utcnow(),
                updated_at=datetime.utcnow(), tech_stack=_detect_stack(all_files),
            ))
            await db.commit()
        print(f"[CodexAI] READY - {repo_id}", flush=True)

    except Exception as exc:
        print(f"[CodexAI] FAILED - {repo_id}: {exc}", flush=True)
        traceback.print_exc()
        await _set_status(repo_id, "failed", str(exc))
    finally:
        ingestion.cleanup()


def _detect_stack(files: list) -> list:
    names = {f["name"].lower() for f in files}
    exts  = {f["extension"] for f in files}
    stack = []
    if "package.json"         in names: stack.append("Node.js")
    if any("next.config" in n for n in names): stack.append("Next.js")
    if ".tsx" in exts or ".jsx" in exts: stack.append("React")
    if "tailwind.config.js"   in names: stack.append("Tailwind CSS")
    if "requirements.txt"     in names or "pyproject.toml" in names: stack.append("Python")
    if any("fastapi" in (f.get("content") or "").lower() for f in files[:30]): stack.append("FastAPI")
    if any("django"  in (f.get("content") or "").lower() for f in files[:30]): stack.append("Django")
    if "go.mod"               in names: stack.append("Go")
    if "cargo.toml"           in names: stack.append("Rust")
    if "pom.xml"              in names or "build.gradle" in names: stack.append("Java")
    if any("docker"           in n for n in names): stack.append("Docker")
    return list(dict.fromkeys(stack))[:8]
