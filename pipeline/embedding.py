"""
Embedding service using Google's free text-embedding-004 model.
No OpenAI key needed - only GEMINI_API_KEY required.
"""
import asyncio
from typing import Any, Dict, List, Optional

import chromadb
import google.generativeai as genai

from core.config import settings
from pipeline.chunking import CodeChunk

_chroma: Optional[chromadb.EphemeralClient] = None


def get_chroma() -> chromadb.EphemeralClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.EphemeralClient()
    return _chroma


class EmbeddingService:

    def _configure(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts or not settings.GEMINI_API_KEY:
            return []

        self._configure()
        all_vecs: List[List[float]] = []
        batch_size = 20  # Google embedding batch limit

        loop = asyncio.get_event_loop()

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            # Filter empty texts
            batch = [t for t in batch if t and t.strip()]
            if not batch:
                continue

            for attempt in range(3):
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda b=batch: genai.embed_content(
                            model=settings.EMBEDDING_MODEL,
                            content=b,
                            task_type="retrieval_document",
                        ),
                    )
                    vecs = result["embedding"]
                    # Google returns a single vector for single input,
                    # or list of vectors for batch
                    if isinstance(vecs[0], float):
                        all_vecs.append(vecs)
                    else:
                        all_vecs.extend(vecs)
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"[CodexAI] Embedding batch failed: {exc}", flush=True)
                        # Return zero vectors as fallback
                        for _ in batch:
                            all_vecs.append([0.0] * 768)
                    else:
                        await asyncio.sleep(2 ** attempt)

        return all_vecs

    async def embed_query(self, text: str) -> List[float]:
        if not text or not settings.GEMINI_API_KEY:
            return [0.0] * 768

        self._configure()
        loop = asyncio.get_event_loop()

        try:
            result = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model=settings.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_query",
                ),
            )
            vec = result["embedding"]
            return vec if isinstance(vec[0], float) else vec[0]
        except Exception as exc:
            print(f"[CodexAI] Query embedding failed: {exc}", flush=True)
            return [0.0] * 768

    async def index_chunks(self, chunks: List[CodeChunk], namespace: str) -> int:
        if not chunks or not settings.GEMINI_API_KEY:
            return 0

        # Filter valid chunks
        valid = [c for c in chunks if c.content and c.content.strip()]
        if not valid:
            return 0

        print(f"[CodexAI] Embedding {len(valid)} chunks with Google...", flush=True)

        texts   = [c.content for c in valid]
        vectors = await self.embed_texts(texts)

        if not vectors:
            print("[CodexAI] No vectors generated", flush=True)
            return 0

        # Ensure lengths match
        min_len = min(len(vectors), len(valid))
        vectors = vectors[:min_len]
        valid   = valid[:min_len]

        col = get_chroma().get_or_create_collection(
            name=namespace,
            metadata={"hnsw:space": "cosine"},
        )

        # Deduplicate
        seen: set = set()
        ids, vecs, metas, docs = [], [], [], []
        for i, chunk in enumerate(valid):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            ids.append(chunk.chunk_id)
            vecs.append(vectors[i])
            metas.append({
                "repo_id":    chunk.repo_id,
                "file_path":  chunk.file_path,
                "language":   chunk.language or "unknown",
                "start_line": chunk.start_line,
                "end_line":   chunk.end_line,
                "content":    chunk.content[:1200],
            })
            docs.append(chunk.content[:1200])

        # Upsert in small batches
        total = 0
        for i in range(0, len(ids), 50):
            b_ids  = ids[i:i + 50]
            b_vecs = vecs[i:i + 50]
            b_meta = metas[i:i + 50]
            b_docs = docs[i:i + 50]
            n = min(len(b_ids), len(b_vecs), len(b_meta), len(b_docs))
            if n == 0:
                continue
            try:
                col.upsert(
                    ids=b_ids[:n],
                    embeddings=b_vecs[:n],
                    metadatas=b_meta[:n],
                    documents=b_docs[:n],
                )
                total += n
            except Exception as e:
                print(f"[CodexAI] Upsert batch failed: {e}", flush=True)

        print(f"[CodexAI] Indexed {total} chunks", flush=True)
        return total

    async def query(self, query_text: str, namespace: str,
                    top_k: int = settings.TOP_K_RETRIEVAL,
                    language: Optional[str] = None) -> List[Dict[str, Any]]:
        if not settings.GEMINI_API_KEY:
            return []
        try:
            col = get_chroma().get_collection(name=namespace)
        except Exception:
            return []

        count = col.count()
        if count == 0:
            return []

        qvec  = await self.embed_query(query_text)
        where = {"language": {"$eq": language}} if language else None

        try:
            res = col.query(
                query_embeddings=[qvec],
                n_results=min(top_k, count),
                where=where,
            )
        except Exception as e:
            print(f"[CodexAI] Query failed: {e}", flush=True)
            return []

        return [
            {
                "id":       res["ids"][0][i],
                "score":    float(res["distances"][0][i]),
                "metadata": res["metadatas"][0][i],
                "content":  res["documents"][0][i],
            }
            for i in range(len(res["ids"][0]))
        ]

    def delete_namespace(self, namespace: str):
        try:
            get_chroma().delete_collection(namespace)
        except Exception:
            pass
