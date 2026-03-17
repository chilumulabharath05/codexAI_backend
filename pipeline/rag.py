"""
RAG Pipeline using Google Gemini (FREE).
Get your key at: https://aistudio.google.com/app/apikey
"""
import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import google.generativeai as genai

from core.config import settings
from pipeline.embedding import EmbeddingService

SYSTEM = (
    "You are an expert software architect and senior engineer. "
    "Explain codebases clearly with specific file names, function names, and line numbers. "
    "Be precise and use Markdown formatting."
)

ARCH_PROMPT = """Analyze this codebase and provide a comprehensive architecture overview.

Repository: {repo_name}
Primary language: {language}
Tech stack: {tech_stack}
Total files: {total_files} | Total lines: {total_lines}

File structure:
{file_tree}

Key files:
{key_files}

Provide:
## Architecture Pattern
What pattern is used? (MVC, microservices, layered, etc.)

## High-Level Design
Major components and how they interact.

## Layer Breakdown
- Entry points
- Business logic
- Data access
- External integrations

## Key Design Decisions
Notable choices and why they matter.

## Tech Stack Analysis
Each technology and its role.

## Improvement Suggestions
Top 3 actionable recommendations."""

FILE_PROMPT = """Analyze this source file.

File: {file_path}
Language: {language}
Lines: {line_count}

```{language}
{content}
```

Provide:
## Purpose
What does this file do?

## Key Components
Classes, functions, and constants.

## Dependencies
What this imports and what depends on it.

## Data Flow
How data enters and exits.

## Issues
Any bugs, security risks, or improvements."""

FUNC_PROMPT = """Explain this function in detail.

File: {file_path}
Function: {fname}
Language: {language}

```{language}
{code}
```

Context:
```{language}
{context}
```

Cover: Purpose, Parameters, Return Value, Logic, Edge Cases."""

SEC_PROMPT = """Perform a security audit of this code.

File: {file_path}
Language: {language}

```{language}
{content}
```

Check for: hardcoded secrets, injection risks, missing auth, data exposure, insecure patterns.
Rate each finding: Critical/High/Medium/Low.
If nothing found, say "No significant issues detected." """

DEP_PROMPT = """Analyze project dependencies.

{dep_content}

Produce:
## Direct Dependencies (name, version, purpose)
## Dev Dependencies
## Security Concerns
## Recommendations"""

CHAT_SYS = (
    "You are an AI assistant specialized in answering questions about a specific code repository. "
    "Repository: {repo_name} | Language: {language} | Stack: {tech_stack}. "
    "Answer ONLY from the provided code snippets. Always cite file paths and line numbers. "
    "If the answer is not in the snippets, say so clearly."
)

CHAT_USER = """Question: {question}

Retrieved code from the repository:

{context}

Answer based only on the code above."""


class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingService()

    def _configure(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)

    def _model(self, system: str = SYSTEM) -> genai.GenerativeModel:
        self._configure()
        return genai.GenerativeModel(
            model_name=settings.GEMINI_MODEL,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=3000,
            ),
        )

    async def _complete(self, prompt: str) -> str:
        if not settings.GEMINI_API_KEY:
            return "Gemini API key not configured. Add GEMINI_API_KEY to your .env file."
        loop   = asyncio.get_event_loop()
        model  = self._model()
        try:
            resp = await loop.run_in_executor(
                None, lambda: model.generate_content(prompt)
            )
            return resp.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}") from e

    # ── RAG Chat ─────────────────────────────────────────────────────────────

    async def answer(self, question: str, namespace: str,
                     repo_meta: Dict, history: List[Dict] = None) -> Dict[str, Any]:
        hits            = await self.embedder.query(question, namespace)
        context, sources = self._ctx(hits)

        sys_prompt = CHAT_SYS.format(
            repo_name  = repo_meta.get("name", ""),
            language   = repo_meta.get("language", ""),
            tech_stack = ", ".join(repo_meta.get("tech_stack", [])),
        )
        user_msg = CHAT_USER.format(question=question, context=context or "No relevant code found.")

        if not settings.GEMINI_API_KEY:
            return {"answer": "Gemini API key not set.", "sources": [], "tokens": 0}

        loop  = asyncio.get_event_loop()
        model = self._model(sys_prompt)

        # Build Gemini chat history
        chat_hist = []
        for msg in (history or [])[-8:]:
            role = "user" if msg["role"] == "user" else "model"
            chat_hist.append({"role": role, "parts": [msg["content"]]})

        try:
            chat = model.start_chat(history=chat_hist)
            resp = await loop.run_in_executor(None, lambda: chat.send_message(user_msg))
            answer_text = resp.text
        except Exception as e:
            answer_text = f"Error: {str(e)}"

        return {"answer": answer_text, "sources": sources, "tokens": 0}

    async def stream_answer(self, question: str, namespace: str,
                            repo_meta: Dict, history: List[Dict] = None) -> AsyncGenerator[str, None]:
        result = await self.answer(question, namespace, repo_meta, history)
        # Yield in chunks for a streaming effect
        text  = result["answer"]
        words = text.split(" ")
        chunk = ""
        for i, word in enumerate(words):
            chunk += word + " "
            if (i + 1) % 8 == 0:
                yield chunk
                chunk = ""
                await asyncio.sleep(0.02)
        if chunk:
            yield chunk

    # ── One-shot analysis ─────────────────────────────────────────────────────

    async def analyze_architecture(self, repo_meta: Dict, key_files: str) -> str:
        return await self._complete(ARCH_PROMPT.format(
            repo_name  = repo_meta.get("name", ""),
            language   = repo_meta.get("language", ""),
            tech_stack = ", ".join(repo_meta.get("tech_stack", [])),
            total_files= repo_meta.get("total_files", 0),
            total_lines= repo_meta.get("total_lines", 0),
            file_tree  = str(repo_meta.get("file_tree", ""))[:2000],
            key_files  = key_files[:5000],
        ))

    async def summarize_file(self, path: str, language: str,
                             content: str, lines: int) -> str:
        return await self._complete(FILE_PROMPT.format(
            file_path  = path,
            language   = language or "text",
            line_count = lines,
            content    = content[:4000],
        ))

    async def explain_function(self, path: str, fname: str, language: str,
                               code: str, context: str) -> str:
        return await self._complete(FUNC_PROMPT.format(
            file_path = path, fname = fname,
            language  = language or "text",
            code      = code[:2500], context = context[:1200],
        ))

    async def audit_security(self, path: str, language: str, content: str) -> str:
        return await self._complete(SEC_PROMPT.format(
            file_path = path, language = language or "text",
            content   = content[:4000],
        ))

    async def analyze_dependencies(self, dep_content: str) -> str:
        return await self._complete(DEP_PROMPT.format(dep_content=dep_content[:5000]))

    def _ctx(self, hits: List[Dict]):
        parts, sources = [], []
        for h in hits:
            m = h["metadata"]
            parts.append(
                "### " + m["file_path"] +
                " (lines " + str(m["start_line"]) + "-" + str(m["end_line"]) + ")\n" +
                "```" + m.get("language", "") + "\n" + h["content"] + "\n```"
            )
            sources.append({
                "file":       m["file_path"],
                "start_line": m["start_line"],
                "end_line":   m["end_line"],
                "score":      round(1 - h.get("score", 0), 3),
            })
        return "\n\n".join(parts), sources
