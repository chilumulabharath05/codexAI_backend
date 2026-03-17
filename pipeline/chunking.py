import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from core.config import settings

CHARS_PER_TOKEN = 4


@dataclass
class CodeChunk:
    chunk_id:     str
    repo_id:      str
    file_path:    str
    language:     Optional[str]
    content:      str
    start_line:   int
    end_line:     int
    chunk_index:  int
    total_chunks: int = 0


class CodeChunker:
    def __init__(self):
        self.max_chars     = settings.CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
        self.overlap_chars = settings.CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN

    def chunk_file(self, repo_id: str, file_path: str,
                   content: str, language: Optional[str]) -> List[CodeChunk]:
        if not content or not content.strip():
            return []
        lines = content.split("\n")
        if language == "python":
            segs = self._split_python(lines)
        elif language in ("javascript", "typescript"):
            segs = self._split_js(lines)
        elif language in ("java", "kotlin", "csharp", "cpp", "c", "rust", "go"):
            segs = self._split_brace(lines)
        else:
            segs = self._sliding(lines)

        chunks = []
        for i, (s, e) in enumerate(segs):
            body = "\n".join(lines[s:e])
            if not body.strip():
                continue
            chunks.append(CodeChunk(
                chunk_id    = file_path + "::chunk_" + str(i),
                repo_id     = repo_id,
                file_path   = file_path,
                language    = language,
                content     = "# File: " + file_path + "\n" + body,
                start_line  = s + 1,
                end_line    = e,
                chunk_index = i,
            ))
        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    def _split_python(self, lines: List[str]) -> List[Tuple[int, int]]:
        bounds = [0]
        for i, line in enumerate(lines):
            s = line.lstrip()
            if s.startswith(("def ", "class ", "async def ")) and len(line) - len(s) == 0 and i > 0:
                bounds.append(i)
        bounds.append(len(lines))
        return self._merge(lines, bounds)

    def _split_js(self, lines: List[str]) -> List[Tuple[int, int]]:
        pat = re.compile(
            r"^(export\s+)?(default\s+)?(async\s+)?function\b|"
            r"^(export\s+)?class\b|"
            r"^(export\s+)?const\s+\w+\s*=\s*(async\s+)?\("
        )
        bounds = [0]
        for i, line in enumerate(lines):
            if pat.match(line.strip()) and i > 0:
                bounds.append(i)
        bounds.append(len(lines))
        return self._merge(lines, bounds)

    def _split_brace(self, lines: List[str]) -> List[Tuple[int, int]]:
        pat = re.compile(
            r"^(public|private|protected|static|func\s|fn\s|"
            r"class\b|struct\b|interface\b|impl\b)"
        )
        bounds = [0]
        for i, line in enumerate(lines):
            if pat.match(line.strip()) and i > 0:
                bounds.append(i)
        bounds.append(len(lines))
        return self._merge(lines, bounds)

    def _merge(self, lines: List[str], bounds: List[int]) -> List[Tuple[int, int]]:
        result: List[Tuple[int, int]] = []
        cur_start = bounds[0]
        cur_lines: List[str] = []
        for b_start, b_end in zip(bounds[:-1], bounds[1:]):
            seg = lines[b_start:b_end]
            combined = "\n".join(cur_lines) + "\n".join(seg)
            if len(combined) <= self.max_chars:
                cur_lines.extend(seg)
            else:
                if cur_lines:
                    result.append((cur_start, cur_start + len(cur_lines)))
                cur_start = b_start
                cur_lines = seg[:]
                if len("\n".join(seg)) > self.max_chars:
                    sub = self._sliding(seg)
                    result.extend([(b_start + s, b_start + e) for s, e in sub])
                    cur_lines = []
        if cur_lines:
            result.append((cur_start, cur_start + len(cur_lines)))
        return result if result else [(0, len(lines))]

    def _sliding(self, lines: List[str]) -> List[Tuple[int, int]]:
        if not lines:
            return []
        avg = max(1, sum(len(l) for l in lines) // max(1, len(lines)))
        lpc = max(1, self.max_chars // avg)
        olp = max(0, self.overlap_chars // avg)
        result, start = [], 0
        while start < len(lines):
            end = min(start + lpc, len(lines))
            result.append((start, end))
            if end == len(lines):
                break
            start = max(start + 1, end - olp)
        return result if result else [(0, len(lines))]
