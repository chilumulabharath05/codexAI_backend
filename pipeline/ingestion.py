import asyncio
import os
import shutil
import zipfile
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
import aiofiles
from core.config import settings

SKIP_DIRS = {
    ".git", ".svn", "__pycache__", ".pytest_cache", "node_modules",
    "venv", ".venv", "env", "dist", "build", ".next", ".nuxt",
    "coverage", "target", "vendor", ".idea", ".vscode",
}
SKIP_EXT = {
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".rar", ".lock", ".sum",
}
LANG_MAP: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".sh": "bash", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".xml": "xml",
    ".html": "html", ".css": "css", ".scss": "scss",
    ".md": "markdown", ".sql": "sql", ".tf": "terraform",
}


class IngestionPipeline:
    def __init__(self, repo_id: str):
        self.repo_id   = repo_id
        self.base_path = Path(f"/tmp/codexai/{repo_id}")

    async def clone_git(self, url: str, branch: str = "main") -> Path:
        self.base_path.mkdir(parents=True, exist_ok=True)
        repo_dir = self.base_path / "repo"
        cmd = ["git", "clone", "--depth", "1", "--branch", branch,
               "--single-branch", "--quiet", url, str(repo_dir)]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            cmd2 = ["git", "clone", "--depth", "1", "--quiet", url, str(repo_dir)]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr2 = await proc2.communicate()
            if proc2.returncode != 0:
                raise RuntimeError(
                    f"git clone failed: {stderr2.decode()}. "
                    "Make sure Git is installed: https://git-scm.com/download/win"
                )
        return repo_dir

    async def extract_zip(self, zip_bytes: bytes) -> Path:
        self.base_path.mkdir(parents=True, exist_ok=True)
        zip_path    = self.base_path / "upload.zip"
        extract_dir = self.base_path / "_extract"
        repo_dir    = self.base_path / "repo"
        async with aiofiles.open(zip_path, "wb") as f:
            await f.write(zip_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in zf.namelist():
                if ".." in m or m.startswith("/"):
                    raise ValueError("Unsafe path in ZIP")
            zf.extractall(extract_dir)
        entries = list(extract_dir.iterdir())
        source  = entries[0] if len(entries) == 1 and entries[0].is_dir() else extract_dir
        shutil.copytree(source, repo_dir)
        return repo_dir

    async def walk_repository(self, repo_dir: Path) -> AsyncGenerator[Dict, None]:
        max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        count = 0
        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
            for filename in sorted(files):
                if count >= settings.MAX_FILES_PER_REPO:
                    return
                fpath = Path(root) / filename
                ext   = fpath.suffix.lower()
                if ext in SKIP_EXT or filename.endswith((".min.js", ".min.css")):
                    continue
                try:
                    stat = fpath.stat()
                except OSError:
                    continue
                if stat.st_size > max_bytes:
                    continue
                rel       = str(fpath.relative_to(repo_dir))
                is_binary = self._sniff_binary(fpath)
                language  = LANG_MAP.get(ext)
                content: Optional[str] = None
                if not is_binary:
                    try:
                        raw     = fpath.read_bytes()
                        content = raw.decode("utf-8", errors="replace")
                        if len(content) > 80_000:
                            content = content[:80_000]
                    except Exception:
                        is_binary = True
                count += 1
                yield {
                    "path": rel, "name": filename, "extension": ext,
                    "language": language, "size_bytes": stat.st_size,
                    "line_count": content.count("\n") + 1 if content else 0,
                    "content": content, "is_binary": is_binary,
                }

    @staticmethod
    def _sniff_binary(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                return b"\x00" in f.read(8192)
        except Exception:
            return True

    def build_file_tree(self, paths: List[str]) -> Dict:
        root: Dict = {"name": "/", "type": "directory", "children": {}}
        for p in sorted(paths):
            parts = Path(p).parts
            node  = root
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    node["children"][part] = {"name": part, "type": "file", "path": p}
                else:
                    if part not in node["children"]:
                        node["children"][part] = {"name": part, "type": "directory", "children": {}}
                    node = node["children"][part]
        return self._to_list(root)

    def _to_list(self, node: Dict) -> Dict:
        if node["type"] == "directory" and "children" in node:
            node["children"] = sorted(
                [self._to_list(c) for c in node["children"].values()],
                key=lambda x: (x["type"] == "file", x["name"].lower()),
            )
        return node

    @staticmethod
    def language_stats(files: List[Dict]) -> Dict[str, float]:
        totals: Dict[str, int] = {}
        grand = 0
        for f in files:
            if f.get("language") and not f.get("is_binary"):
                totals[f["language"]] = totals.get(f["language"], 0) + f.get("line_count", 0)
                grand += f.get("line_count", 0)
        if not grand:
            return {}
        return {lang: round(lines / grand * 100, 1)
                for lang, lines in sorted(totals.items(), key=lambda x: -x[1])}

    def cleanup(self):
        if self.base_path.exists():
            shutil.rmtree(self.base_path, ignore_errors=True)
