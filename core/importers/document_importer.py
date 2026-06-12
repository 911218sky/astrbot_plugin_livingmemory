"""Import plain-text documents into memory-sized chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DocumentImportError(ValueError):
    """Raised when a document import request is invalid."""


@dataclass(frozen=True)
class DocumentChunk:
    """One importable chunk from a source document."""

    source_path: str
    title: str
    content: str
    chunk_index: int
    chunk_count: int
    metadata: dict[str, Any] | None = None


class DocumentImporter:
    """Scan Markdown/text files and split them into bounded chunks."""

    SUPPORTED_EXTENSIONS = frozenset({".md", ".markdown", ".txt", ".json"})

    def __init__(
        self,
        *,
        max_files: int = 50,
        max_chunks: int = 200,
        chunk_size: int = 1800,
        chunk_overlap: int = 180,
    ):
        if chunk_size < 500:
            raise ValueError("chunk_size must be at least 500")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.max_files = max_files
        self.max_chunks = max_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_chunks(self, import_path: str) -> list[DocumentChunk]:
        """Load all supported files under import_path and return import chunks."""
        files = self._collect_files(import_path)
        documents = []
        for file_path in files:
            documents.append((str(file_path), self._read_text(file_path)))
        return self.load_text_documents(documents)

    def load_text_documents(
        self,
        documents: list[tuple[str, str]],
    ) -> list[DocumentChunk]:
        """Load already-decoded text documents and return import chunks."""
        if not documents:
            raise DocumentImportError("no uploaded documents found")
        if len(documents) > self.max_files:
            raise DocumentImportError(f"too many files; limit is {self.max_files}")

        chunks: list[DocumentChunk] = []
        for source_path, text in documents:
            source_name = str(source_path or "uploaded-document").strip()
            if not self._is_supported(Path(source_name)):
                continue
            normalized = self._normalize_text(text)
            if not normalized:
                continue

            if Path(source_name).suffix.lower() == ".json":
                chunks.extend(self._load_livingmemory_json(source_name, normalized))
                if len(chunks) > self.max_chunks:
                    raise DocumentImportError(
                        f"too many chunks; limit is {self.max_chunks}"
                    )
                continue

            raw_chunks = self._split_text(normalized)
            chunk_count = len(raw_chunks)
            title = self._extract_title(normalized, Path(source_name))
            for index, content in enumerate(raw_chunks, start=1):
                chunks.append(
                    DocumentChunk(
                        source_path=source_name,
                        title=title,
                        content=content,
                        chunk_index=index,
                        chunk_count=chunk_count,
                    )
                )
                if len(chunks) > self.max_chunks:
                    raise DocumentImportError(
                        f"too many chunks; limit is {self.max_chunks}"
                    )

        if not chunks:
            raise DocumentImportError("no importable document content found")
        return chunks

    def _load_livingmemory_json(
        self,
        source_name: str,
        text: str,
    ) -> list[DocumentChunk]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DocumentImportError(f"invalid JSON export: {source_name}") from exc

        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise DocumentImportError(
                "JSON import must be a LivingMemory export with an items array"
            )

        chunks: list[DocumentChunk] = []
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            content = self._normalize_text(str(item.get("text") or ""))
            if not content:
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            title = (
                str(metadata.get("canonical_summary") or "").strip()
                or self._extract_title(content, Path(source_name))
            )
            chunks.append(
                DocumentChunk(
                    source_path=f"{source_name}#memory-{item.get('id', index)}",
                    title=title[:120],
                    content=content,
                    chunk_index=1,
                    chunk_count=1,
                    metadata={
                        "exported_memory_id": item.get("id"),
                        "exported_doc_id": item.get("doc_id"),
                        "exported_metadata": metadata,
                        "exported_created_at": item.get("created_at"),
                        "exported_updated_at": item.get("updated_at"),
                    },
                )
            )

        if not chunks:
            raise DocumentImportError("no importable memories found in JSON export")
        return chunks

    def _collect_files(self, import_path: str) -> list[Path]:
        cleaned = (import_path or "").strip().strip('"').strip("'")
        if not cleaned:
            raise DocumentImportError("path is empty")

        path = Path(cleaned).expanduser()
        if not path.exists():
            raise DocumentImportError(f"path does not exist: {path}")

        if path.is_file():
            files = [path] if self._is_supported(path) else []
        elif path.is_dir():
            files = [
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and self._is_supported(candidate)
            ]
        else:
            files = []

        files = sorted(files, key=lambda item: str(item))
        if not files:
            raise DocumentImportError(
                "no supported files found (.md, .markdown, .txt)"
            )
        if len(files) > self.max_files:
            raise DocumentImportError(f"too many files; limit is {self.max_files}")
        return files

    def _is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def _read_text(self, path: Path) -> str:
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise DocumentImportError(f"failed to decode text file: {path}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        return text.strip()

    @staticmethod
    def _extract_title(text: str, path: Path) -> str:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                if title:
                    return title[:120]
            if stripped:
                return stripped[:120]
        return path.stem

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                split_at = self._find_split(text, start, end)
                if split_at > start:
                    end = split_at
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    @staticmethod
    def _find_split(text: str, start: int, end: int) -> int:
        search_start = start + max((end - start) // 2, 1)
        for marker in ("\n## ", "\n# ", "\n\n", "\n", "。", ". "):
            pos = text.rfind(marker, search_start, end)
            if pos != -1:
                return pos + len(marker)
        return end
