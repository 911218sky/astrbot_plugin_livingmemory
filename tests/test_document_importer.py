from __future__ import annotations

import pytest

from astrbot_plugin_livingmemory.core.importers import (
    DocumentImportError,
    DocumentImporter,
)


def test_loads_markdown_and_text_files(tmp_path):
    (tmp_path / "note.md").write_text("# Project Notes\n\nAlpha facts.", encoding="utf-8")
    (tmp_path / "raw.txt").write_text("Plain text memory.", encoding="utf-8")
    (tmp_path / "skip.json").write_text("{}", encoding="utf-8")

    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_chunks(
        str(tmp_path)
    )

    assert len(chunks) == 2
    assert {chunk.title for chunk in chunks} == {"Project Notes", "Plain text memory."}
    assert all(chunk.chunk_index == 1 for chunk in chunks)
    assert all(chunk.chunk_count == 1 for chunk in chunks)


def test_splits_large_document_with_overlap(tmp_path):
    document = tmp_path / "large.md"
    document.write_text("# Large\n\n" + ("paragraph sentence. " * 120), encoding="utf-8")

    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_chunks(
        str(document)
    )

    assert len(chunks) > 1
    assert chunks[0].title == "Large"
    assert [chunk.chunk_index for chunk in chunks] == list(range(1, len(chunks) + 1))
    assert all(chunk.chunk_count == len(chunks) for chunk in chunks)
    assert all(len(chunk.content) <= 500 for chunk in chunks)


def test_rejects_too_many_supported_files(tmp_path):
    for index in range(3):
        (tmp_path / f"note-{index}.md").write_text("content", encoding="utf-8")

    with pytest.raises(DocumentImportError, match="too many files"):
        DocumentImporter(max_files=2).load_chunks(str(tmp_path))


def test_rejects_missing_supported_content(tmp_path):
    (tmp_path / "data.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DocumentImportError, match="no supported files"):
        DocumentImporter().load_chunks(str(tmp_path))
