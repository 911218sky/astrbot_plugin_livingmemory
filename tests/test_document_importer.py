from __future__ import annotations

import json

import pytest

from astrbot_plugin_livingmemory.core.importers import (
    DocumentImportError,
    DocumentImporter,
)


def test_loads_markdown_and_text_files(tmp_path):
    (tmp_path / "note.md").write_text("# Project Notes\n\nAlpha facts.", encoding="utf-8")
    (tmp_path / "raw.txt").write_text("Plain text memory.", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("{}", encoding="utf-8")

    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_chunks(
        str(tmp_path)
    )

    assert len(chunks) == 2
    assert {chunk.title for chunk in chunks} == {"Project Notes", "Plain text memory."}
    assert all(chunk.chunk_index == 1 for chunk in chunks)
    assert all(chunk.chunk_count == 1 for chunk in chunks)


def test_loads_livingmemory_json_export(tmp_path):
    export_file = tmp_path / "livingmemory-export.json"
    export_file.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": 42,
                        "doc_id": "doc-42",
                        "text": "Exported memory body.",
                        "metadata": {
                            "session_id": "old-session",
                            "persona_id": "old-persona",
                            "canonical_summary": "Export Summary",
                        },
                        "created_at": "created",
                        "updated_at": "updated",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_chunks(
        str(export_file)
    )

    assert len(chunks) == 1
    assert chunks[0].title == "Export Summary"
    assert chunks[0].content == "Exported memory body."
    assert chunks[0].metadata["exported_memory_id"] == 42
    assert chunks[0].metadata["exported_metadata"]["session_id"] == "old-session"


def test_extracts_markdown_title_from_utf8_bom_file(tmp_path):
    document = tmp_path / "bom.md"
    document.write_text("# BOM Title\n\nBody", encoding="utf-8-sig")

    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_chunks(
        str(document)
    )

    assert len(chunks) == 1
    assert chunks[0].title == "BOM Title"


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


def test_loads_uploaded_text_documents():
    chunks = DocumentImporter(chunk_size=500, chunk_overlap=50).load_text_documents(
        [("upload.md", "# Uploaded\n\nMemory body.")]
    )

    assert len(chunks) == 1
    assert chunks[0].source_path == "upload.md"
    assert chunks[0].title == "Uploaded"
    assert chunks[0].content == "# Uploaded\n\nMemory body."


def test_rejects_non_livingmemory_json_documents():
    with pytest.raises(DocumentImportError, match="LivingMemory export"):
        DocumentImporter().load_text_documents([("data.json", "{}")])


def test_rejects_too_many_supported_files(tmp_path):
    for index in range(3):
        (tmp_path / f"note-{index}.md").write_text("content", encoding="utf-8")

    with pytest.raises(DocumentImportError, match="too many files"):
        DocumentImporter(max_files=2).load_chunks(str(tmp_path))


def test_rejects_missing_supported_content(tmp_path):
    (tmp_path / "data.csv").write_text("{}", encoding="utf-8")

    with pytest.raises(DocumentImportError, match="no supported files"):
        DocumentImporter().load_chunks(str(tmp_path))
