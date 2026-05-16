"""Tests for DocumentExtractionHook."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.document_extraction_hook import DocumentExtractionHook
from nanobot.agent.hook import AgentHookContext


@pytest.mark.asyncio
async def test_hook_extracts_text_and_mutates_message():
    """Hook appends extracted document text to the message content."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": "Read this tender.",
            "media": ["/tmp/fake-tender.pdf"],
        }],
    )

    with patch(
        "pathlib.Path.is_file",
        return_value=True,
    ), patch(
        "nanobot.agent.document_extraction_hook.extract_documents",
        return_value=("Read this tender.\n\n[File: fake-tender.pdf]\nPage 1\nTender text here", []),
    ):
        await hook.before_iteration(context)

    assert "Tender text here" in context.messages[0]["content"]
    assert context.messages[0]["media"] is None


@pytest.mark.asyncio
async def test_hook_ignores_non_user_messages():
    """Assistant / tool messages are never processed."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[
            {"role": "assistant", "content": "I'll read that", "media": ["/tmp/x.pdf"]},
            {"role": "user", "content": "OK thanks", "media": []},
        ],
    )

    await hook.before_iteration(context)
    # extract_documents should NOT be called at all
    assert context.messages[0]["media"] == ["/tmp/x.pdf"]


@pytest.mark.asyncio
async def test_hook_ignores_messages_without_media():
    """Messages with no media field are skipped entirely."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{"role": "user", "content": "hello"}],
    )

    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents"
    ) as mock_extract:
        await hook.before_iteration(context)
        mock_extract.assert_not_called()

    assert context.messages[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_hook_ignores_messages_with_empty_media():
    """Empty media list is treated as no media."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{"role": "user", "content": "hi", "media": []}],
    )

    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents"
    ) as mock_extract:
        await hook.before_iteration(context)
        mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_hook_handles_extraction_failure_gracefully():
    """If extract_documents raises, the message is left unchanged."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": "original content",
            "media": ["/tmp/bad.pdf"],
        }],
    )

    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents",
        side_effect=RuntimeError("corrupted file"),
    ):
        # Must not raise
        await hook.before_iteration(context)

    assert context.messages[0]["content"] == "original content"
    assert context.messages[0]["media"] == ["/tmp/bad.pdf"]


@pytest.mark.asyncio
async def test_hook_processes_multiple_user_messages():
    """Multiple user messages in context are all scanned."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[
            {"role": "assistant", "content": "Please attach the docs"},
            {"role": "user", "content": "Here are two files", "media": ["/tmp/a.pdf", "/tmp/b.docx"]},
            {"role": "user", "content": "And another one", "media": ["/tmp/c.txt"]},
        ],
    )

    def side_effect(content, media_paths):
        doc_names = [Path(p).name for p in media_paths]
        extracted = content + "\n\n" + "\n\n".join(f"[File: {n}]\ndoc text" for n in doc_names)
        return extracted, []

    with patch(
        "pathlib.Path.is_file",
        return_value=True,
    ), patch(
        "nanobot.agent.document_extraction_hook.extract_documents",
        side_effect=side_effect,
    ):
        await hook.before_iteration(context)

    # First user message should get appended text
    assert "doc text" in context.messages[1]["content"]
    # Second user message should also get appended text
    assert "doc text" in context.messages[2]["content"]


@pytest.mark.asyncio
async def test_hook_skips_non_file_paths():
    """Paths that don't look like files (e.g. URLs, data URIs) are skipped."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": "check this",
            "media": ["https://example.com/doc.pdf", "data:application/pdf;base64,abc"],
        }],
    )

    # These paths won't pass Path(path).is_file() so no extraction should happen
    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents"
    ) as mock_extract:
        await hook.before_iteration(context)
        mock_extract.assert_not_called()


@pytest.mark.asyncio
async def test_hook_preserves_media_when_no_documents_found():
    """If extract_documents returns unchanged content, media stays intact."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": "hello",
            "media": ["/tmp/photo.png"],  # image only, no documents
        }],
    )

    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents",
        return_value=("hello", ["/tmp/photo.png"]),
    ):
        await hook.before_iteration(context)

    assert context.messages[0]["content"] == "hello"
    assert context.messages[0]["media"] == ["/tmp/photo.png"]


@pytest.mark.asyncio
async def test_hook_with_content_as_list():
    """Content as a list (OpenAI multi-block format) should be skipped, not crash."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
            "media": ["/tmp/x.pdf"],
        }],
    )

    # Should not raise — content is not a string, so it's skipped
    await hook.before_iteration(context)
    assert context.messages[0]["content"] == [{"type": "text", "text": "hello"}]


@pytest.mark.asyncio
async def test_hook_with_null_content():
    """Null content should be skipped safely."""
    hook = DocumentExtractionHook()
    context = AgentHookContext(
        iteration=0,
        messages=[{
            "role": "user",
            "content": None,
            "media": ["/tmp/x.pdf"],
        }],
    )

    with patch(
        "nanobot.agent.document_extraction_hook.extract_documents"
    ) as mock_extract:
        await hook.before_iteration(context)
        mock_extract.assert_not_called()
