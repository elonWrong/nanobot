"""Hook that extracts text from document attachments on ingress.

Scans messages for document files in their ``media`` lists, extracts text
via the existing ``extract_documents`` utility, and mutates the message
content in-place.  Used as a programmatic entry point for business
intelligence document ingestion (e.g. tender documents).

Failure semantics
-----------------
Errors are logged and the message is left unchanged — extraction never
crashes the agent iteration.
"""

from __future__ import annotations

from loguru import logger

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.utils.document import extract_documents


class DocumentExtractionHook(AgentHook):
    """Extract text from document attachments and inject into message content."""

    async def before_iteration(self, context: AgentHookContext) -> None:
        for i, msg in enumerate(context.messages):
            if msg.get("role") != "user":
                continue

            media_paths = msg.get("media")
            if not media_paths:
                continue

            content = msg.get("content")
            if not content or not isinstance(content, str):
                continue

            # Filter to only paths that are still file-like (not already
            # resolved to image-only by prior extraction steps).
            doc_paths = [
                p for p in media_paths
                if isinstance(p, str) and _looks_like_file(p)
            ]
            if not doc_paths:
                continue

            try:
                extracted_content, remaining_media = extract_documents(
                    content, doc_paths,
                )

                # Only mutate if we actually extracted something.
                if extracted_content != content:
                    msg["content"] = extracted_content
                    msg["media"] = remaining_media or None
                    logger.debug(
                        "DocumentExtractionHook extracted text from {} "
                        "document(s) in message at index {}",
                        sum(1 for p in doc_paths if _is_document_file(p)),
                        i,
                    )

            except Exception:
                logger.exception(
                    "DocumentExtractionHook failed to extract text "
                    "from message at index {}; leaving message unchanged",
                    i,
                )


def _is_document_file(path: str) -> bool:
    """Heuristic: does this path look like a document (not an image)?"""
    lower = path.lower()
    doc_exts = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".md", ".csv", ".json", ".log"}
    return any(lower.endswith(ext) for ext in doc_exts)


def _looks_like_file(path: str) -> bool:
    """Return True if the path looks like a real filesystem file."""
    from pathlib import Path
    return Path(path).is_file()
