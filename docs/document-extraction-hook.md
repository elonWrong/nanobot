# Document Extraction Hook

Extract text from document attachments (PDF, DOCX, XLSX, PPTX, plain text, etc.) and inject the extracted content into the agent loop.

## Quick Start

Register the hook when creating your `AgentLoop`:

```python
from nanobot.agent import AgentLoop, DocumentExtractionHook

loop = AgentLoop(
    hooks=[DocumentExtractionHook()],
    # ... other config
)
```

Or with the `AgentRunner` directly:

```python
from nanobot.agent import AgentRunner, AgentRunSpec, AgentHook
from nanobot.agent.document_extraction_hook import DocumentExtractionHook

spec = AgentRunSpec(
    hook=DocumentExtractionHook(),
    # ... other config
)
result = await AgentRunner(provider).run(spec)
```

## How It Works

The hook runs **before each agent iteration** (in `before_iteration()`). It:

1. Scans every user message for files in the `media` field
2. Identifies document files (PDF, DOCX, XLSX, PPTX, `.txt`, `.md`, `.csv`, `.json`, `.yaml`, etc.)
3. Extracts text using nanobot's existing `extract_documents()` utility
4. Appends extracted text directly into the message content so the agent can read it
5. Preserves image paths separately for vision processing

### Example

An inbound message with a PDF attachment:

```python
message = {
    "role": "user",
    "content": "Please review this tender document.",
    "media": ["/path/to/tender.pdf", "/path/to/photo.png"],
}
```

After the hook runs, the message content becomes:

```
Please review this tender document.

[File: tender.pdf]
Page 1 of 12
... extracted PDF text ...

--- Page 2 ---
... more extracted text ...
```

The `photo.png` stays in `media` as an image path for vision processing.

## What Formats Are Supported

| Format | Parser | Notes |
|--------|--------|-------|
| `.pdf` | pypdf | Page-by-page extraction |
| `.docx` | python-docx | Paragraph-level text |
| `.xlsx` | openpyxl | Sheet-by-sheet, row-by-row |
| `.pptx` | python-pptx | Slide-by-slide, shape-by-shape |
| `.txt` | built-in | UTF-8 with latin-1 fallback |
| `.md`, `.csv`, `.json`, `.yaml`, `.yml`, `.toml`, `.xml`, `.html`, `.log` | built-in | Direct file read |

## Constraints

- **No new dependencies** — reuses existing `pypdf`, `python-docx`, `openpyxl`, `python-pptx`
- **File size limit**: 50 MB per file (skipped with a warning if larger)
- **Text length limit**: 200,000 chars per file (truncated with a note)
- **No new dependencies** — all parsers are already optional imports, loaded lazily

## Failure Handling

If extraction fails for any file, the hook:

1. Logs the exception with `logger.exception()`
2. Leaves the message unchanged — the agent never sees an error
3. Continues processing other messages

This means **bad documents cannot crash the agent iteration**.

## Customization

The hook works out of the box. You can customize it by subclassing:

```python
from nanobot.agent.document_extraction_hook import DocumentExtractionHook

class CustomDocHook(DocumentExtractionHook):
    async def before_iteration(self, context):
        # Pre-filter: only process files from certain paths
        # ... your custom logic here
        await super().before_iteration(context)
```

## Where Documents Come From

Documents arrive via the `media` field on user messages. They're typically added by:

- **Channel ingestion** — chat channels (Telegram, Discord, etc.) that upload files
- **SDK/manual injection** — creating `InboundMessage` objects with `media` paths in your application code
- **Web UI uploads** — the web interface passes file paths into `media`

By the time the hook sees a message, the files should already exist on disk at the paths listed in `media`.
