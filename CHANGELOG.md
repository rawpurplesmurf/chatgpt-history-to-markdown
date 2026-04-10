# Changelog

All notable changes to the ChatGPT History Converter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Hybrid search index (FTS5 keyword + optional vector embeddings) built during conversion.
- `/search` endpoint and web UI search panel for fast retrieval.
- `run_with_vectors.sh` helper to create a venv, install vector dependencies, and run the converter.
- Basic unittest coverage for the search index builder and query behavior.
- Node-based UI smoke test for search panel structure.
- Advanced search controls (mode, scope, role, date range, sort, limit, snippet size).
- Voice message transcripts: conversations containing audio turns now include a `*[Voice message transcription]*` block with the embedded transcript text.
- serve.py pre-warms the sentence-transformer model at startup so the first vector/hybrid search is not delayed.
- serve.py logs a startup message indicating whether vector search is enabled or disabled.
- Web UI shows a specific "timed out" message (90 s timeout) if the server is still loading the model on first vector/hybrid search.

### Changed
- Search indexing is now disabled by default; set `CHATGPT_HISTORY_BUILD_SEARCH_INDEX=1` or use `./run_with_vectors.sh` to enable it.

### Fixed
- Index generation now handles missing/invalid conversation timestamps and groups them under an "Unknown Date" section.
- Converter no longer writes bare section headers for turns that produce no text or attachment output (e.g. audio-only turns with no transcript).
- Hyphenated search queries (e.g. `follow-up`, `co-pay`) no longer crash the FTS5 search with `OperationalError: no such column`.

## [2.0.0] - 2026-01-03

### Added
- **Comprehensive error handling** - Graceful failure handling with detailed error messages
- **Console logging** - Real-time progress updates with INFO, WARNING, and ERROR levels
- **Image and attachment support** - Automatically copies and embeds images/files from conversations
- **Mermaid diagram rendering** - Automatically renders mermaid diagrams to PNG images using pyppeteer
- **Organized directory structure** - Separate `attachments/` folder for media files
- **Enhanced index** - Chronological grouping by month with metadata
- **Proper OOP design** - Refactored into `ChatGPTConverter` class
- **Statistics tracking** - Conversion summary with success/failure counts
- **Message type emoji** - Visual indicators for user/assistant/system/tool messages
- **Timestamp formatting** - Full creation and update timestamps in output
- **Conversation metadata** - Includes conversation ID and timestamps in each file

### Changed
- **Input path** - Now reads from `chatgpt-history-source/` directory (was root directory)
- **Output path** - Now outputs to `markdown/` directory (was `ChatGPT_Library/`)
- **Message filtering** - Properly skips hidden system messages and empty tool calls
- **File naming** - Improved sanitization and date prefixing
- **Index format** - Enhanced with emoji, better formatting, and month grouping

### Fixed
- **Message extraction** - Properly handles multi-part messages and various content types
- **Encoding issues** - Consistent UTF-8 handling throughout
- **Missing timestamps** - Graceful handling of conversations without timestamps
- **Empty conversations** - Skips conversations with no displayable messages

### Technical
- Added comprehensive logging with `logging` module
- Implemented `pathlib.Path` for robust path handling
- Added type hints for better code clarity
- Structured code with clear separation of concerns
- Added detailed docstrings for all methods

## [1.0.0] - Initial Version

### Features
- Basic conversation export to Markdown
- Simple index generation
- Filename sanitization
- Date-based organization
