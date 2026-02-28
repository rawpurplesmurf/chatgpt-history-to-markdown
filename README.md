# ChatGPT History Converter

Convert your ChatGPT conversation exports into organized, searchable Markdown documentation.

## Features

- 📝 Converts ChatGPT `conversations.json` exports to readable Markdown files
- 🖼️ Automatically copies and embeds images and attachments
- 📊 Renders Mermaid diagrams to PNG images (requires pyppeteer)
- 📚 Generates a chronological index grouped by month
- ✨ Emoji-tagged messages for easy visual scanning
- Optional hybrid search index (keyword + optional vector search) for fast retrieval
- 🔍 Comprehensive logging and error handling
- 📁 Organized output with dedicated attachments folder

## Quick Start

### Prerequisites

- Python 3.7+
- Your ChatGPT export data (see [How to Export](#how-to-export-chatgpt-data) below)

### Installation

1. Clone or download this repository
2. Activate the virtual environment:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Place your ChatGPT export in the `chatgpt-history-source/` directory
2. Run the converter:

```bash
python converter.py
```

3. Find your converted conversations in the `markdown/` directory

### Helper Scripts

This repo includes convenience scripts that set up a venv and run common tasks:

- `./run.sh` - Install dependencies and run the converter.
- `./serve.sh --open` - Start the local viewer and open the browser.
- `./run_with_vectors.sh` - Install dependencies + sentence-transformers and run the converter with search indexing enabled.
- `./run_with_vectors.sh --serve --open` - Convert and immediately launch the viewer.

### Web Viewer (Optional)

After generating `markdown/`, you can browse a formatted archive locally:

```bash
./serve.sh --open
```

This starts a small local server (default: `http://127.0.0.1:8000`) that renders Markdown in your browser.
Mermaid (```mermaid) and PlantUML (```plantuml / `@startuml`) blocks are rendered in the browser; no local diagram tooling required. PlantUML uses a PlantUML server (default: `plantuml.com`) and can be changed via the **PlantUML** button in the header.

### Search (Hybrid)

The converter can build a local search index in `markdown/` for the web viewer. Use the search box to run keyword + vector search and jump to matching conversations. Indexing is disabled by default.

Search index files:
- `markdown/search.sqlite` (FTS5 keyword index)
- `markdown/search_meta.json` (chunk metadata)
- `markdown/search_embeddings.bin` (optional vectors)

Enable indexing:

```bash
CHATGPT_HISTORY_BUILD_SEARCH_INDEX=1 python converter.py
```

Vector search uses `sentence-transformers` when available; otherwise it falls back to a lightweight hashing embedder.

```bash
pip install sentence-transformers
```

Or use the helper script (creates a venv, installs deps + sentence-transformers, runs the converter):

```bash
./run_with_vectors.sh
```

Optional tuning:
- `CHATGPT_HISTORY_EMBEDDING_BACKEND=auto|sentence-transformers|hash|off`
- `CHATGPT_HISTORY_EMBEDDING_MODEL=all-MiniLM-L6-v2`
- `CHATGPT_HISTORY_CHUNK_WORDS=320`
- `CHATGPT_HISTORY_CHUNK_OVERLAP=64`
- `CHATGPT_HISTORY_VECTOR_SCAN_LIMIT=20000` (cap vector scan size; 0 = no cap)

Note: SQLite must be built with FTS5 enabled (default on most Python builds).

Search panel controls include:
- Mode (hybrid/keyword/vector), scope (title/body), and role filter.
- Sort (relevance/newest), limit, snippet length, and date range filters.

## How to Export ChatGPT Data

1. Go to ChatGPT settings → Data Controls
2. Click "Export data"
3. Wait for the email with your download link
4. Extract the ZIP file to the `chatgpt-history-source/` folder

Your export should include:
- `conversations.json` (main conversation data)
- Various `file-*` attachments (images, documents)
- Other JSON files (user.json, etc.)

## Output Structure

```
markdown/
├── index.md                           # Chronological navigation index
├── 2024-01-15_Python_Tips.md         # Individual conversation files
├── 2024-01-14_Code_Review.md
└── attachments/                       # Images and files from conversations
    ├── file-abc123.png
    └── file-xyz789.pdf
```

## Features in Detail

### Message Formatting

Messages are tagged with emoji for quick identification:
- 👤 **User** - Your messages
- 🤖 **Assistant** - ChatGPT responses
- ⚙️ **System** - System messages
- 🔧 **Tool** - Tool/plugin outputs

Voice messages are converted using the transcript embedded in your export:

```
*[Voice message transcription]*

What's the best way to handle errors in async Python?
```

Turns that contain only audio with no transcript are omitted from the output.

### Attachments

Images are automatically:
- Copied to `markdown/attachments/`
- Embedded inline in Markdown with `![Attachment](path)`

Other files are linked as downloadable attachments.

### Index Generation

The `index.md` file organizes conversations:
- Grouped by month (newest first)
- Includes creation dates
- Links directly to conversation files

## Troubleshooting

### No conversations found

Check that `conversations.json` exists in `chatgpt-history-source/` and is valid JSON.

### Missing images

Ensure image files from your export are in `chatgpt-history-source/`. The converter looks for files matching references in `conversations.json`.

### Conversion failures

Check the console output for specific error messages. Failed conversations are logged with details.

### Vector search times out or returns no results on first use

The sentence-transformer model loads on first use, which can take 20–40 seconds. The server pre-warms the model at startup in the background, but if the server just started, your first vector or hybrid search may show a "timed out" message. Wait a moment and search again — subsequent searches are fast.

## Development

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for AI agent guidance.

### Running Tests

Unit tests (search index) use the standard library test runner:

```bash
python -m unittest discover -s tests
```

UI smoke test (Node 18+):

```bash
node --test tests/smoke_ui.test.js
```

Note: Tests require SQLite FTS5 support (enabled in most Python builds). Manual checks are still recommended for rendering output in `markdown/`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## License

MIT License - feel free to use and modify as needed.
