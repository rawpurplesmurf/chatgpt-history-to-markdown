# ChatGPT History Converter

Convert your ChatGPT conversation exports into organized, searchable Markdown documentation.

## Features

- 📝 Converts ChatGPT `conversations.json` exports to readable Markdown files
- 🖼️ Automatically copies and embeds images and attachments
- 📊 Renders Mermaid diagrams to PNG images (requires pyppeteer)
- 📚 Generates a chronological index grouped by month
- ✨ Emoji-tagged messages for easy visual scanning
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

## Development

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for AI agent guidance.

### Running Tests

Currently, testing is manual. Run the converter and verify output in `markdown/`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and changes.

## License

MIT License - feel free to use and modify as needed.

