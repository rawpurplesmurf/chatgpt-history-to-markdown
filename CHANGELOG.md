# Changelog

All notable changes to the ChatGPT History Converter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

