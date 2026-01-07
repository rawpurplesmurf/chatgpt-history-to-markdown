# Repository Guidelines

This repository provides a small Python utility that converts a ChatGPT export (`chatgpt-history-source/conversations.json` plus `file-*` attachments) into a searchable Markdown archive in `markdown/`.

## Project Structure & Module Organization
- `converter.py`: main entrypoint; contains the `ChatGPTConverter` class and conversion logic.
- `run.sh`: venv wrapper to install deps and run the converter.
- `serve.py`, `serve.sh`, `web/`: lightweight local web viewer for the generated archive.
- `requirements.txt`: runtime dependencies (`pyppeteer` is used for Mermaid diagram rendering).
- `chatgpt-history-source/`: local input directory for exports (keep empty in commits).
- `markdown/`: generated output directory created at runtime (archive + `attachments/`).

## Build, Test, and Development Commands
```bash
./run.sh --no-mermaid --stream --workers 6
./serve.sh --open
```
- The converter exits if `chatgpt-history-source/conversations.json` is missing.
- Output files follow `markdown/YYYY-MM-DD_<Title>.md` and `markdown/index.md`.
- To change paths, adjust `ChatGPTConverter(source_dir=..., output_dir=...)` in `converter.py`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, clear docstrings, type hints where practical.
- Prefer `pathlib.Path` for filesystem work and keep logging consistent with the existing `logging` setup.
- File names are `snake_case.py`; generated Markdown is date-prefixed.
- Update `CHANGELOG.md` for user-visible behavior changes (Keep a Changelog format).

## Testing Guidelines
- No automated test suite yet; changes should include a manual smoke test on a small export.
- Quick sanity checks: `python -m compileall converter.py` and verify links/attachments in `markdown/index.md`.

## Commit & Pull Request Guidelines
- Commit history is lightweight (e.g., “Initial commit”, “first push”); use short, imperative subjects.
- PRs should describe behavior changes, include repro steps (what input shape was used), and note whether Mermaid rendering was exercised (pyppeteer may download Chromium on first run).

## Security & Data Handling
- ChatGPT exports can contain sensitive data. Do not commit real exports, attachments, or generated `markdown/` output unless they are intentionally anonymized fixtures.
