# Product Requirements Document

## Project
chatgpt-history-to-markdown — converts ChatGPT export JSON to browsable Markdown archive with hybrid search.

## Current Features (shipped)
- Convert conversations.json to per-conversation .md files
- Local web viewer (serve.py + web/index.html)
- Hybrid search: FTS5 keyword + sentence-transformer vector (search_index.py)
- Search UI controls: mode (hybrid/keyword/vector), scope, role filter, date range, sort, limit
- Mermaid diagram rendering via pyppeteer (optional)

## Active Requirements

### Vector Search First-Load UX (Sources: CR-20260227-0001; D-20260227-0001)
The server loads the sentence-transformer model lazily on the first search request.
This causes a 20–40s delay that makes vector search appear broken to users.
The UX should clearly communicate loading state so users wait instead of giving up.

## Next / Backlog
- [ ] Show a loading indicator or status message during first vector search (model warm-up delay)
- [ ] Pre-warm the sentence-transformer model at server startup (optional background load)
- [ ] Add server startup log line clearly indicating whether vector search is available/enabled
