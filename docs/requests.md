# Customer Requests (append-only)

## CR-20260227-0001
Date: 2026-02-27 08:13
Source: chat

Request (verbatim):
I do not think the vector search is working

Notes:
- User ran ./run.sh (conversion only), then ./serve.sh --open
- Follow-up diagnostics confirmed: keyword search works, vector mode returns no results in UI
- Backend API tested directly: all three modes (keyword, vector, hybrid) return correct results
- Root cause investigation found backend fully functional; likely UX/latency issue on first load

## CR-20260227-0002
Date: 2026-02-27 09:45
Source: chat

Request (verbatim):
yes use Athena to document the bug and investigate whats causing

Notes:
- Context: 85 markdown files (including 11 from May 2025) are excluded from search index
- Files have section headers (### 🤖 **Assistant**) but empty message bodies
- Example: 2025-05-26_PostgreSQL_Question.md has 46 assistant headers, all empty
- Root cause is in converter.py — certain message types produce empty content blocks
- User wants bug documented and root cause found in the converter

## CR-20260227-0003
Date: 2026-02-27 09:39
Source: chat (server log shared by user)

Request (verbatim):
see this error [sqlite3.OperationalError: no such column: up]

Notes:
- Error triggered by hyphenated search queries (e.g. "follow-up", "co-pay")
- _tokenize() regex included hyphen in token character class
- FTS5 parsed "follow-up" as "follow NOT up", treating "up" as column reference
- Fix: remove hyphen from _tokenize regex (one-char change in search_index.py line 81)
