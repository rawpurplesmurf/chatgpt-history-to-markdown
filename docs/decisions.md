# Decisions (append-only)

## D-20260227-0001
Date: 2026-02-27 08:30
Inputs: CR-20260227-0001
PRD: Vector Search / First-Load Latency

Decision:
Root cause of "vector search not working" is first-request latency, not a code bug.
The sentence-transformer model (all-MiniLM-L6-v2) is loaded lazily on the first search
request, taking 20–40s. During that time the browser appears to hang or return nothing.
The backend, index integrity, and API are all correct.

Rationale:
- Direct Python test confirmed SearchIndex loads and returns results for all modes.
- Live curl tests against running server confirmed all three modes return results.
- sentence-transformers v5.2.0 and torch 2.9.1 are installed and import cleanly.
- 93,290 chunks indexed with 384-dim embeddings; count/dimension integrity verified.

Alternatives considered:
- API incompatibility with sentence-transformers v5.2.0 (rejected: API is compatible, tolist() works)
- PyTorch missing or broken (rejected: torch 2.9.1 imports and works)
- serve.sh not using venv Python (rejected: serve.sh correctly uses .venv/bin/python)
- Index count mismatch between meta.json and embeddings.bin (rejected: counts match at 93,290)
- Frontend JS bug (rejected: modeEl wiring and runSearch() logic are correct)

Acceptance / test:
- Run keyword search, then immediately run vector search; vector should return results on second attempt.
- Or: make a direct curl request to /search?q=test&mode=vector and confirm count > 0.
