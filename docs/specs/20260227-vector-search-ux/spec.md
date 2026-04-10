# Feature Spec: 20260227-vector-search-ux

Status: Done
Created: 2026-02-27 08:30
Inputs: CR-20260227-0001
Decisions: D-20260227-0001

## Summary
Vector search appears broken on first use because the sentence-transformer model loads lazily,
causing a 20–40s silent wait. The fix is to surface this latency clearly so users know
to wait, and optionally pre-warm the model at server startup.

## User Stories & Acceptance

### US1: First-load feedback (Priority: P1)
Narrative:
- As a user, I want to know when vector search is loading, so that I don't think it's broken.

Acceptance scenarios:
1. Given the server just started, When I run a vector/hybrid search, Then I see a loading
   indicator until results appear (not just empty results). (Verifies: FR-001)
2. Given the server just started, When I look at the terminal, Then I can see a log line
   confirming whether vector search is enabled. (Verifies: FR-002)

### US2: Model pre-warm (Priority: P2)
Narrative:
- As a user, I want the model to be ready when I first search, so that my first search is fast.

Acceptance scenarios:
1. Given serve.py starts with a valid search index, When startup completes, Then the
   sentence-transformer model is already loaded. (Verifies: FR-003)

## Requirements

Functional requirements:
- FR-001: The UI must show a visible loading/waiting state during the first vector/hybrid
  search while the model is loading. (Sources: CR-20260227-0001; D-20260227-0001)
- FR-002: serve.py must log a clear startup message indicating vector search is enabled
  (or disabled), so the user can verify status in the terminal. (Sources: CR-20260227-0001; D-20260227-0001)
- FR-003 (optional): serve.py should pre-warm the sentence-transformer model in a background
  thread at startup so the first search is not delayed. (Sources: CR-20260227-0001; D-20260227-0001)

## Edge cases
- If sentence-transformers is unavailable at serve time, the startup log should say vector
  search is disabled (not silently fail). (Verifies: FR-002)
- If pre-warm is enabled but model load fails, fall back gracefully and log the error. (Verifies: FR-003)
