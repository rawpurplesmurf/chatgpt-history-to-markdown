# Traceability (How to follow the audit trail)

Start here:
0) If this repo adopted ATHENA after it already had history, review `docs/audit/git-history.md`
   (derived from Git; not customer verbatim).
1) Find the relevant raw request in `docs/requests.md` (CR-...).
2) Read linked interpretations/tradeoffs in `docs/decisions.md` (D-...).
3) Open the feature spec at `docs/specs/<FEATURE_ID>/spec.md`.
   - Requirements use IDs (FR-...) and include `Sources: CR-...; D-...`.
   - Acceptance scenarios include `Verifies: FR-...`.
4) Open the feature task list at `docs/specs/<FEATURE_ID>/tasks.md`.
   - Tasks include `Implements: FR-...`.
5) Review execution notes in `docs/progress.txt` for commands, outcomes, and completion.

## Active Features
- `20260227-vector-search-ux` — Fix vector search first-load UX (latency transparency)
