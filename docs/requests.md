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
