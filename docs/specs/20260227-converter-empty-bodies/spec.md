# Feature Spec: 20260227-converter-empty-bodies

Status: Done
Created: 2026-02-27 10:30
Inputs: CR-20260227-0002
Decisions: D-20260227-1030

## Summary
85 markdown files have section headers (`### 🤖 **Assistant**`) with no body text because
converter.py silently drops audio_transcription dicts and writes headers unconditionally.
The fix extracts voice transcripts and gates header output on body content being non-empty.

## User Stories & Acceptance

### US1: Voice transcripts surfaced (Priority: P1)
Narrative:
- As a user, I want voice-message turns to show their transcript in the markdown output,
  so that those conversations are searchable and readable.

Acceptance scenarios:
1. Given a conversation with audio_transcription parts, When I run the converter,
   Then the markdown contains *[Voice message transcription]* blocks with the text. (Verifies: FR-001)
2. Given a conversation with only audio asset pointers and no transcripts, When I run
   the converter, Then no bare section headers appear in the output. (Verifies: FR-002)

### US2: Index coverage improved (Priority: P1)
Narrative:
- As a user, I want all conversations with text content to appear in search results,
  so that May 2025 conversations are findable.

Acceptance scenarios:
1. Given 85 previously-empty files, When I re-run the converter and rebuild the index,
   Then the total indexed file count increases from 5,490 toward 5,575. (Verifies: FR-003)

## Requirements

Functional requirements:
- FR-001: process_message_parts() must extract the `text` field from audio_transcription
  dicts and label it *[Voice message transcription]*. (Sources: CR-20260227-0002; D-20260227-1030)
- FR-002: format_message() must return "" when no body content is produced, preventing
  bare section headers from appearing in output. (Sources: CR-20260227-0002; D-20260227-1030)
- FR-003: After re-running the converter, files with voice transcripts must be included
  in the search index. (Sources: CR-20260227-0002; D-20260227-1030)

## Edge cases
- Audio-only turns with no transcript text produce no output at all (not even a header). (Verifies: FR-002)
- Unknown dict part types are logged at DEBUG level so future types can be discovered. (Verifies: FR-001)
- messages_processed counter only increments when format_message() returns non-empty output. (Verifies: FR-002)
