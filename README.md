# VaadAI

A Retrieval-Augmented-Generation (RAG) chatbot that answers Indian legal questions in plain language. It pulls relevant judgments and statutes from [Indian Kanoon](https://indiankanoon.org), then synthesises a short, citation-backed answer with Claude. Available over HTTP and WhatsApp.

## What it does

- **Classifies** the user's message — legal vs. conversational vs. out-of-scope — and short-circuits the non-legal cases.
- **Plans an Indian Kanoon search** with Claude, fetches the top documents in parallel, and builds a context.
- **Synthesises** a 6–10 line plain-language answer with a strict format and disclaimer.
- **Suggests** 3 follow-up questions and surfaces 2 related cases.
- **Sessions are first-class** — per-user FIFO queue, 8-question cap, 10-minute idle timeout, cross-Gunicorn-worker safe (SQLite-backed). Concurrent suggestion taps on WhatsApp are processed in order, never dropped, and stale webhook redeliveries are discarded.

## Project layout

| File | Purpose |
|------|---------|
| `flask_app.py` | HTTP routes (`/ask`, `/recommendations`, `/webhook`) and the WhatsApp webhook + background worker |
| `claude_rag_test.py` | RAG pipeline: classify → plan search → fetch from IK → synthesise → generate follow-ups |
| `session_store.py` | SQLite-backed session, history, suggestion-map and FIFO message queue |
| `gunicorn.conf.py` | Production server config (2 workers × 4 threads) |
| `requirements.txt` | Python dependencies |

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/agastyasingh/vaadai-backend.git
cd vaadai-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` in the project root:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
INDIAN_KANOON_API_KEY=...

# WhatsApp (only needed if you're wiring the WA webhook)
WA_TOKEN=...
PHONE_NUMBER_ID=...
VERIFY_TOKEN=vaadaiVerificationToken2001

# Optional — override the SQLite session-store path (default: /tmp/vaadai_sessions.db)
VAADAI_SESSION_DB=/var/lib/vaadai/sessions.db
```

### 3. Run

Local development:
```bash
python flask_app.py
```

Production:
```bash
gunicorn -c gunicorn.conf.py flask_app:app
```

## HTTP API

### `POST /ask`

```json
{
  "question": "Can an FIR be quashed under Section 482 CrPC?",
  "session_id": "client-uuid",
  "is_suggested": false
}
```

Returns `answer`, `suggestions`, `citations`, `disclaimer`, `session_id`, `question_count`, `questions_remaining`.

Set `is_suggested: true` when the user tapped a question from the previous response's suggestions — the conversation history is then included in the prompt. Fresh, standalone questions get a clean slate.

### `POST /recommendations`

Fast follow-ups for an already-answered question — pass `question` and `answer` and you get back 3 suggestions in one short Claude call.

### `POST /webhook`

WhatsApp Business webhook — handles plain text and interactive list-reply taps. Returns `200` immediately; the actual RAG work runs in a per-user background thread, FIFO ordered.

## Notes

- The RAG pipeline routes through Claude (Sonnet) for classification, search planning, answer synthesis and follow-up generation. Per-request latency is dominated by Claude + Indian Kanoon API calls — Gunicorn settings won't shrink it.
- The session store is plain SQLite (stdlib, WAL mode). No Redis or external service required.
