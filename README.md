# Chitragupta

A multi-service, privacy-first personal document vault. Upload an Aadhaar
card, marksheet, passport, or any sensitive document and the system
extracts structured metadata, builds a searchable semantic index, and
answers questions about the document via a two-step confirmation protocol
that keeps PII out of the response unless you explicitly opt in.

## Architecture

Five independent microservices, all started by `python activate.py`:

| Service                    | Port | Purpose                                                                 |
|----------------------------|------|-------------------------------------------------------------------------|
| Document Management        | 8080 | Postgres repository, Qdrant vector store, file storage, MCP tools       |
| Model                      | 8081 | Vision-LLM inference (HuggingFace → Groq fallback), text extraction     |
| Web Search                 | 8082 | Web search backend for cross-document lookups                           |
| Validator                  | 8083 | Document template matching, expiry and structure checks                 |
| Orchestrator + UI          | 8084 | Chat persona, tool dispatch, conversational front-end                   |

PostgreSQL and Qdrant run in Docker (`docker compose up -d`). The other
services are plain Python processes.

## The two-step confirmation flow

`get_field_value(document_id, version, "aadhaar_number")` is intentionally
non-revealing on the first call:

```json
{
  "status": "requires_confirmation",
  "value": null,
  "source": {"name": "aadhaar.jpg", "document_id": "...", "relation": "SELF"},
  "suggestion": "Field 'aadhaar_number' is cached for this document. Confirm to see the value."
}
```

A second call with `confirm: true` returns the actual value:

```json
{
  "status": "ok",
  "value": "4345 7364 7274",
  "source": {"name": "aadhaar.jpg", "document_id": "...", "relation": "SELF"},
  "suggestion": "Source: aadhaar_number is available. Press 'Retrieve original file' to view the full document."
}
```

## PII redaction at the vector layer

Qdrant stores only **field names**, never values. After a document is
ingested, the chunk payload looks like:

```json
{
  "field_pointers": ["aadhaar_number", "name", "gender", "date_of_birth", ...],
  "owner_type": "SELF",
  "relation": null,
  "relation_name": null,
  "text": "enc:<Fernet-ciphertext>",
  "description": "enc:<Fernet-ciphertext>",
  "metadata": "enc:<Fernet-ciphertext>",
  "privacy": "SENSITIVE"
}
```

`text`, `description`, `metadata`, and `original_filename` are encrypted
with a per-document key derived from `ENCRYPTION_MASTER_KEY` + document ID
+ version + upload date (scheme v2 — no description text is stored
alongside the ciphertext). The plaintext `field_pointers` and
`owner_type` are filterable so you can search "all my mother's documents"
or "all docs that have an aadhaar number" without ever exposing the value
at the vector layer.

## Quick start

```bash
# 1. Start Postgres + Qdrant
docker compose up -d

# 2. Copy and edit .env (see .env.example for the full set of variables)
cp .env.example .env
# Then add your HUGGINGFACE_TOKEN and GROQ_API_KEY.

# 3. Start all services in the background
python activate.py --bg

# 4. Upload a document
curl -X POST -F "file=@data/test_fixtures/aadhaar.jpg" \
     -F "description_hint=Aadhaar card" \
     http://127.0.0.1:8084/api/upload

# 5. Verify health
python activate.py --status
```

## Tests

```bash
python -m pytest tests/
```

The test suite uses a live Postgres + Qdrant stack. Each test runs in its
own private Postgres schema and Qdrant collection (see
`tests/_live_stack.py`) so tests are fully isolated and run in any order.
If either dependency is unreachable, the affected tests are skipped with
a clear message.

## Project layout

```
src/
  document_mgmt_service/    Postgres + Qdrant + filesystem-backed service
  model_service/            Vision-LLM inference with HuggingFace → Groq fallback
  web_search_service/       Web search adapter
  validator_service/        Document template + structure validation
  orchestrator_service/     Chat persona, tool dispatch, sensitive-access gates
tests/                      pytest suite, one file per service plus shared helpers
scripts/                    install-wsl.sh for WSL setup
```

## Configuration

All configuration is read from environment variables (see `.env.example`).
The two most important:

- `HF_TOKEN` — HuggingFace inference API token. Free tier covers OCR and
  some classification. When credits run out the model service
  automatically falls through to Groq.
- `GROQ_API_KEY` — Groq API key. Required for the `qwen/qwen3.8-27b`
  fallback path on both the model service and the document service.
- `ENCRYPTION_MASTER_KEY` — Used to derive per-document Fernet keys for
  Qdrant payload encryption. Changing it invalidates all existing
  encrypted data.

### Chat LLM notes (Gemini function calling)

Gemini 2.5+ models attach an opaque `thoughtSignature` to every function
call and reject follow-up turns that replay the call without it (`400
INVALID_ARGUMENT`). The orchestrator therefore persists each call's
signature on the `ToolCall` record (including across restarts via the
session and confirmation stores) and replays it verbatim in the
conversation history. If the specific `thought_signature` 400 still
occurs (e.g. sessions stored before this behavior existed), the provider
retries once with prior function calls stripped from history. Qwen/Groq
use OpenAI-style APIs and are unaffected.

## Privacy and redaction guarantees

- Field values never enter Qdrant — only field *names* (as `field_pointers`).
- `text`, `description`, and `metadata` are Fernet-encrypted at rest in
  Qdrant with a per-document key.
- Document access for `SENSITIVE` documents requires an explicit
  `request_sensitive_access` approval.
- All sensitive-access attempts are recorded in `audit_events`.
- `description_safe` is a redacted, non-identifying summary safe to show
  in chat; the full `description` is only returned after approval.

