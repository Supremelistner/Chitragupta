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

## Managing your documents (list, open, delete)

The chat UI's **Settings → My Documents** panel lists every document you
have saved and lets you open or delete each one. All three actions have
dedicated, user-scoped orchestrator endpoints (the JWT owner is the only
account that can see or touch its documents):

```bash
# List the signed-in user's stored documents (names, type, versions, expiry)
curl http://127.0.0.1:8084/api/documents -H "Authorization: Bearer ***"

# Open the original file. Sensitive documents are gated: the first call
# returns 202 {"status":"requires_confirmation"}; repeat with ?approve=true
# (after the user confirms) to stream the original file back.
curl http://127.0.0.1:8084/api/documents/<id>/versions/1/retrieve -H "Authorization: Bearer ***"
curl "http://127.0.0.1:8084/api/documents/<id>/versions/1/retrieve?approve=true" -H "Authorization: Bearer ***"

# Delete. Omit ?version to delete the whole document (all versions);
# pass ?version=N to delete only that version. Cascades across Postgres,
# file storage, Qdrant vectors, and the device-sync hub.
curl -X DELETE http://127.0.0.1:8084/api/documents/<id> -H "Authorization: Bearer ***"
curl -X DELETE "http://127.0.0.1:8084/api/documents/<id>?version=2" -H "Authorization: Bearer ***"
```

Retrieve is driven by code (the document id comes from your own document
list, never the LLM), so it reuses the proven approval gate without any
risk of a hallucinated id. `delete_document` is a registered tool but is
**not** exposed to the chat LLM — deletion only happens through explicit
UI action.

**Chats vs documents are independent.** Deleting a chat session
(the trash icon in the sidebar, or `DELETE /api/sessions/{id}`) removes
only the conversation record — your uploaded documents are untouched.
Deleting a document never affects your chats.

## Quick start

**Prerequisite: Docker must be running.** Postgres + Qdrant run in Docker
and there is no in-memory/SQLite fallback. On **Windows/WSL**, `activate.py`
auto-starts Docker Desktop and waits for the daemon if it is down. On
**Linux/macOS**, start the daemon yourself first (`sudo systemctl start
docker` / `open -a Docker`).

```bash
# 1. Start Postgres + Qdrant (skipped automatically if already running;
#    activate.py --bg also does this for you)
docker compose up -d

# 2. Copy and edit .env (see .env.example for the full set of variables)
cp .env.example .env
# Then add your HUGGINGFACE_TOKEN and GROQ_API_KEY.
# For multi-user login, also set a long random CHITRAGUPTA_JWT_SECRET:
#   python -c "import secrets; print(secrets.token_hex(32))"

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
- `CHITRAGUPTA_JWT_SECRET` — Signs multi-user login tokens (email +
  password → JWT, 7-day TTL with sliding refresh). Required for
  `/api/auth/*`; use a long random string. Must match on every
  instance that verifies logins.

## Multi-user and device sync (local-primary)

Each login owns its documents end to end: Postgres rows and Qdrant
points carry a `user_id` (migration 006 backfills old rows as
`__local__`), Qdrant payloads encrypt under a per-user key domain, and
all list/search calls are tenant-filtered.

```bash
# Register, then log in (UI login screen calls the same endpoints)
curl -X POST http://127.0.0.1:8084/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"at-least-8-chars"}'
# → {"user_id": "...", "token": "<JWT>"}

# Authenticated calls carry the token; uploads without one fall back
# to the legacy __local__ owner so old clients keep working
curl http://127.0.0.1:8084/api/me -H "Authorization: Bearer <JWT>"
```

Device backup/sync is docs-only (chat sessions never leave the device):

- Uploads push gzipped blobs + a manifest entry to the filesystem hub
  at `data/global-blobs/<user_id>/` (Docker phase; same interface will
  target shared Postgres/Qdrant in production via `GLOBAL_POSTGRES_DSN`
  / `GLOBAL_QDRANT_URL`).
- Switching users wipes the device cache (`POST /api/auth/logout`
  with `{"wipe": true}`) and pulls from scratch:
  `POST /api/sync/pull` to list, `POST /api/sync/restore` for a
  version-preserving restore (same document IDs + versions).

Runtime data paths (services run with cwd `src/`, so relative paths
resolve there): sessions/confirmations/users under `src/data/`;
file blobs under `data/files/`; the sync hub under `data/global-blobs/`
(absolute, cwd-independent).

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
- **Postgres storage boundary:** the structured `extracted_fields` column
  (the full PII values) is stored as **plaintext JSONB** and GIN-indexed so
  it stays queryable — it is **not** encrypted at rest in the database.
  Confidentiality for these values rests on database access control (the
  DB runs in local Docker, not exposed publicly), the two-step reveal gate
  (`confirm: true` required), and audit logging — not on column encryption.
  Encrypting this column at rest is tracked as future work (audit CG-003).
- Document access for `SENSITIVE` documents requires an explicit
  `request_sensitive_access` approval.
- All sensitive-access attempts are recorded in `audit_events`.
- `description_safe` is a redacted, non-identifying summary safe to show
  in chat; the full `description` is only returned after approval.

