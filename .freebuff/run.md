# Run Doc — Chitragupta (All Services)

## How to reproduce uncommitted artifacts

No build artifacts needed — this is a Python project. Install deps:

```bash
pip install fastapi uvicorn[standard] python-multipart requests python-dotenv Pillow groq
```

Copy `.env` from the main checkout if needed (contains API keys).

## How to run all services

```bash
python activate.py
```

This starts all 5 services in foreground (Ctrl+C to stop). For background:

```bash
python activate.py --bg
```

### Service Ports

| Port | Service |
|------|---------|
| 8080 | Document Management Service |
| 8081 | Model Service |
| 8082 | Web Search Service |
| 8083 | Validator Service |
| 8084 | Orchestrator + UI (FastAPI/Uvicorn) |

### Preview URL

http://localhost:8084/

### Detaching Orchestrator (Windows)

```powershell
powershell -NoProfile -Command "(Start-Process -FilePath 'python' -ArgumentList '-m','orchestrator_service','http' -RedirectStandardOutput '<log>' -RedirectStandardError '<log>.err' -WindowStyle Hidden -PassThru).Id"
```
