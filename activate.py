"""Chitragupta — Start all microservices.

Usage:
    python activate.py          Start all services (foreground)
    python activate.py --bg     Start all services in background
    python activate.py --dev    Developer mode: auto-restart on code changes
    python activate.py --stop   Stop all running services
    python activate.py --status Check which services are running

Platforms:
    • Windows (native)        — uses .venv\\Scripts\\python.exe
    • WSL  (any distro)       — auto-finds a Linux venv with project deps
                                installed. See scripts/install-wsl.sh.
    • Linux / macOS           — uses .venv/bin/python

WSL setup (one-time):
    cd /mnt/c/Users/MANISH/OneDrive/Attachments/Chitragupta
    ./scripts/install-wsl.sh
    python activate.py --bg

Services:
    :8080  Document Management Service
    :8081  Model Service
    :8082  Web Search Service
    :8083  Validator Service
    :8084  Orchestrator + UI (FastAPI/Uvicorn)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

# ─── Service definitions ──────────────────────────────────────────
SERVICES = [
    {
        "name": "Document Service",
        "module": "document_mgmt_service",
        "args": ["--http-only"],
        "port": 8080,
        "color": "\033[36m",   # cyan
    },
    {
        "name": "Model Service",
        "module": "model_service",
        "args": ["--http-only"],
        "port": 8081,
        "color": "\033[35m",   # magenta
    },
    {
        "name": "Web Search Service",
        "module": "web_search_service",
        "args": ["--http-only"],
        "port": 8082,
        "color": "\033[33m",   # yellow
    },
    {
        "name": "Validator Service",
        "module": "validator_service",
        "args": ["--http"],
        "port": 8083,
        "color": "\033[32m",   # green
    },
    {
        "name": "Orchestrator + UI",
        "module": "orchestrator_service",
        "args": ["http"],
        "port": 8084,
        "color": "\033[34m",   # blue
    },
]

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

LOG_DIR = Path(__file__).parent / "data" / "logs"
PID_FILE = Path(__file__).parent / "data" / ".service_pids"
# Per-platform virtualenv locations. On WSL, .venv-wsl/ is used (kept
# separate from the Windows .venv/ because the two Pythons can't share
# site-packages). On all other platforms we fall back to the standard .venv/.
VENV_CANDIDATES_WSL = [".venv-wsl", "venv", ".venv"]
VENV_CANDIDATES_OTHER = [".venv", "venv", ".venv-wsl"]


# ─── Platform helpers ─────────────────────────────────────────────
IS_WINDOWS = sys.platform.startswith("win")
IS_WSL = False
if not IS_WINDOWS and "microsoft" in platform.release().lower():
    IS_WSL = True
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"


def is_wsl() -> bool:
    """Return True if running inside Windows Subsystem for Linux."""
    return IS_WSL


def wsl_translate_path(path: str | os.PathLike) -> str:
    """Translate a Windows-style path (C:\\foo\\bar) to its WSL mount
    (/mnt/c/foo/bar). Native Linux paths are returned unchanged.
    """
    if not IS_WSL:
        return str(path)
    s = str(path)
    # Only translate if it actually looks like a Windows path
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if not m:
        return s
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _venv_python() -> str | None:
    """Return the absolute path to the Python interpreter inside the
    project's virtualenv, or None if no venv is found.

    Resolution order:
      1. ``CHITRAGUPTA_VENV`` env var (points at the venv directory itself)
      2. Project-local venvs: ``.venv-wsl`` first on WSL, then ``.venv``;
         on other platforms ``.venv`` first, then ``.venv-wsl``.

    On POSIX, candidates are validated to be:
      * A runnable file (ELF or proper shebang) — rejects Windows shims
      * Actually able to import a key project dep (``requests``) — rejects
        venvs whose ``python`` symlink resolves to system Python without
        our project packages installed (a real issue on /mnt/c mounts)
    """
    # 1) Explicit override (e.g. on WSL, where the venv may live on the
    #    fast native filesystem to avoid the slow /mnt/c 9P mount).
    override = os.environ.get("CHITRAGUPTA_VENV", "").strip()
    if override:
        base = Path(wsl_translate_path(override)) if IS_WSL else Path(override)
        for cand in (("Scripts", "python.exe"), ("bin", "python"), ("bin", "python3")):
            py = base / cand[0] / cand[1]
            if (
                py.exists()
                and _is_runnable_python(py)
                and _python_has_deps(str(py))
            ):
                return str(py)

    # 2) Walk the standard candidate list.
    root = _project_root()
    candidates = VENV_CANDIDATES_WSL if IS_WSL else VENV_CANDIDATES_OTHER
    for name in candidates:
        base = root / name
        if IS_WINDOWS:
            py = base / "Scripts" / "python.exe"
        else:
            py = base / "bin" / "python"
        if (
            py.exists()
            and _is_runnable_python(py)
            and _python_has_deps(str(py))
        ):
            return str(py)
        # On POSIX also accept `python3` inside bin/
        if not IS_WINDOWS:
            py3 = base / "bin" / "python3"
            if (
                py3.exists()
                and _is_runnable_python(py3)
                and _python_has_deps(str(py3))
            ):
                return str(py3)
    return None


def _is_runnable_python(path: Path) -> bool:
    """Return True if `path` is a real Linux executable (or any executable
    on Windows). We use this on POSIX to reject Windows-Python shim files
    that exist on the /mnt/c mount but would mis-behave if launched
    directly from WSL.
    """
    if IS_WINDOWS:
        return True
    try:
        # os.access X_OK is the cheapest signal; on /mnt/c the file is
        # typically a text file, so this will be False and we skip it.
        if not os.access(str(path), os.X_OK):
            return False
        # Sanity-check the first 4 bytes for an ELF magic number, or
        # a shebang that points at /usr/bin/env or /usr/bin/python*.
        with open(path, "rb") as f:
            head = f.read(4)
        if head[:4] == b"\x7fELF":
            return True
        # Allow real shebangs (binary scripts are fine; Windows shims
        # start with `#!C:\...` which is not a real interpreter).
        with open(path, "rb") as f:
            first_line = f.readline(256).decode("utf-8", "replace")
        if first_line.startswith("#!") and (
            "/python" in first_line or "/env " in first_line
        ):
            return True
        return False
    except OSError:
        return False


def _python_has_deps(path: str, modules: tuple[str, ...] = ("requests",)) -> bool:
    """Run `python -c "import X"` for each module in `modules` and return
    True only if all imports succeed. Used to disqualify a venv candidate
    that points at system Python (which exists at /usr/bin/python3 but
    lacks our project deps).
    """
    code = "import " + ", ".join(modules)
    try:
        r = subprocess.run(
            [path, "-c", code],
            capture_output=True,
            timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _resolve_python() -> str:
    """Pick the Python interpreter used to spawn every microservice.

    Priority:
      1. A project-local virtualenv (`.venv-wsl` on WSL, `.venv` elsewhere)
      2. The currently running interpreter
    """
    venv_py = _venv_python()
    if venv_py is not None:
        return venv_py

    # WSL-only last-resort fallbacks: the install script puts the venv
    # in $HOME/.local/chitragupta-wsl-venv (persistent, not in /tmp which
    # is wiped on WSL reboot) because /mnt/c is too slow for `pip install`.
    if IS_WSL:
        home = Path(os.path.expanduser("~"))
        candidates = [
            home / ".local" / "chitragupta-wsl-venv",
            Path("/tmp/chitragupta-wsl-venv"),
            Path("/tmp/wslvenv"),
        ]
        for base in candidates:
            for name in ("bin/python", "bin/python3"):
                py = base / name
                if py.exists() and _is_runnable_python(py) and _python_has_deps(str(py)):
                    return str(py)

    return sys.executable


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)


def start_docker() -> None:
    """Start PostgreSQL and Qdrant via Docker Compose if not running."""
    if IS_WSL:
        print(f"  {DIM}Detected WSL — using Docker Desktop's Linux engine over the WSL socket{RESET}")
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"  {DIM}Docker Compose not available — skipping infrastructure{RESET}")
            return

        running = set()
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                svc = json.loads(line)
                if svc.get("State") == "running":
                    running.add(svc.get("Service", ""))
            except json.JSONDecodeError:
                pass

        needed = {"postgres", "qdrant"}
        if needed.issubset(running):
            print(f"  \033[32m✓{RESET} Docker: PostgreSQL + Qdrant already running")
            return

        print(f"  \033[33m⟳{RESET} Docker: Starting PostgreSQL + Qdrant...")
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=True, timeout=60,
        )
        # Wait for health
        time.sleep(8)
        print(f"  \033[32m✓{RESET} Docker: PostgreSQL + Qdrant started")
    except FileNotFoundError:
        print(f"  {DIM}Docker not found — skipping infrastructure{RESET}")
    except Exception as e:
        print(f"  {DIM}Docker start failed: {e}{RESET}")


def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port(port: int) -> None:
    """Kill any process listening on the given port. Cross-platform."""
    if IS_WINDOWS:
        _kill_port_windows(port)
    else:
        _kill_port_posix(port)


def _kill_port_windows(port: int) -> None:
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            if f":{port}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                subprocess.run(
                    ["taskkill", "/F", "/PID", pid],
                    capture_output=True, timeout=5,
                )
    except Exception:
        pass


def _kill_port_posix(port: int) -> None:
    """Kill a listener on `port` on Linux/macOS/WSL.

    Tries `lsof` first (most reliable), then `fuser`, then a /proc scan.
    """
    pids: set[int] = set()

    # 1) lsof  (preferred)
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        for tok in out.stdout.split():
            if tok.strip().isdigit():
                pids.add(int(tok))
    except FileNotFoundError:
        pass
    except Exception:
        pass

    # 2) fuser  (fallback if lsof is missing)
    if not pids:
        try:
            out = subprocess.run(
                ["fuser", f"{port}/tcp"],
                capture_output=True, text=True, timeout=5,
            )
            for tok in out.stdout.split():
                digits = "".join(ch for ch in tok if ch.isdigit())
                if digits:
                    pids.add(int(digits))
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # 3) /proc scan (last-resort fallback; no extra deps)
    if not pids:
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open(f"/proc/{entry}/net/tcp", "r") as f:
                        content = f.read()
                    with open(f"/proc/{entry}/net/tcp6", "r") as f:
                        content += f.read()
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                # port in hex appears as :PORT in inode listings
                if f":{port:04X}" in content.upper():
                    pids.add(int(entry))
        except Exception:
            pass

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Try with sudo as a last resort (WSL/managed envs)
            subprocess.run(
                ["sudo", "kill", "-9", str(pid)],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def _popen_kwargs() -> dict:
    """Return the platform-appropriate Popen kwargs so backgrounded children
    can later be terminated cleanly via process group / job object.
    """
    if IS_WINDOWS:
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    # POSIX (Linux, macOS, WSL): detach into its own process group so we
    # can SIGTERM/SIGKILL the whole tree.
    return {"start_new_session": True}


def save_pids(pids: dict[str, int]) -> None:
    PID_FILE.write_text(json.dumps(pids))


def load_pids() -> dict[str, int]:
    if PID_FILE.exists():
        try:
            return json.loads(PID_FILE.read_text())
        except Exception:
            pass
    return {}


def clear_pids() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


# ─── Start services ───────────────────────────────────────────────
def start_all(background: bool = False) -> None:
    ensure_dirs()
    start_docker()
    pids = load_pids()

    print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║     Chitragupta — Starting Services     ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════╝\n")

    py = _resolve_python()
    py_marker = "  [WSL] " if IS_WSL else ""
    if py == sys.executable and _venv_python() is None and IS_WSL:
        print(f"  {DIM}{py_marker}Using system Python — many services will crash{RESET}")
        print(f"  {DIM}         Create a venv:  python3 -m venv .venv-wsl && "
              f"source .venv-wsl/bin/activate && pip install -r requirements.txt{RESET}")
    else:
        print(f"  {DIM}{py_marker}Python: {py}{RESET}")

    for svc in SERVICES:
        name = svc["name"]
        module = svc["module"]
        args = svc["args"]
        port = svc["port"]
        color = svc["color"]

        # Kill existing process on this port
        if is_port_open(port):
            print(f"  {color}⟳{RESET} {name} — port {port} in use, killing old process...")
            kill_port(port)
            time.sleep(0.5)

        # Build command
        cmd = [_resolve_python(), "-m", module] + args
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent / "src")

        log_file = LOG_DIR / f"{module}.log"
        log_err = LOG_DIR / f"{module}.err"

        if background:
            stdout = open(log_file, "w")
            stderr = open(log_err, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent / "src"),
                env=env,
                stdout=stdout,
                stderr=stderr,
                **_popen_kwargs(),
            )
            pids[module] = proc.pid
            print(f"  {color}✓{RESET} {name} — port {port} — PID {proc.pid} — {DIM}log: {log_file.name}{RESET}")
        else:
            print(f"  {color}▶{RESET} {name} — port {port}")

    if background:
        save_pids(pids)
        print(f"\n  {DIM}PIDs saved to {PID_FILE}{RESET}")
        print(f"  {DIM}Logs in {LOG_DIR}/{RESET}")
        print(f"\n  Run {BOLD}python activate.py --status{RESET} to check health\n")

    if not background:
        # Start all in background threads, then wait
        procs = []
        for svc in SERVICES:
            name = svc["name"]
            module = svc["module"]
            args = svc["args"]
            port = svc["port"]
            color = svc["color"]

            if is_port_open(port):
                kill_port(port)
                time.sleep(0.5)

            cmd = [_resolve_python(), "-m", module] + args
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parent / "src")

            log_file = LOG_DIR / f"{module}.log"
            log_err = LOG_DIR / f"{module}.err"

            stdout = open(log_file, "w")
            stderr = open(log_err, "w")

            proc = subprocess.Popen(
                cmd,
                cwd=str(Path(__file__).parent / "src"),
                env=env,
                stdout=stdout,
                stderr=stderr,
                **_popen_kwargs(),
            )
            procs.append((svc, proc, stdout, stderr))
            pids[module] = proc.pid
            print(f"  {color}✓{RESET} {name} — port {port} — PID {proc.pid}")
            time.sleep(1)  # Stagger starts

        save_pids(pids)

        # Wait for health checks
        print(f"\n  {DIM}Waiting for services to be ready...{RESET}")
        time.sleep(3)
        for svc, proc, stdout, stderr in procs:
            port = svc["port"]
            color = svc["color"]
            if is_port_open(port):
                print(f"  {color}●{RESET} {svc['name']} — {BOLD}ready{RESET} on :{port}")
            else:
                print(f"  {color}✗{RESET} {svc['name']} — {DIM}not yet ready on :{port}{RESET}")

        ui_url = f"http://localhost:{SERVICES[-1]['port']}/"
        api_url = f"http://localhost:{SERVICES[-1]['port']}/api/docs"
        print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
        print(f"{BOLD}║           All services running           ║{RESET}")
        print(f"{BOLD}╚══════════════════════════════════════════╝\n")
        print(f"  {BOLD}UI:{RESET}  {ui_url}")
        print(f"  {BOLD}API:{RESET}  {api_url}")
        print(f"\n  {DIM}Press Ctrl+C to stop all services{RESET}\n")

        # Handle graceful shutdown
        def shutdown(sig=None, frame=None):
            print(f"\n  {DIM}Stopping all services...{RESET}")
            for svc, proc, stdout, stderr in procs:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
                stdout.close()
                stderr.close()
                print(f"  {DIM}✓ {svc['name']} stopped{RESET}")
            clear_pids()
            print(f"\n  {BOLD}All services stopped.{RESET}")
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Keep alive
        try:
            while True:
                time.sleep(1)
                # Restart dead services
                for i, (svc, proc, stdout, stderr) in enumerate(procs):
                    if proc.poll() is not None:
                        color = svc["color"]
                        print(f"  {color}⟳{RESET} {svc['name']} died (code {proc.returncode}), restarting...")
                        module = svc["module"]
                        args = svc["args"]
                        cmd = [_resolve_python(), "-m", module] + args
                        env = os.environ.copy()
                        env["PYTHONPATH"] = str(Path(__file__).parent / "src")
                        new_log = LOG_DIR / f"{module}.log"
                        new_err = LOG_DIR / f"{module}.err"
                        new_stdout = open(new_log, "w")
                        new_err_fh = open(new_err, "w")
                        new_proc = subprocess.Popen(
                            cmd,
                            cwd=str(Path(__file__).parent / "src"),
                            env=env,
                            stdout=new_stdout,
                            stderr=new_err_fh,
                            **_popen_kwargs(),
                        )
                        procs[i] = (svc, new_proc, new_stdout, new_err_fh)
                        pids[module] = new_proc.pid
                        save_pids(pids)
                        print(f"  {color}✓{RESET} {svc['name']} restarted — PID {new_proc.pid}")
        except KeyboardInterrupt:
            shutdown()


# ─── Stop services ────────────────────────────────────────────────
def stop_all() -> None:
    print(f"\n  {BOLD}Stopping all Chitragupta services...{RESET}\n")

    for svc in SERVICES:
        port = svc["port"]
        name = svc["name"]
        color = svc["color"]
        if is_port_open(port):
            kill_port(port)
            print(f"  {color}✓{RESET} {name} stopped (port {port})")
        else:
            print(f"  {DIM}·{RESET} {name} not running (port {port})")

    clear_pids()
    print(f"\n  {BOLD}All services stopped.{RESET}\n")


# ─── Status ───────────────────────────────────────────────────────
def check_status() -> None:
    print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║        Chitragupta — Service Status     ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════╝\n")

    all_healthy = True
    for svc in SERVICES:
        port = svc["port"]
        name = svc["name"]
        color = svc["color"]

        if is_port_open(port):
            # Try health check — try root /healthz first, then /api/healthz
            try:
                import urllib.request
                data = None
                for path in ["/healthz", "/api/healthz"]:
                    try:
                        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2)
                        data = json.loads(resp.read().decode())
                        break
                    except Exception:
                        continue
                if data:
                    status = data.get("status", "unknown")
                    if status in ("ok", "healthy"):
                        print(f"  {color}●{RESET} {name:<25} :{port}  {BOLD}healthy{RESET}")
                    else:
                        print(f"  {color}◐{RESET} {name:<25} :{port}  {DIM}{status}{RESET}")
                else:
                    print(f"  {color}◐{RESET} {name:<25} :{port}  {DIM}running (health check failed){RESET}")
            except Exception:
                print(f"  {color}◐{RESET} {name:<25} :{port}  {DIM}running (health check failed){RESET}")
        else:
            print(f"  {color}✗{RESET} {name:<25} :{port}  {DIM}down{RESET}")
            all_healthy = False

    print()
    if all_healthy:
        print(f"  {BOLD}All services healthy ✓{RESET}")
    else:
        print(f"  {DIM}Some services are down — run {BOLD}python activate.py{RESET}{DIM} to start them{RESET}")
    print()


# ─── Dev mode: auto-restart on code changes ─────────────────────
def _module_for_path(filepath: str) -> str | None:
    """Map a changed .py file to its owning service module."""
    p = filepath.replace("\\", "/")
    for svc in SERVICES:
        mod = svc["module"]
        if mod.replace("_", "-") in p or mod in p:
            return mod
    return None


def start_dev() -> None:
    """Developer mode: watch for .py changes, auto-restart affected service."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print(f"  {BOLD}watchdog{RESET} not installed. Run: pip install watchdog")
        sys.exit(1)

    ensure_dirs()
    start_docker()

    print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║     Chitragupta — Developer Mode        ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════╝\n")
    print(f"  {DIM}Watching src/ for changes...{RESET}")
    print(f"  {DIM}Save any .py file → affected service restarts automatically{RESET}\n")

    # Start all services
    procs: list[tuple[dict, subprocess.Popen, object, object]] = []
    pids: dict[str, int] = {}
    restart_times: dict[str, float] = {}  # debounce

    def start_service(svc: dict) -> tuple[dict, subprocess.Popen, object, object]:
        name = svc["name"]
        module = svc["module"]
        args = svc["args"]
        port = svc["port"]
        color = svc["color"]

        if is_port_open(port):
            kill_port(port)
            time.sleep(0.5)

        cmd = [_resolve_python(), "-m", module] + args
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent / "src")
        env["CHITRAGUPTA_DEV"] = "1"
        env["ORCHESTRATOR_LOG_LEVEL"] = "DEBUG"
        env["DOCUMENT_SERVICE_LOG_LEVEL"] = "DEBUG"

        log_file = LOG_DIR / f"{module}.log"
        log_err = LOG_DIR / f"{module}.err"
        stdout = open(log_file, "w")
        stderr = open(log_err, "w")

        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).parent / "src"),
            env=env,
            stdout=stdout,
            stderr=stderr,
            **_popen_kwargs(),
        )
        pids[module] = proc.pid
        print(f"  {color}✓{RESET} {name} — port {port} — PID {proc.pid}")
        return (svc, proc, stdout, stderr)

    for svc in SERVICES:
        procs.append(start_service(svc))
        time.sleep(1)

    save_pids(pids)

    # Health check
    print(f"\n  {DIM}Waiting for services...{RESET}")
    time.sleep(3)
    for svc, proc, _, _ in procs:
        port = svc["port"]
        color = svc["color"]
        status = f"{BOLD}ready{RESET}" if is_port_open(port) else f"{DIM}not ready{RESET}"
        print(f"  {color}●{RESET} {svc['name']} — {status} on :{port}")

    ui_url = f"http://localhost:{SERVICES[-1]['port']}/"
    print(f"\n  {BOLD}UI:{RESET}  {ui_url}")
    print(f"  {BOLD}API:{RESET}  {ui_url}api/docs")
    print(f"\n  {DIM}Press Ctrl+C to stop{RESET}\n")

    # File watcher
    class _ChangeHandler(FileSystemEventHandler):
        def on_modified(self, event):
            self._handle(event)
        def on_created(self, event):
            self._handle(event)
        def _handle(self, event):
            if event.is_directory:
                return
            fp = event.src_path
            if not fp.endswith(".py"):
                return
            if "__pycache__" in fp:
                return
            if "_run_tests" in fp:
                return
            module = _module_for_path(fp)
            if module is None:
                return
            # Debounce: ignore if same service changed < 2s ago
            now = time.time()
            last = restart_times.get(module, 0)
            if now - last < 2:
                return
            restart_times[module] = now

            # Find and restart the service
            for i, (svc, proc, stdout, stderr) in enumerate(procs):
                if svc["module"] == module:
                    color = svc["color"]
                    rel_path = Path(fp).relative_to(Path(__file__).parent / "src")
                    print(f"\n  {color}⟳{RESET} {rel_path} changed — restarting {svc['name']}...")
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                    stdout.close()
                    stderr.close()
                    procs[i] = start_service(svc)
                    break

    src_dir = str(Path(__file__).parent / "src")
    observer = Observer()
    observer.schedule(_ChangeHandler(), path=src_dir, recursive=True)
    observer.start()

    # Keep alive + restart dead services
    def shutdown(sig=None, frame=None):
        print(f"\n  {DIM}Stopping...{RESET}")
        observer.stop()
        observer.join()
        for svc, proc, stdout, stderr in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            stdout.close()
            stderr.close()
        clear_pids()
        print(f"  {BOLD}All stopped.{RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
            for i, (svc, proc, stdout, stderr) in enumerate(procs):
                if proc.poll() is not None:
                    color = svc["color"]
                    print(f"  {color}⟳{RESET} {svc['name']} crashed (code {proc.returncode}), restarting...")
                    procs[i] = start_service(svc)
    except KeyboardInterrupt:
        shutdown()


# ─── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chitragupta — start/stop/status all microservices",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bg", action="store_true", help="Start services in background")
    group.add_argument("--dev", action="store_true", help="Developer mode: auto-restart on code changes")
    group.add_argument("--stop", action="store_true", help="Stop all running services")
    group.add_argument("--status", action="store_true", help="Check service health")
    args = parser.parse_args()

    if args.stop:
        stop_all()
    elif args.status:
        check_status()
    elif args.dev:
        start_dev()
    else:
        start_all(background=args.bg)


if __name__ == "__main__":
    main()
