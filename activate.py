"""Chitragupta — Start all microservices.

Usage:
    python activate.py          Start all services (foreground)
    python activate.py --bg     Start all services in background
    python activate.py --stop   Stop all running services
    python activate.py --status Check which services are running

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


def ensure_dirs() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)


def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port(port: int) -> None:
    """Kill any process listening on the given port."""
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
    pids = load_pids()

    print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║     Chitragupta — Starting Services     ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════╝\n")

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
        cmd = [sys.executable, "-m", module] + args
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
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
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

            cmd = [sys.executable, "-m", module] + args
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
                        cmd = [sys.executable, "-m", module] + args
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


# ─── CLI ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chitragupta — start/stop/status all microservices",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bg", action="store_true", help="Start services in background")
    group.add_argument("--stop", action="store_true", help="Stop all running services")
    group.add_argument("--status", action="store_true", help="Check service health")
    args = parser.parse_args()

    if args.stop:
        stop_all()
    elif args.status:
        check_status()
    else:
        start_all(background=args.bg)


if __name__ == "__main__":
    main()
