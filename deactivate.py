"""Chitragupta — Stop all microservices.

Usage:
    python deactivate.py          Stop all services
    python deactivate.py --docker Also stop PostgreSQL + Qdrant containers
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

SERVICES = [
    {"name": "Document Service", "module": "document_mgmt_service", "port": 8080, "color": "\033[36m"},
    {"name": "Model Service", "module": "model_service", "port": 8081, "color": "\033[35m"},
    {"name": "Web Search Service", "module": "web_search_service", "port": 8082, "color": "\033[33m"},
    {"name": "Validator Service", "module": "validator_service", "port": 8083, "color": "\033[32m"},
    {"name": "Orchestrator + UI", "module": "orchestrator_service", "port": 8084, "color": "\033[34m"},
]

PID_FILE = Path(__file__).parent / "data" / ".service_pids"


def is_port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port(port: int) -> bool:
    """Kill any process listening on the given port. Cross-platform.

    Returns True if at least one process was found and signalled.
    """
    if sys.platform.startswith("win"):
        return _kill_port_windows(port)
    return _kill_port_posix(port)


def _kill_port_windows(port: int) -> bool:
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
                return True
    except Exception:
        pass
    return False


def _kill_port_posix(port: int) -> bool:
    """Kill a listener on `port` on Linux/macOS/WSL.

    Tries `lsof` first (most reliable), then `fuser`, then a /proc scan.
    Returns True if at least one PID was signalled.
    """
    import signal as _sig
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

    killed = False
    for pid in pids:
        try:
            os.kill(pid, _sig.SIGTERM)
            killed = True
        except ProcessLookupError:
            pass
        except PermissionError:
            subprocess.run(
                ["sudo", "kill", "-9", str(pid)],
                capture_output=True, timeout=5,
            )
            killed = True
        except Exception:
            pass
        try:
            os.kill(pid, _sig.SIGKILL)
        except Exception:
            pass
    return killed


def stop_services() -> None:
    print(f"\n{BOLD}╔══════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║     Chitragupta — Stopping Services     ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════╝\n")

    stopped = 0
    for svc in SERVICES:
        port = svc["port"]
        name = svc["name"]
        color = svc["color"]

        if is_port_open(port):
            killed = kill_port(port)
            if killed:
                print(f"  {color}✓{RESET} {name} stopped (port {port})")
                stopped += 1
            else:
                print(f"  {color}?{RESET} {name} — port {port} open but couldn't kill")
        else:
            print(f"  {DIM}·{RESET} {name} not running (port {port})")

    # Clean up PID file
    if PID_FILE.exists():
        PID_FILE.unlink()

    print(f"\n  {BOLD}{stopped}/{len(SERVICES)} services stopped.{RESET}\n")


def _docker_compose_down() -> bool:
    """Tear down the compose stack using the same fallback chain as
    activate.py: try v2, then v1, then v2 with an explicit -f flag.
    """
    compose_file = str(Path(__file__).resolve().parent / "docker-compose.yml")
    invocations = [
        ["docker", "compose", "down"],
        ["docker-compose", "down"],
        ["docker", "compose", "-f", compose_file, "down"],
    ]
    for cmd in invocations:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  {DIM}compose {cmd!r} failed: {e}{RESET}")
            continue
        if r.returncode == 0:
            return True
    return False


def stop_docker() -> None:
    print(f"\n  {DIM}Stopping Docker containers...{RESET}")
    if _docker_compose_down():
        print(f"  \033[32m✓{RESET} PostgreSQL + Qdrant stopped")
        return
    print(
        f"  \033[33m!{RESET} Could not stop the compose stack.\n"
        f"      Tried `docker compose down`, `docker-compose down`, and the\n"
        f"      explicit-file form. The Docker daemon may not be running\n"
        f"      or no compose project is loaded in the current dir."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop all Chitragupta services")
    parser.add_argument("--docker", action="store_true", help="Also stop PostgreSQL + Qdrant containers")
    args = parser.parse_args()

    stop_services()

    if args.docker:
        stop_docker()


if __name__ == "__main__":
    main()
