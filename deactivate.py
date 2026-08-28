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
    """Kill any process listening on the given port. Returns True if killed."""
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


def stop_docker() -> None:
    print(f"\n  {DIM}Stopping Docker containers...{RESET}")
    try:
        result = subprocess.run(
            ["docker", "compose", "down"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"  \033[32m✓{RESET} PostgreSQL + Qdrant stopped")
        else:
            print(f"  \033[33m!{RESET} Docker compose returned code {result.returncode}")
    except FileNotFoundError:
        print(f"  {DIM}Docker not found — skipping{RESET}")
    except Exception as e:
        print(f"  \033[33m!{RESET} Docker stop failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stop all Chitragupta services")
    parser.add_argument("--docker", action="store_true", help="Also stop PostgreSQL + Qdrant containers")
    args = parser.parse_args()

    stop_services()

    if args.docker:
        stop_docker()


if __name__ == "__main__":
    main()
