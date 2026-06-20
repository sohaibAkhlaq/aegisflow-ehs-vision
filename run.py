#!/usr/bin/env python3
"""
run.py — Convenience launcher for the Factory Compliance & Alert Escalation System.

Usage:
    python run.py                   # start server (auto-detects .env)
    python run.py --demo            # pre-load demo violations then start
    python run.py --port 8080       # custom port
    python run.py --no-reload       # production mode (no auto-reload)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Factory Compliance & Alert Escalation System"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--demo", action="store_true", help="Pre-load demo violations on start")
    parser.add_argument("--log-level", default="info", choices=["debug","info","warning","error"])
    return parser.parse_args()


def check_env():
    """Verify .env file exists and warn about missing API key."""
    root = Path(__file__).parent
    env_file = root / ".env"
    if not env_file.exists():
        print("[!] No .env file found -- copying from .env.example")
        example = root / ".env.example"
        if example.exists():
            env_file.write_text(example.read_text())
            print("   Created .env -- please set GEMINI_API_KEY for VLM fallback")
        else:
            print("   Warning: .env.example not found")
    else:
        content = env_file.read_text()
        if "your_gemini_api_key_here" in content or not content.strip():
            print("[i] GEMINI_API_KEY not set -- using offline/deterministic mode")


def print_banner(host: str, port: int):
    print()
    print("+--------------------------------------------------------------+")
    print("|            AegisFlow EHS Vision Platform  v1.0               |")
    print("|      Policy: KMP-OHS-POL-001 | Advanced Safety Automation     |")
    print("+--------------------------------------------------------------+")
    print(f"|  Dashboard:  http://{host}:{port:<38} |")
    print(f"|  API Docs:   http://{host}:{port}/docs{' '*33}|")
    print(f"|  WebSocket:  ws://{host}:{port}/ws{' '*36}|")
    print("+--------------------------------------------------------------+")
    print("|  Modules: Detection · Severity · Escalation · Reports · UI  |")
    print("+--------------------------------------------------------------+")
    print()


def main():
    args = parse_args()
    check_env()

    # Ensure outputs dir exists
    (Path(__file__).parent / "outputs").mkdir(exist_ok=True)
    (Path(__file__).parent / "data").mkdir(exist_ok=True)

    print_banner(args.host, args.port)

    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
