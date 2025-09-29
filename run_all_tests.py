#!/usr/bin/env python3
"""
Run all backend tests sequentially for the Sherpa API.

Order:
1) Unit tests (no server):
   - tests/test_agent_system.py
   - tests/test_style_system.py
2) Start local API server (uvicorn app.main:app)
3) Live API tests (require server):
   - tests/test_api.py
   - tests/test_conversations_api.py

Usage:
  python run_all_tests.py                # default host:port 127.0.0.1:8000
  python run_all_tests.py --host 0.0.0.0 --port 8000

Notes:
- No external services are required; LLM calls are handled with fallbacks.
- Ensure dependencies are installed (see sherpa/README.md) and uvicorn is available.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def _python_exec() -> str:
    return sys.executable or "python3"


def run_cmd(cmd, cwd: str | None = None, env: dict | None = None, name: str = "") -> int:
    print(f"\n=== Running: {name or ' '.join(cmd)} ===")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    proc.wait()
    code = proc.returncode or 0
    print(f"=== Exit: {name or cmd[0]} -> {code} ===\n")
    return code


def wait_for_health(url: str, timeout_s: int = 20) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    failures: list[tuple[str, int]] = []

    # 1) Unit tests (no server)
    for script in ["tests/test_agent_system.py", "tests/test_style_system.py"]:
        code = run_cmd([_python_exec(), script], cwd=base_dir, name=script)
        if code != 0:
            failures.append((script, code))

    # 2) Start server
    server_cmd = [
        _python_exec(),
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        args.port,
    ]
    print("\n=== Starting Sherpa API server ===")
    server_env = os.environ.copy()
    server = subprocess.Popen(server_cmd, cwd=base_dir)
    try:
        ok = wait_for_health(f"http://{args.host}:{args.port}/healthz", timeout_s=30)
        if not ok:
            print("Server failed to become healthy within timeout.")
            server.terminate()
            server.wait(timeout=5)
            failures.append(("uvicorn(app.main:app)", 1))
            # Still run the summary below
        else:
            # 3) Live API tests (require server)
            for script in ["tests/test_api.py", "tests/test_conversations_api.py"]:
                code = run_cmd([_python_exec(), script], cwd=base_dir, name=script)
                if code != 0:
                    failures.append((script, code))
    finally:
        # Stop server
        print("\n=== Stopping Sherpa API server ===")
        try:
            if server.poll() is None:
                if os.name == "nt":
                    server.terminate()
                else:
                    os.kill(server.pid, signal.SIGTERM)
                server.wait(timeout=10)
        except Exception:
            pass

    # Summary
    if failures:
        print("\n=== Test Summary: FAIL ===")
        for name, code in failures:
            print(f"- {name}: exit {code}")
        sys.exit(1)
    else:
        print("\n=== Test Summary: PASS ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
