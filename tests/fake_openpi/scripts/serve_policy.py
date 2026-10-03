"""Stand-in for openpi's scripts/serve_policy.py: serves the deterministic fake policy on --port."""

import os
import pathlib
import sys

repo = pathlib.Path(__file__).resolve().parents[3]
port = sys.argv[sys.argv.index("--port") + 1]
python = str(repo / ".venv" / "bin" / "python")
os.execv(python, [python, str(repo / "tests" / "fake_policy_server.py"), "--port", port])
