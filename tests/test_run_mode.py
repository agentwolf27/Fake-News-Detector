"""The documented commands start the app from anywhere, with the debugger off and without pandas.

Each server runs on a free port (never 8080), from a temporary working directory, in its own
process group so it can be stopped completely.
"""

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

from tests._support import CONTENT_FILES, FORMAT_FILES, ROOT, model_dir, sample_text

PYTHON = sys.executable
GUNICORN = str(Path(sys.executable).with_name("gunicorn"))
SCRUB = ("PORT", "FLASK_DEBUG", "FND_MODEL_DIR", "PYTHONPATH", "FLASK_APP")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def environment(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def call(port: int, method: str, path: str, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


@contextmanager
def server(cmd, port: int, **env):
    with tempfile.TemporaryDirectory(prefix="fnd-cwd-") as cwd, tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=environment(PORT=port, **env), start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=log)
        try:
            deadline = time.monotonic() + 45
            while True:
                if proc.poll() is not None:
                    log.seek(0)
                    raise AssertionError(f"{cmd} exited {proc.returncode}: {log.read().decode()[-800:]}")
                try:
                    call(port, "GET", "/health")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise AssertionError(f"{cmd} did not answer on {port}")
                    time.sleep(0.2)
            yield proc
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)


class DocumentedCommandTests(unittest.TestCase):
    def check_serving(self, port: int):
        status, body = call(port, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["models"], {"format_model": "available", "content_model": "available"})
        self.assertEqual(call(port, "GET", "/console")[0], 404)
        status, body = call(port, "POST", "/api/predict", {"text": sample_text("wire_report")})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["input_check"]["status"], "ok")
        status, body = call(port, "GET", "/?sample=sensational_dateline")
        self.assertEqual(status, 200)
        self.assertIn(b'data-state="disagreement"', body)

    def test_app_py_from_another_directory(self):
        port = free_port()
        with server([PYTHON, str(ROOT / "app" / "app.py")], port):
            self.check_serving(port)

    def test_main_py_run(self):
        port = free_port()
        with server([PYTHON, str(ROOT / "main.py"), "run"], port):
            self.check_serving(port)

    def test_gunicorn(self):
        port = free_port()
        cmd = [GUNICORN, "--chdir", str(ROOT), "--bind", f"127.0.0.1:{port}", "--workers", "2", "--preload", "app.app:app"]
        with server(cmd, port):
            self.check_serving(port)

    def test_the_debugger_is_on_only_when_asked(self):
        port = free_port()
        with server([PYTHON, str(ROOT / "app" / "app.py")], port, FLASK_DEBUG="0"):
            self.assertEqual(call(port, "GET", "/console")[0], 404)


class ModelDirectoryTests(unittest.TestCase):
    def test_format_files_only(self):
        port = free_port()
        with model_dir(*FORMAT_FILES) as directory:
            with server([PYTHON, str(ROOT / "app" / "app.py")], port, FND_MODEL_DIR=directory):
                status, body = call(port, "GET", "/health")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)["models"], {"format_model": "available", "content_model": "not_available"})
                status, body = call(port, "POST", "/api/predict", {"text": sample_text("wire_report")})
                self.assertEqual(status, 200)
                body = json.loads(body)
                self.assertNotIn("content_model", body["model_breakdown"])
                self.assertIsNone(body["input_check"]["coverage"]["content_model"])
                status, page = call(port, "GET", "/?sample=wire_report")
                self.assertIn("Format model only — content model not loaded", page.decode())

    def test_no_models(self):
        port = free_port()
        with model_dir() as directory:
            with server([PYTHON, str(ROOT / "app" / "app.py")], port, FND_MODEL_DIR=directory):
                status, body = call(port, "GET", "/health")
                self.assertEqual(status, 503)
                self.assertEqual(json.loads(body)["reason"], "fake_news_model.pkl not found")
                status, body = call(port, "POST", "/api/predict", {"text": "The council met."})
                self.assertEqual((status, json.loads(body)["code"]), (503, "model_unavailable"))
                self.assertEqual(call(port, "GET", "/")[0], 200)

    def test_content_files_only_is_unavailable(self):
        port = free_port()
        with model_dir(*CONTENT_FILES) as directory:
            with server([PYTHON, str(ROOT / "app" / "app.py")], port, FND_MODEL_DIR=directory):
                self.assertEqual(call(port, "GET", "/health")[0], 503)


class EnvironmentTests(unittest.TestCase):
    def test_the_app_runs_without_pandas(self):
        # A finder that makes pandas look uninstalled (ModuleNotFoundError, as in the Docker image).
        # Putting None in sys.modules is not the same: it raises a plain ImportError, which libraries
        # that probe for pandas do not expect.
        code = (
            "import sys\n"
            "class NoPandas:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'pandas' or name.startswith('pandas.'):\n"
            "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
            "sys.meta_path.insert(0, NoPandas())\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import app.app as a\n"
            "r = a.app.test_client().post('/api/predict', json={'text': 'The city council met on Monday to approve the budget.'})\n"
            "assert r.status_code == 200, r.status_code\n"
            "assert 'pandas' not in sys.modules or sys.modules['pandas'] is None\n"
            "print('ok')\n"
        )
        result = subprocess.run([PYTHON, "-c", code, str(ROOT)], cwd=tempfile.gettempdir(), env=environment(),
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertEqual(result.stdout.strip(), "ok")

    def test_requirements_pin_what_is_installed(self):
        pins = {}
        for line in (ROOT / "requirements.txt").read_text().splitlines():
            m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$", line.strip())
            if m:
                pins[m.group(1)] = m.group(2)
            else:
                self.assertTrue(not line.strip() or line.startswith("#"), line)
        self.assertEqual(len(pins), 14)
        self.assertNotIn("pandas", {k.lower() for k in pins})
        for name, pinned in pins.items():
            with self.subTest(package=name):
                self.assertEqual(version(name), pinned)

    def test_training_is_refused_and_the_models_are_untouched(self):
        served = [ROOT / "model" / n for n in (*FORMAT_FILES, *CONTENT_FILES)]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in served]
        for action in ("train", "all"):
            with self.subTest(action=action):
                result = subprocess.run([PYTHON, str(ROOT / "main.py"), action], capture_output=True, text=True,
                                        env=environment(), timeout=30)
                self.assertEqual(result.returncode, 1)
                self.assertIn("DECISIONS.md D1", result.stdout)
        self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in served], before)


if __name__ == "__main__":
    unittest.main()
