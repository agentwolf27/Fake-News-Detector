#!/usr/bin/env python3
"""Save exactly what the live server served for the recording, so a later edit can't go unnoticed.

    python scripts/rehearse.py --record --window 1280x712
    python scripts/rehearse.py --record --base http://127.0.0.1:8080 --window 1280x712

Run it right after a take, against the server that was recorded (started after the last edit,
debugger off). It fetches every page, asset and API response the shot list shows and writes
docs/rehearsal.json: a sha256 per response, the page state, and the whole percents shown.
tests/test_rehearsal.py renders the same requests through the current code and fails if any byte
differs — the take no longer matches the code, so re-record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "rehearsal.json"

# (method, path, JSON body file or None) — the shot list, in order.
SHOTS = [
    ("GET", "/", None),
    ("GET", "/?sample=wire_report", None),
    ("GET", "/?sample=sensational_post", None),
    ("GET", "/?sample=sensational_dateline", None),
    ("GET", "/?sample=headline_only", None),
    ("GET", "/static/style.css", None),
    ("GET", "/static/app.js", None),
    ("GET", "/api/models", None),
    ("POST", "/api/predict", "samples/wire_report.json"),
]


def fetch(base: str, method: str, path: str, body_file: str | None):
    data = (ROOT / body_file).read_bytes() if body_file else None
    request = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def describe(body: bytes) -> dict:
    """The page state and shown percents, for a person reading the record."""
    sys.path.insert(0, str(ROOT))
    from tests._support import parse_page  # noqa: E402

    page = parse_page(body.decode("utf-8"))
    return {"state": page.states[0] if page.states else None, "headline": page.headlines[0].strip() if page.headlines else None,
            "shown": page.nums}


def record(base: str, window: str, note: str | None) -> int:
    entries = []
    for method, path, body_file in SHOTS:
        try:
            status, body = fetch(base, method, path, body_file)
        except OSError as err:
            print(f"rehearse: could not reach {base}{path}: {err}")
            return 1
        entry = {"method": method, "path": path, "body_file": body_file, "status": status,
                 "sha256": hashlib.sha256(body).hexdigest()}
        if path == "/" or path.startswith("/?"):
            entry.update(describe(body))
        entries.append(entry)
        print(f"  {status} {method} {path}  {entry.get('state') or ''} {' | '.join(entry.get('shown') or [])}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "base": base,
        "window": window,
        "note": note,
        "shots": entries,
    }, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rehearse: wrote {OUT.relative_to(ROOT)} ({len(entries)} responses)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", action="store_true", help="fetch the shot list and write docs/rehearsal.json")
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--window", required="--record" in sys.argv, help="the browser's inner size, e.g. 1280x712")
    parser.add_argument("--note", default=None)
    args = parser.parse_args()
    if not args.record:
        parser.print_help()
        return 2
    return record(args.base, args.window, args.note)


if __name__ == "__main__":
    sys.exit(main())
