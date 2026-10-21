#!/usr/bin/env python3
"""Record what the app at 23207ce answered, before this branch changed any code.

    python scripts/capture_golden.py          # writes tests/golden.json
    python scripts/capture_golden.py --check  # re-derives it and fails if a byte differs

The old code is unpacked from `git archive` into a temporary directory and run there, in a
separate process with a scrubbed environment, so the edited code on this branch can never be the
thing that answers. The old `utils/preprocess.py` imports pandas, which this branch's
requirements drop, so this script needs the development venv that still has it. It is not part of
the unit suite; `tests/test_golden.py` compares the app against the committed file instead.
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_COMMIT = "23207ce"
OUT = ROOT / "tests" / "golden.json"
PICKLES = [
    "model/fake_news_model.pkl",
    "model/vectorizer.pkl",
    "model/content_model.pkl",
    "model/content_vectorizer.pkl",
]

WIRE = (
    "City officials said on Monday that the regional transit authority will extend late-night bus "
    "service on four routes starting next month, after a six-month pilot showed ridership between "
    "11 p.m. and 2 a.m. rose by about 18 percent. The authority's board approved the change in a 7-2 "
    "vote. Board chair Maria Lopez said the extension would cost roughly $2.4 million a year and "
    "would be funded through a mix of state grants and fare revenue. Two board members who voted "
    "against the plan said they were concerned about the long-term budget, noting that the state "
    "grant expires in 2028. The authority said it would publish quarterly ridership figures and "
    "review the program after one year. Riders at a public hearing last week largely supported the "
    "change, with several shift workers saying the pilot had cut their commute times. The new "
    "schedule takes effect on the first Sunday of next month."
)
HYPE = (
    "You won't believe what they're hiding from you! Insiders have finally EXPOSED the shocking "
    "truth that the mainstream media refuses to report. Thousands of people are waking up to the "
    "fact that the government has been secretly adding chemicals to tap water to control our minds, "
    "and the so-called experts are terrified. Share this before it gets deleted!!! A brave "
    "whistleblower who asked not to be named says the elites have known for years and are laughing "
    "at us. Doctors are furious because this simple trick protects your family completely. Wake up, "
    "people, and do your own research before it is too late. They will try to silence this story, "
    "but the truth cannot be stopped. Everything you have been told is a lie and the proof is right "
    "in front of you."
)
DATELINE = "WASHINGTON (Reuters) - "

INPUTS = {
    "harvard_diabetes": "Scientists at Harvard have discovered a new treatment for diabetes that shows promising results in clinical trials.",
    "area51_aliens": "BREAKING: Aliens have landed in Area 51 and are demanding to speak with our planet's manager!",
    "stock_market": "The stock market closed up 2% today following positive economic indicators and strong quarterly earnings reports.",
    "weird_fruit": "Eating this one weird fruit will cure all diseases and make you immortal, doctors hate this trick!",
    "mit_water": "Scientists at MIT develop new water purification technology using solar energy",
    "birds_drones": "Government plans to replace all birds with surveillance drones",
    "central_bank": "The central bank left interest rates unchanged, citing steady inflation and a stable labor market.",
    "central_bank_dateline": DATELINE + "The central bank left interest rates unchanged, citing steady inflation and a stable labor market.",
    "gibberish": "zzzz qqqq xxxx",
    "html_wrapped": "<p>The city council approved a new <b>budget</b> on Tuesday.</p> http://example.com",
    "wire_article": WIRE,
    "wire_article_dateline": DATELINE + WIRE,
    "sensational_article": HYPE,
    "sensational_article_dateline": DATELINE + HYPE,
}

# Inputs for the HTML-stripping regex: what the old clean_text made of each.
REGEX_PROBES = [
    "x < y < z > w",
    "<p>a</p> b > c",
    "a <b> c < d",
    "<<x>> y",
    "no tags here",
    "< not a tag",
    "5 > 3 and 2 < 4",
    "<a href='x'>link</a> and <br/> break",
    "tail < open",
    ">> leading",
    "<<<>>>",
    "mid <x> and <y",
]

CHILD = r"""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
root = os.getcwd()
import app.app as old_app
import utils.preprocess as old_pre
for mod in (old_app, old_pre):
    if not os.path.realpath(mod.__file__).startswith(os.path.realpath(root) + os.sep):
        sys.exit(f"imported {mod.__name__} from {mod.__file__}, not from the archive")
spec = json.load(sys.stdin)
client = old_app.app.test_client()
out = {"inputs": {}, "regex_probes": {}}
for name, text in spec["inputs"].items():
    rec = {"text": text}
    for route, key in (("/api/predict", "predict"), ("/api/predict/simple", "simple")):
        r = client.post(route, json={"text": text})
        body = r.get_json()
        if r.status_code != 200 or body.get("prediction") not in ("FAKE", "REAL"):
            sys.exit(f"{name} {route}: status {r.status_code}, body {body!r}")
        if "input_check" in body:
            sys.exit(f"{name} {route}: the old app returned input_check")
        rec[key] = {"status": r.status_code, "body": body}
    out["inputs"][name] = rec
for probe in spec["regex_probes"]:
    out["regex_probes"][probe] = old_pre.clean_text(probe)
json.dump(out, sys.stdout)
"""


def derive() -> bytes:
    with tempfile.TemporaryDirectory(prefix="fnd-golden-") as tmp:
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", SOURCE_COMMIT, "app", "utils", "templates", "static", *PICKLES],
            check=True,
            capture_output=True,
        ).stdout
        tar_path = Path(tmp) / "src.tar"
        tar_path.write_bytes(archive)
        src = Path(tmp) / "src"
        src.mkdir()
        with tarfile.open(tar_path) as tar:
            tar.extractall(src, filter="data")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "FLASK_DEBUG", "PORT", "FND_MODEL_DIR")}
        spec = json.dumps({"inputs": INPUTS, "regex_probes": REGEX_PROBES})
        child = subprocess.run(
            [sys.executable, "-B", "-c", CHILD],
            cwd=src,
            env=env,
            input=spec,
            capture_output=True,
            text=True,
        )
        if child.returncode != 0:
            if "No module named 'pandas'" in child.stderr:
                sys.exit("capture_golden: the code at 23207ce needs pandas; run this with the development venv that has it")
            sys.exit(f"capture_golden: the archived app failed:\n{child.stderr.strip()}")
    data = json.loads(child.stdout)
    record = {
        "source_commit": SOURCE_COMMIT,
        "note": "Bodies and statuses from app/app.py at 23207ce through Flask's test client. Never regenerated on this branch.",
        "inputs": data["inputs"],
        "regex_probes": data["regex_probes"],
    }
    return (json.dumps(record, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> int:
    fresh = derive()
    if "--check" in sys.argv[1:]:
        if not OUT.exists():
            print(f"capture_golden: {OUT.relative_to(ROOT)} does not exist")
            return 1
        if OUT.read_bytes() != fresh:
            print(f"capture_golden: {OUT.relative_to(ROOT)} differs from what 23207ce answers now")
            return 1
        print(f"capture_golden: {OUT.relative_to(ROOT)} matches 23207ce byte for byte ({len(INPUTS)} inputs, {len(REGEX_PROBES)} regex probes)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(fresh)
    print(f"capture_golden: wrote {OUT.relative_to(ROOT)} ({len(fresh)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
