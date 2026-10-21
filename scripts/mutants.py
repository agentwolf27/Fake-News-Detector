#!/usr/bin/env python3
"""Prove the tests fail when the code is wrong.

    python scripts/mutants.py

1. Runs the whole suite; fails if anything fails, is skipped, or if no tests ran.
2. In a temporary copy of the repository, applies each mutant below — one deliberate bug at a
   time — and runs the test module that should catch it. Every mutant must make its module fail.
   A mutant whose target text is missing counts as a failure, so the list cannot rot silently.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVED = ["fake_news_model.pkl", "vectorizer.pkl", "content_model.pkl", "content_vectorizer.pkl"]
COPY_DIRS = ["app", "utils", "templates", "static", "samples", "tests", "docs", "scripts"]
COPY_FILES = ["main.py", "PLAN.md", "requirements.txt"]

# (name, file, text to replace, replacement, test module that must fail)
MUTANTS = [
    ("blend weight 0.6 -> 0.5", "utils/preprocess.py", "FORMAT_WEIGHT = 0.6", "FORMAT_WEIGHT = 0.5", "tests.test_golden"),
    ("FAKE/REAL columns swapped", "utils/models.py",
     'fake_index, real_index = classes.index("FAKE"), classes.index("REAL")',
     'fake_index, real_index = classes.index("REAL"), classes.index("FAKE")', "tests.test_golden"),
    ("naive tag regex", "utils/preprocess.py", '_TAG = re.compile(r"<[^>]+>")', '_TAG = re.compile(r"<[^<>]*>")',
     "tests.test_golden"),
    ("no-signal edge 3.0 -> 2.0", "utils/preprocess.py", "NO_SIGNAL_POINTS = 3.0", "NO_SIGNAL_POINTS = 2.0",
     "tests.test_thresholds"),
    ("no-signal edge 3.0 -> 4.0", "utils/preprocess.py", "NO_SIGNAL_POINTS = 3.0", "NO_SIGNAL_POINTS = 4.0",
     "tests.test_thresholds"),
    ("no-signal edge inclusive -> exclusive", "utils/preprocess.py",
     "r.points_from_empty <= NO_SIGNAL_POINTS", "r.points_from_empty < NO_SIGNAL_POINTS", "tests.test_thresholds"),
    ("coverage floor 0.5 -> 0.3", "utils/preprocess.py", "COVERAGE_FLOOR = 0.5", "COVERAGE_FLOOR = 0.3",
     "tests.test_thresholds"),
    ("half-up rounding -> truncation", "utils/present.py",
     "return int(math.floor(float(probability) * 100 + 0.5))", "return int(float(probability) * 100)",
     "tests.test_present"),
    ("complement rounded on its own", "utils/present.py",
     'return f"{pct_text(fake)}{NBSP}fake · {pct_text(100 - fake)}{NBSP}real"',
     'return f"{pct_text(fake)}{NBSP}fake · {pct_text(shown(1 - probability_fake))}{NBSP}real"', "tests.test_present"),
    ("lean edge 55 -> 56", "utils/present.py", "LEAN_FAKE_AT = 55", "LEAN_FAKE_AT = 56", "tests.test_present"),
    ("page trusts the API's confidence level", "utils/present.py",
     "    if value < 55:\n", "    if value < 56:\n", "tests.test_present"),
    ("debugger on by default", "app/app.py", 'os.environ.get("FLASK_DEBUG", "") == "1"',
     'os.environ.get("FLASK_DEBUG", "1") == "1"', "tests.test_run_mode"),
    ("text cap exclusive", "app/app.py", "    if size > MAX_TEXT_BYTES:\n        raise ApiError",
     "    if size >= MAX_TEXT_BYTES:\n        raise ApiError", "tests.test_errors"),
    ("exception text in the 500 body", "app/app.py",
     'jsonify({"error": "Internal server error.", "code": "internal_error"',
     'jsonify({"error": str(exc), "code": "internal_error"', "tests.test_errors"),
    ("models reloaded per request", "utils/preprocess.py",
     "    models = get_models()\n    if not models.ready:",
     "    from utils.models import load_models\n    models = load_models()\n    if not models.ready:",
     "tests.test_api_contract"),
    ("refusal shows the input's score", "utils/present.py",
     'f"({fake_real(_empty_blend(models))}), so the score',
     'f"({fake_real(fmt.fake)}), so the score', "tests.test_page"),
    ("disagreement ignored", "utils/present.py", "    if lf and lc and lf != lc:", "    if False:",
     "tests.test_samples"),
]


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def whole_suite() -> bool:
    result = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], ROOT)
    tail = result.stderr.strip().splitlines()[-3:]
    ran = re.search(r"Ran (\d+) tests?", result.stderr)
    count = int(ran.group(1)) if ran else 0
    skipped = re.search(r"skipped=\d+", result.stderr)
    ok = result.returncode == 0 and count > 0 and not skipped
    print(f"suite: {count} tests, exit {result.returncode}{', SKIPS' if skipped else ''} — {' / '.join(tail)}")
    return ok


def copy_repo(dest: Path) -> None:
    for name in COPY_DIRS:
        shutil.copytree(ROOT / name, dest / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in COPY_FILES:
        shutil.copy2(ROOT / name, dest / name)
    (dest / "model").mkdir()
    for name in SERVED:
        shutil.copy2(ROOT / "model" / name, dest / "model" / name)


def main() -> int:
    if not whole_suite():
        print("mutants: the unmutated suite is not green; fix that first")
        return 1
    failures = 0
    with tempfile.TemporaryDirectory(prefix="fnd-mutants-") as tmp:
        work = Path(tmp)
        copy_repo(work)
        for name, rel, old, new, module in MUTANTS:
            target = work / rel
            original = (ROOT / rel).read_text(encoding="utf-8")
            if original.count(old) != 1:
                print(f"  MISSING  {name}: target text found {original.count(old)} times in {rel}")
                failures += 1
                continue
            target.write_text(original.replace(old, new), encoding="utf-8")
            try:
                result = run([sys.executable, "-B", "-m", "unittest", module], work)
            finally:
                target.write_text(original, encoding="utf-8")
            if result.returncode == 0:
                print(f"  SURVIVED {name} ({module} still passes)")
                failures += 1
            else:
                print(f"  killed   {name} ({module})")
    print(f"mutants: {len(MUTANTS) - failures} of {len(MUTANTS)} killed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
