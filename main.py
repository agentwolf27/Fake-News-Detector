#!/usr/bin/env python3
"""Fake News Detector — command-line entry point.

    python main.py run        start the web app on http://127.0.0.1:8080 (PORT to move it)
    python main.py test       run the test suite
    python main.py samples    score the demo texts in samples/
    python main.py download   fetch the Kaggle dataset (needs Kaggle credentials)
    python main.py train      refused: see DECISIONS.md D1
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_app() -> None:
    """Replace this process with the web app, using the same interpreter."""
    app = ROOT / "app" / "app.py"
    os.execv(sys.executable, [sys.executable, str(app)])


def run_tests() -> int:
    return subprocess.call([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=ROOT)


def score_samples() -> int:
    sys.path.insert(0, str(ROOT))
    from utils import present
    from utils.preprocess import blend, read

    for entry in json.loads((ROOT / "samples" / "index.json").read_text(encoding="utf-8")):
        text = json.loads((ROOT / "samples" / entry["file"]).read_text(encoding="utf-8"))["text"]
        reading = read(text)
        fake, _ = blend(reading)
        print(f"{entry['label']:36} {reading.status:15} {present.fake_real(fake)}")
    return 0


def download_dataset() -> int:
    return subprocess.call([sys.executable, str(ROOT / "model" / "download_dataset.py")], cwd=ROOT)


def refuse_training() -> int:
    print(
        "Training is switched off in this version.\n"
        "The served models (model/fake_news_model.pkl, model/content_model.pkl) were trained with scripts\n"
        "and data that are not in this repository; model/train_model.py would overwrite them with a\n"
        "different, older model. See DECISIONS.md D1."
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Fake News Detector")
    parser.add_argument("action", choices=["run", "test", "samples", "download", "train", "all"])
    action = parser.parse_args().action
    if action == "run":
        run_app()
        return 0  # not reached
    if action == "test":
        return run_tests()
    if action == "samples":
        return score_samples()
    if action == "download":
        return download_dataset()
    return refuse_training()  # train, all


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__.strip())
        sys.exit(0)
    sys.exit(main())
