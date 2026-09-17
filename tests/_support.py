"""Shared helpers for the test suite (standard library only)."""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from app.app import app  # noqa: E402
from utils import models as models_module  # noqa: E402

COPY = json.loads((ROOT / "tests" / "copy.json").read_text(encoding="utf-8"))
GOLDEN = json.loads((ROOT / "tests" / "golden.json").read_text(encoding="utf-8"))
PREFIX = "WASHINGTON (Reuters) - "
FORMAT_FILES = ("fake_news_model.pkl", "vectorizer.pkl")
CONTENT_FILES = ("content_model.pkl", "content_vectorizer.pkl")


def client():
    return app.test_client()


def strict_json(data: bytes):
    """Parse JSON the way a strict client would: NaN or Infinity is an error."""

    def refuse(name):
        raise ValueError(f"non-standard JSON constant {name}")

    return json.loads(data, parse_constant=refuse)


def sample_text(sample_id: str) -> str:
    return json.loads((ROOT / "samples" / f"{sample_id}.json").read_text(encoding="utf-8"))["text"]


def sample_index() -> list[dict]:
    return json.loads((ROOT / "samples" / "index.json").read_text(encoding="utf-8"))


@contextmanager
def model_dir(*basenames: str):
    """A temporary model directory holding only the named served files."""
    with tempfile.TemporaryDirectory(prefix="fnd-models-") as tmp:
        for name in basenames:
            shutil.copy2(ROOT / "model" / name, Path(tmp) / name)
        yield Path(tmp)


@contextmanager
def serving(*basenames: str):
    """Serve only the named model files for the duration of the block."""
    with model_dir(*basenames) as directory:
        with models_module.use_models(models_module.load_models(directory)) as loaded:
            yield loaded


class PageParser(HTMLParser):
    """Collects what a reader sees: text outside <textarea>/<script>/<style>, the title, and the
    text-bearing attributes. Also records every data-state and the headline element's text."""

    ATTRS = ("title", "aria-label", "placeholder", "value", "alt")
    SKIP = {"textarea", "script", "style"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.attributes: list[str] = []
        self.states: list[str] = []
        self.codes: list[str] = []
        self.headlines: list[str] = []
        self.nums: list[str] = []
        self.result_text: list[str] = []
        self.textarea = ""
        self._stack: list[str] = []
        self._capture: str | None = None
        self._in_result = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self._stack.append(tag)
        if "data-state" in attrs:
            self.states.append(attrs["data-state"])
            if "data-code" in attrs:
                self.codes.append(attrs["data-code"])
        if tag == "section" and "result" in (attrs.get("class") or ""):
            self._in_result += 1
        elif tag == "section" and self._in_result:
            self._in_result += 1
        for name in self.ATTRS:
            if attrs.get(name):
                self.attributes.append(attrs[name])
        cls = attrs.get("class") or ""
        if tag == "h2" and "headline" in cls:
            self._capture = "headline"
            self.headlines.append("")
        elif tag == "span" and "nums" in cls.split():
            self._capture = "nums"
            self.nums.append("")
        elif tag == "textarea":
            self._capture = "textarea"

    def handle_endtag(self, tag):
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        if tag in ("h2", "span", "textarea"):
            self._capture = None
        if tag == "section" and self._in_result:
            self._in_result -= 1

    def handle_data(self, data):
        if self._capture == "headline":
            self.headlines[-1] += data
        elif self._capture == "nums":
            self.nums[-1] += data
        elif self._capture == "textarea":
            self.textarea += data
        if any(t in self.SKIP for t in self._stack):
            return
        self.visible.append(data)
        if self._in_result:
            self.result_text.append(data)


def parse_page(html: str) -> PageParser:
    parser = PageParser()
    parser.feed(html)
    parser.close()
    return parser


def squash(text: str) -> str:
    """Collapse layout whitespace only; the page's non-breaking spaces are content."""
    return re.sub(r"[ \t\r\n\f\v]+", " ", text).strip(" ")
