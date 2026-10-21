"""Real inputs reach every named state, and the page shows only what the formatter allows.

Expected copy comes from tests/copy.json, transcribed from PLAN.md; numbers are recomputed from
the raw scores through utils/present.py.
"""

import re
import unittest

from flask import render_template

from tests._support import COPY, FORMAT_FILES, ROOT, app, client, parse_page, sample_index, sample_text, serving, squash
from utils import present
from utils.models import get_models
from utils.preprocess import MAX_TEXT_BYTES, ModelReading, Reading, read

DECIMAL_PERCENT = re.compile(r"\d[.,]\d+\s*%")
ENUM_WORDS = re.compile(r"\b(HIGH|MEDIUM|LOW|VERY_LOW|FAKE|REAL|None|nan|NaN|True|False)\b")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\U0001F000-\U0001F2FF]")
BANNED_PHRASES = ("consider carefully", "calibrated", "accurate", "verified", "Likely Fake", "Likely Real")

ALL_HEADLINES = (
    [COPY["headlines"][k] for k in ("first_load", "no_signal", "unrecognised", "url_only", "disagreement")]
    + COPY["headlines"]["verdict"]
    + list(COPY["headlines"]["error"].values())
)

LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
         "dolore magna aliqua.")


def expected_numbers(text: str) -> list[str]:
    r = read(text)
    blended = 0.6 * r.format.fake + 0.4 * r.content.fake
    return [present.fake_real(blended), present.fake_real(r.format.fake), present.fake_real(r.content.fake)]


class StateTests(unittest.TestCase):
    def setUp(self):
        self.client = client()

    def get_page(self, response, state, headline, status=200):
        self.assertEqual(response.status_code, status)
        html = response.get_data(as_text=True)
        page = parse_page(html)
        self.assertEqual(page.states, [state])
        self.assertEqual(len(page.headlines), 1)
        self.assertEqual(page.headlines[0].strip(), headline)
        for other in ALL_HEADLINES:
            if other != headline:
                self.assertNotEqual(page.headlines[0].strip(), other)
        self.assert_clean(page)
        return page

    def assert_clean(self, page):
        seen = squash(" ".join(page.visible + page.attributes))
        self.assertIsNone(DECIMAL_PERCENT.search(seen), seen)
        self.assertIsNone(ENUM_WORDS.search(seen), ENUM_WORDS.search(seen))
        self.assertIsNone(EMOJI.search(seen))
        for phrase in BANNED_PHRASES:
            self.assertNotIn(phrase.lower(), seen.lower())
        for always in (COPY["always"]["lede"], COPY["always"]["scope"], COPY["always"]["samples_intro"],
                       COPY["always"]["footer_source"]):
            self.assertIn(always, seen)
        self.assertIn(COPY["always"]["placeholder"], page.attributes)

    def result(self, page) -> str:
        return squash(" ".join(page.result_text))

    def test_first_load(self):
        page = self.get_page(self.client.get("/"), "first_load", COPY["headlines"]["first_load"])
        self.assertIn(COPY["bodies"]["first_load"], self.result(page))
        self.assertNotIn("%", self.result(page))
        self.assertEqual(page.textarea, "")

    def test_samples_fill_the_box_and_score(self):
        labels = [s["label"] for s in sample_index()]
        self.assertEqual(labels, COPY["always"]["sample_labels"])
        for sample in sample_index():
            with self.subTest(sample=sample["id"]):
                response = self.client.get(f"/?sample={sample['id']}")
                page = parse_page(response.get_data(as_text=True))
                self.assertEqual(page.states, [sample["expect"]])
                self.assertEqual(page.textarea, sample_text(sample["id"]))

    def test_verdict(self):
        text = sample_text("wire_report")
        page = self.get_page(self.client.post("/", data={"text": text}), "verdict", "Reads like real reporting")
        self.assertEqual(page.nums, expected_numbers(text))
        body = self.result(page)
        self.assertIn("Both models lean real.", body)
        self.assertIn(f"Read {present.words(len(text.split()))}.", body)
        for row in ("blend", "format", "content", "format_note", "content_note"):
            self.assertIn(COPY["rows"][row], body)
        self.assertIn(COPY["always"]["scope"], body)
        self.assertNotIn("— leans", body)  # lean labels belong to the disagreement card only

    def test_verdict_that_leans(self):
        text = sample_text("sensational_post")
        page = self.get_page(self.client.post("/", data={"text": text}), "verdict", "Leans fake")
        self.assertEqual(page.nums, expected_numbers(text))
        self.assertIn("Both models lean fake.", self.result(page))

    def test_disagreement(self):
        text = sample_text("sensational_dateline")
        page = self.get_page(self.client.post("/", data={"text": text}), "disagreement", COPY["headlines"]["disagreement"])
        self.assertEqual(page.nums, expected_numbers(text))
        body = self.result(page)
        self.assertIn("— leans real", body)
        self.assertIn("— leans fake", body)
        self.assertIn(COPY["bodies"]["disagreement"], body)

    def test_no_signal_shows_only_the_empty_text_score(self):
        page = self.get_page(self.client.get("/?sample=headline_only"), "no_signal", COPY["headlines"]["no_signal"])
        body = self.result(page)
        self.assertIn(COPY["bodies"]["no_signal_both"], body)
        self.assertEqual(body.count("%"), 2)  # only "(59 % fake · 41 % real)"
        self.assertIn(COPY["actions"]["edit"], body)
        self.assertEqual(page.nums, [])

    def test_link_only(self):
        page = self.get_page(self.client.post("/", data={"text": "https://example.com/news/2026/story.html"}),
                             "url_only", COPY["headlines"]["url_only"])
        self.assertIn(COPY["bodies"]["url_only"], self.result(page))
        self.assertNotIn("%", self.result(page))

    def test_unrecognised(self):
        page = self.get_page(self.client.post("/", data={"text": LOREM}), "unrecognised", COPY["headlines"]["unrecognised"])
        self.assertIn(COPY["bodies"]["unrecognised"], self.result(page))
        self.assertIn(COPY["actions"]["edit"], self.result(page))
        self.assertNotIn("%", self.result(page))

    def test_empty(self):
        page = self.get_page(self.client.post("/", data={"text": "  "}), "error",
                             COPY["headlines"]["error"]["text_empty"], status=400)
        self.assertIn(COPY["bodies"]["text_empty"], self.result(page))
        self.assertNotIn("%", self.result(page))

    def test_too_large_offers_the_draft_back(self):
        response = self.client.post("/", data={"text": "a " * (MAX_TEXT_BYTES // 2 + 1)})
        page = self.get_page(response, "error", COPY["headlines"]["error"]["too_large"], status=413)
        self.assertIn(COPY["actions"]["restore"], response.get_data(as_text=True))
        self.assertIn("data-restore", response.get_data(as_text=True))

    def test_model_unavailable(self):
        with serving():
            self.get_page(self.client.get("/?sample=wire_report"), "error",
                          COPY["headlines"]["error"]["model_unavailable"], status=503)

    def test_format_only(self):
        with serving(*FORMAT_FILES):
            page = self.get_page(self.client.get("/?sample=wire_report"), "verdict", "Reads like real reporting")
            body = self.result(page)
            self.assertIn(COPY["rows"]["format_only"], body)
            self.assertIn(COPY["rows"]["content_missing"], body)
            fmt = read(sample_text("wire_report")).format
            self.assertEqual(page.nums[0], present.fake_real(fmt.fake))

    def test_one_state_per_page(self):
        for url in ["/", "/?sample=wire_report", "/?sample=headline_only", "/nope"]:
            with self.subTest(url=url):
                page = parse_page(self.client.get(url).get_data(as_text=True))
                self.assertEqual(len(page.states), 1)


class RenderEdgeTests(unittest.TestCase):
    def render(self, view) -> str:
        with app.test_request_context("/"):
            return render_template("index.html", view=view, text="", count="0 words", samples=[],
                                   active_sample=None, scope=present.SCOPE_LINE)

    def test_a_zero_probability_still_renders_its_row(self):
        fmt = ModelReading("format_model", 0.3, 0.7, "REAL", 10, 0.9, 30.0, 0.6)
        content = ModelReading("content_model", 0.0, 1.0, "REAL", 10, 0.9, 49.0, 0.4)
        reading = Reading("text", "text", fmt, content, "ok", 1)
        page = parse_page(self.render(present.scored(reading, get_models())))
        self.assertEqual(page.states, ["verdict"])
        self.assertEqual(page.nums[2], "<1\u00a0%\u00a0fake · >99\u00a0%\u00a0real")

    def test_a_missing_value_says_not_scored(self):
        row = present.Row("Content model", present.fake_real(None))
        view = present.View("verdict", "Leans real", rows=[present.Row("Blended 60/40", present.fake_real(0.3)), row])
        page = parse_page(self.render(view))
        self.assertEqual(page.nums, ["30\u00a0%\u00a0fake · 70\u00a0%\u00a0real", "not scored"])


class ScriptTests(unittest.TestCase):
    def test_the_script_never_disables_the_button(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        code = "\n".join(line for line in js.splitlines() if not line.strip().startswith("//"))
        self.assertNotRegex(code, r"\.disabled\b|setAttribute\(\s*['\"]disabled")
        self.assertIn("requestSubmit()", code)
        self.assertIn("pageshow", code)

    def test_the_script_formats_no_probability(self):
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertNotRegex(js, r"toFixed|Math\.round|%")


if __name__ == "__main__":
    unittest.main()
