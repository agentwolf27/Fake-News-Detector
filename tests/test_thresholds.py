"""The "too little to go on" and "too few known words" rules, pinned either side of each edge.

The probe tables are DECISIONS.md D2 and D3, measured before any sample existed. Moving
NO_SIGNAL_POINTS to 2.0 flips "Stock markets…"; moving it to 4.0 flips "Bank…", "Workers…" and
"Retail sales…"; moving COVERAGE_FLOOR to 0.3 admits the French sentence and the frog headline.
"""

import json
import unittest

from tests._support import COPY, CONTENT_FILES, FORMAT_FILES, client, parse_page, serving, squash, strict_json
from utils import preprocess
from utils.models import get_models
from utils.preprocess import COVERAGE_FLOOR, NO_SIGNAL_POINTS, ModelReading, read

NO_SIGNAL_IN = {
    "Bridge closes for inspection after cracks found": 2.92,
    "Airline cancels hundreds of flights as storm approaches": 2.79,
    "Stock markets fall as investors weigh inflation data": 2.45,
    "City council votes to expand bike lanes downtown": 1.70,
    "Scientists at MIT develop new water purification technology using solar energy": 1.62,
}
NO_SIGNAL_OUT = {
    "Bank raises savings rates for customers": 3.08,
    "Workers finish repairs on main water line": 3.83,
    "Retail sales rise for third straight month": 3.87,
    "Senate passes bill to fund highway repairs": 4.15,
    "Government plans to replace all birds with surveillance drones": 9.53,
    "Apple unveils new iPhone with faster chip and longer battery life": 10.15,
}
REFUSED_AS_UNKNOWN = [
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore "
    "magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo "
    "consequat.",
    "asdf jkl qwer uiop zxcv bnm hjkl yuio tyui ghjk fdsa rewq vcxz mnbv lkjh poiu",
    "Le conseil municipal a voté mardi pour prolonger le service de bus de nuit sur quatre lignes à partir du mois "
    "prochain, après un essai de six mois.",
    "Researchers find new species of frog in rainforest",
]
ADMITTED = [
    "Museum reopens after two-year renovation",
    "Scientists at MIT develop new water purification technology using solar energy",
]


def stub(points: float, coverage: float | None = 1.0, key: str = "format_model") -> ModelReading:
    return ModelReading(key=key, fake=0.5, real=0.5, label="FAKE", known_features=3, coverage=coverage,
                        points_from_empty=points, weight=0.6)


class NoSignalTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(NO_SIGNAL_POINTS, 3.0)

    def test_empty_text_scores_are_the_recorded_ones(self):
        models = get_models()
        self.assertEqual(repr(models.format.empty_fake), "0.6551573256765925")
        self.assertEqual(repr(models.content.empty_fake), "0.49755588867465184")

    def test_probes_inside_the_edge_are_refused(self):
        for text, distance in NO_SIGNAL_IN.items():
            with self.subTest(text=text):
                reading = read(text)
                self.assertAlmostEqual(max(r.points_from_empty for r in reading.models), distance, delta=0.006)
                self.assertEqual(reading.status, "no_signal")

    def test_probes_outside_the_edge_are_scored(self):
        for text, distance in NO_SIGNAL_OUT.items():
            with self.subTest(text=text):
                reading = read(text)
                self.assertAlmostEqual(max(r.points_from_empty for r in reading.models), distance, delta=0.006)
                self.assertEqual(reading.status, "ok")

    def test_the_edge_itself_is_inside(self):
        exactly = abs(0.03 - 0.0) * 100  # a base of 0.0, so this is exactly 3.0
        self.assertEqual(exactly, 3.0)
        self.assertEqual(preprocess._status("words here", [stub(exactly)]), "no_signal")
        self.assertEqual(preprocess._status("words here", [stub(3.0000001)]), "ok")
        self.assertEqual(preprocess._status("words here", [stub(1.0), stub(3.5, key="content_model")]), "ok")

    def test_page_copy_quotes_the_stored_empty_scores(self):
        c = client()
        page = parse_page(c.post("/", data={"text": "Bridge closes for inspection after cracks found"}).get_data(as_text=True))
        self.assertEqual(page.states, ["no_signal"])
        self.assertIn(COPY["bodies"]["no_signal_both"], squash(" ".join(page.result_text)))
        with serving(*FORMAT_FILES):
            page = parse_page(c.post("/", data={"text": "Tech company recalls faulty chargers"}).get_data(as_text=True))
            self.assertEqual(page.states, ["no_signal"])
            self.assertIn(COPY["bodies"]["no_signal_format_only"], squash(" ".join(page.result_text)))

    def test_api_reports_the_rule_and_the_distance(self):
        body = strict_json(client().post("/api/predict", json={"text": "Stock markets fall as investors weigh inflation data"}).data)
        self.assertEqual(body["input_check"]["status"], "no_signal")
        self.assertEqual(body["input_check"]["points_from_empty"]["content_model"], 2.45)


class CoverageTests(unittest.TestCase):
    def test_constant_keeps_the_copy_true(self):
        # "Most of this text isn't in the vocabulary" is only true while the floor is at most a half.
        self.assertEqual(COVERAGE_FLOOR, 0.5)
        self.assertLessEqual(COVERAGE_FLOOR, 0.5)

    def test_named_nonsense_and_the_frog_headline_are_refused(self):
        for text in REFUSED_AS_UNKNOWN:
            with self.subTest(text=text[:40]):
                reading = read(text)
                self.assertEqual(reading.status, "no_known_words")
                self.assertTrue(all(r.coverage is None or r.coverage < COVERAGE_FLOOR for r in reading.models))

    def test_named_news_is_admitted(self):
        museum = read(ADMITTED[0])
        self.assertEqual(max(r.coverage for r in museum.models), 0.5)  # exactly at the floor
        for text in ADMITTED:
            with self.subTest(text=text):
                self.assertNotEqual(read(text).status, "no_known_words")

    def test_equal_to_the_floor_is_admitted(self):
        self.assertEqual(preprocess._status("words", [stub(10, coverage=0.5)]), "ok")
        self.assertEqual(preprocess._status("words", [stub(10, coverage=0.4999)]), "no_known_words")
        self.assertEqual(preprocess._status("words", [stub(10, coverage=None)]), "no_known_words")
        self.assertEqual(preprocess._status("words", [stub(10, coverage=None), stub(10, coverage=0.6, key="content_model")]), "ok")

    def test_links_come_first(self):
        for text in ["https://example.com/a", "www.example.com", "bbc.co.uk/news/world-europe-12345678",
                     "https://a.com/x http://b.org/y"]:
            with self.subTest(text=text):
                self.assertEqual(read(text).status, "url_only")
        self.assertNotEqual(read("Read more at https://example.com/a about the council vote").status, "url_only")

    def test_unrecognised_page_copy(self):
        page = parse_page(client().post("/", data={"text": REFUSED_AS_UNKNOWN[0]}).get_data(as_text=True))
        self.assertEqual(page.states, ["unrecognised"])
        self.assertIn(COPY["bodies"]["unrecognised"], squash(" ".join(page.result_text)))

    def test_content_only_directory_is_not_served(self):
        with serving(*CONTENT_FILES):
            response = client().post("/api/predict", json={"text": "The council met on Monday."})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(json.loads(response.data)["code"], "model_unavailable")


if __name__ == "__main__":
    unittest.main()
