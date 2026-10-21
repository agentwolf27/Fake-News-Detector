"""Each committed demo text produces the state its label claims, with room to spare.

The recording relies on these. A sample that sits within a point of an edge could change state
on a different machine or a later library, so the margins are checked on the raw values.
"""

import re
import unittest

from tests._support import GOLDEN, PREFIX, ROOT, client, parse_page, sample_index, sample_text
from utils.models import get_models
from utils.preprocess import COVERAGE_FLOOR, NO_SIGNAL_POINTS, blend, read

LEAN_REAL_BELOW = 45.5   # shown fake <= 45
LEAN_FAKE_FROM = 54.5    # shown fake >= 55
BAND_EDGES = (54.5, 64.5, 79.5)
DATELINE_PAIRS = [("central_bank", "central_bank_dateline"), ("wire_article", "wire_article_dateline"),
                  ("sensational_article", "sensational_article_dateline")]


class SampleTests(unittest.TestCase):
    def setUp(self):
        models = get_models()
        loaded = [rec.basename for pair in models.pairs.values() if pair.loaded for rec in pair.files]
        self.assertEqual(sorted(loaded), sorted(["fake_news_model.pkl", "vectorizer.pkl", "content_model.pkl",
                                                 "content_vectorizer.pkl"]))

    def test_index_lists_the_four_samples(self):
        index = sample_index()
        self.assertEqual([s["id"] for s in index],
                         ["wire_report", "sensational_post", "sensational_dateline", "headline_only"])
        self.assertEqual([s["expect"] for s in index], ["verdict", "verdict", "disagreement", "no_signal"])
        for s in index:
            self.assertEqual(s["file"], f"{s['id']}.json")
            self.assertTrue((ROOT / "samples" / s["file"]).is_file())

    def test_each_sample_shows_its_state(self):
        c = client()
        for s in sample_index():
            with self.subTest(sample=s["id"]):
                page = parse_page(c.get(f"/?sample={s['id']}").get_data(as_text=True))
                self.assertEqual(page.states, [s["expect"]])

    def test_the_dateline_sample_is_the_post_plus_the_prefix(self):
        self.assertEqual(sample_text("sensational_dateline"), PREFIX + sample_text("sensational_post"))

    def test_margins(self):
        for s in sample_index():
            r = read(sample_text(s["id"]))
            with self.subTest(sample=s["id"]):
                for m in r.models:
                    self.assertGreaterEqual(m.coverage, COVERAGE_FLOOR + 0.1)
                farthest = max(m.points_from_empty for m in r.models)
                if s["expect"] == "no_signal":
                    self.assertLessEqual(farthest, NO_SIGNAL_POINTS - 1)
                    continue
                self.assertGreaterEqual(farthest, NO_SIGNAL_POINTS + 1)
                for m in r.models:
                    fake = m.fake * 100
                    self.assertGreaterEqual(min(abs(fake - LEAN_REAL_BELOW), abs(fake - LEAN_FAKE_FROM)), 1, m.key)
                top = max(blend(r)) * 100
                self.assertGreaterEqual(min(abs(top - e) for e in BAND_EDGES), 1)

    def test_the_verdict_samples_lean_the_way_their_labels_say(self):
        wire, post = read(sample_text("wire_report")), read(sample_text("sensational_post"))
        self.assertTrue(all(m.fake * 100 < LEAN_REAL_BELOW for m in wire.models))
        self.assertTrue(all(m.fake * 100 >= LEAN_FAKE_FROM for m in post.models))

    def test_the_held_sample_splits_with_room(self):
        r = read(sample_text("sensational_dateline"))
        self.assertLessEqual(r.format.fake * 100, LEAN_REAL_BELOW - 2)
        self.assertGreaterEqual(r.content.fake * 100, LEAN_FAKE_FROM + 2)

    def test_the_split_survives_dropping_any_sentence(self):
        sentences = re.split(r"(?<=[.!?])\s+", sample_text("sensational_post").strip())
        self.assertGreaterEqual(len(sentences), 5)
        for i in range(len(sentences)):
            variant = PREFIX + " ".join(sentences[:i] + sentences[i + 1:])
            with self.subTest(dropped=i):
                r = read(variant)
                self.assertLess(r.format.fake * 100, LEAN_REAL_BELOW)
                self.assertGreaterEqual(r.content.fake * 100, LEAN_FAKE_FROM)

    def test_a_dateline_moves_only_the_format_model(self):
        pairs = [(sample_text("sensational_post"), sample_text("sensational_dateline"))]
        pairs += [(GOLDEN["inputs"][a]["text"], GOLDEN["inputs"][b]["text"]) for a, b in DATELINE_PAIRS]
        for plain, dated in pairs:
            with self.subTest(text=plain[:40]):
                a, b = read(plain), read(dated)
                self.assertEqual(a.content.fake, b.content.fake)
                self.assertGreaterEqual((a.format.fake - b.format.fake) * 100, 10)


if __name__ == "__main__":
    unittest.main()
