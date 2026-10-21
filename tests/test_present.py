"""The one formatter: rounding, complements, clamps, bands, leans and agreement lines."""

import math
import re
import unittest

from tests._support import COPY
from utils import present
from utils.preprocess import ModelReading

PAIR = re.compile(r"^(<1|>99|\d{1,2})\u00a0%\u00a0fake · (<1|>99|\d{1,2})\u00a0%\u00a0real$")


def reading(key: str, fake: float, known: int = 5) -> ModelReading:
    return ModelReading(key=key, fake=fake, real=1 - fake, label="FAKE" if fake >= 0.5 else "REAL",
                        known_features=known, coverage=0.9, points_from_empty=10.0, weight=0.5)


class RoundingTests(unittest.TestCase):
    def test_every_value_shows_a_pair_that_adds_to_100(self):
        for i in range(100001):  # 0 to 100 percent in steps of 0.001
            p = i / 100000
            text = present.fake_real(p)
            match = PAIR.match(text)
            self.assertIsNotNone(match, f"{p}: {text!r}")
            fake, real = match.groups()
            if fake.isdigit() and real.isdigit():
                self.assertEqual(int(fake) + int(real), 100, f"{p}: {text!r}")
                self.assertTrue(1 <= int(fake) <= 99)
            else:
                self.assertIn((fake, real), {("<1", ">99"), (">99", "<1")}, f"{p}: {text!r}")

    def test_half_up_from_the_raw_value(self):
        self.assertEqual(present.shown(0.125), 13)  # round() would give 12, int() would give 12
        self.assertEqual(present.shown(0.375), 38)
        self.assertEqual(present.shown(0.12499), 12)
        self.assertEqual(present.shown(0.5249), 52)

    def test_clamps(self):
        self.assertEqual(present.fake_real(0.004), "<1\u00a0%\u00a0fake · >99\u00a0%\u00a0real")
        self.assertEqual(present.fake_real(0.0), "<1\u00a0%\u00a0fake · >99\u00a0%\u00a0real")
        self.assertEqual(present.fake_real(0.996), ">99\u00a0%\u00a0fake · <1\u00a0%\u00a0real")
        self.assertEqual(present.fake_real(1.0), ">99\u00a0%\u00a0fake · <1\u00a0%\u00a0real")
        self.assertEqual(present.fake_real(0.006), "1\u00a0%\u00a0fake · 99\u00a0%\u00a0real")

    def test_only_finite_numbers_are_shown(self):
        for value in (None, math.nan, math.inf, -math.inf, True, "0.5"):
            with self.subTest(value=value):
                self.assertEqual(present.fake_real(value), "not scored")
                self.assertIsNone(present.shown(value))

    def test_word_counts(self):
        self.assertEqual(present.words(0), "0 words")
        self.assertEqual(present.words(1), "1 word")
        self.assertEqual(present.words(1152), "1,152 words")


class BandTests(unittest.TestCase):
    def test_verdict_bands_read_the_shown_number(self):
        cases = {
            0.50: "Too close to call",
            0.54: "Too close to call",
            0.55: "Leans fake",
            0.64: "Leans fake",
            0.65: "Reads like fake news",
            0.79: "Reads like fake news",
            0.80: "Reads strongly like fake news",
            0.46: "Too close to call",
            0.45: "Leans real",
            0.36: "Leans real",
            0.35: "Reads like real reporting",
            0.21: "Reads like real reporting",
            0.20: "Reads strongly like real reporting",
            0.5449: "Too close to call",
            0.5451: "Leans fake",
        }
        for value, headline in cases.items():
            with self.subTest(blended_fake=value):
                self.assertEqual(present.verdict_headline(value), headline)
                self.assertIn(headline, COPY["headlines"]["verdict"])

    def test_lean_edges(self):
        self.assertEqual(present.lean(0.45), "real")
        self.assertIsNone(present.lean(0.46))
        self.assertIsNone(present.lean(0.54))
        self.assertEqual(present.lean(0.55), "fake")
        self.assertEqual(present.lean(0.4549), "real")
        self.assertIsNone(present.lean(0.4551))


class AgreementTests(unittest.TestCase):
    def test_all_agreement_lines(self):
        cases = [
            ((0.70, 0.80), "Both models lean fake."),
            ((0.20, 0.30), "Both models lean real."),
            ((0.50, 0.52), "Neither model leans either way."),
            ((0.70, 0.50), "The format model leans fake; the content model has no lean."),
            ((0.30, 0.50), "The format model leans real; the content model has no lean."),
            ((0.50, 0.70), "The content model leans fake; the format model has no lean."),
            ((0.50, 0.30), "The content model leans real; the format model has no lean."),
        ]
        seen = set()
        for (f, c), line in cases:
            with self.subTest(format=f, content=c):
                got = present._agreement(reading("format_model", f), reading("content_model", c))
                self.assertEqual(got, line)
                seen.add(got)
        self.assertEqual(seen, set(COPY["agreement"]))

    def test_opposite_leans_have_no_agreement_line(self):
        self.assertIsNone(present._agreement(reading("format_model", 0.3), reading("content_model", 0.7)))

    def test_a_model_that_recognises_nothing_does_not_lean(self):
        fmt = reading("format_model", 0.66, known=0)  # the format model's empty-text answer leans fake
        self.assertEqual(present._agreement(fmt, reading("content_model", 0.7)),
                         "The content model leans fake; the format model has no lean.")
        self.assertEqual(present._model_row(fmt, with_lean=True).numbers, COPY["rows"]["no_words"])

    def test_lean_labels_only_on_disagreement_rows(self):
        row = present._model_row(reading("format_model", 0.3), with_lean=True)
        self.assertEqual(row.lean_text, "leans real")
        self.assertIsNone(present._model_row(reading("format_model", 0.3), with_lean=False).lean_text)


if __name__ == "__main__":
    unittest.main()
