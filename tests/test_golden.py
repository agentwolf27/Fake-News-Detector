"""The app still answers exactly what 23207ce answered, on both predict routes.

tests/golden.json was captured from the archived code before this branch changed any (see
scripts/capture_golden.py). Its bytes are pinned here, so it cannot be regenerated quietly.
"""

import hashlib
import unittest

from tests._support import GOLDEN, ROOT, client, strict_json
from utils.preprocess import clean_text

GOLDEN_SHA256 = "b8635791ea466902c1c2476f74ed683077b50fba50d75459d14f699517b7c36e"
ALLOWED_NEW_KEYS = {"input_check"}


class GoldenFileTests(unittest.TestCase):
    def test_golden_file_is_the_pinned_capture(self):
        digest = hashlib.sha256((ROOT / "tests" / "golden.json").read_bytes()).hexdigest()
        self.assertEqual(digest, GOLDEN_SHA256)

    def test_golden_has_fourteen_inputs_from_23207ce(self):
        self.assertEqual(GOLDEN["source_commit"], "23207ce")
        self.assertEqual(len(GOLDEN["inputs"]), 14)
        self.assertEqual(len(GOLDEN["regex_probes"]), 12)


class GoldenRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = client()

    def _compare(self, route: str, key: str):
        checked = 0
        for name, record in GOLDEN["inputs"].items():
            with self.subTest(input=name, route=route):
                response = self.client.post(route, json={"text": record["text"]})
                expected = record[key]
                self.assertEqual(response.status_code, expected["status"])
                body = strict_json(response.data)
                self.assertEqual(set(body) - set(expected["body"]), ALLOWED_NEW_KEYS)
                self.assertEqual({k: v for k, v in body.items() if k not in ALLOWED_NEW_KEYS}, expected["body"])
                checked += 1
        self.assertEqual(checked, 14)

    def test_predict_bodies_are_unchanged(self):
        self._compare("/api/predict", "predict")

    def test_simple_bodies_are_unchanged(self):
        self._compare("/api/predict/simple", "simple")

    def test_html_stripping_matches_the_old_output(self):
        for probe, expected in GOLDEN["regex_probes"].items():
            with self.subTest(probe=probe):
                self.assertEqual(clean_text(probe), expected)


if __name__ == "__main__":
    unittest.main()
