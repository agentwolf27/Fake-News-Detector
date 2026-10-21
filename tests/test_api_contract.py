"""Every route keeps its fields and types; the new fields have the agreed shape; models load once."""

import unittest
import warnings
from unittest import mock

import joblib

from tests._support import GOLDEN, client, sample_text, strict_json
from utils import models as models_module

STATUSES = {"ok", "url_only", "no_known_words", "no_signal"}
LEVELS = {"HIGH", "MEDIUM", "LOW", "VERY_LOW"}


class InputCheckShape:
    def assert_input_check(self, check):
        self.assertEqual(set(check), {"status", "word_count", "known_features", "coverage", "points_from_empty"})
        self.assertIn(check["status"], STATUSES)
        self.assertIsInstance(check["word_count"], int)
        for field in ("known_features", "coverage", "points_from_empty"):
            self.assertEqual(set(check[field]), {"format_model", "content_model"})
        for value in check["known_features"].values():
            self.assertTrue(value is None or (isinstance(value, int) and value >= 0))
        for value in check["coverage"].values():
            self.assertTrue(value is None or (isinstance(value, float | int) and 0 <= value <= 1))
        for value in check["points_from_empty"].values():
            self.assertTrue(value is None or (isinstance(value, float | int) and value >= 0))


class PredictContractTests(unittest.TestCase, InputCheckShape):
    def setUp(self):
        self.client = client()

    def test_predict_fields_and_types(self):
        response = self.client.post("/api/predict", json={"text": sample_text("wire_report")})
        self.assertEqual(response.status_code, 200)
        body = strict_json(response.data)
        self.assertEqual(
            set(body),
            {"prediction", "confidence", "confidence_level", "is_uncertain", "models_agree", "text",
             "ensemble_probabilities", "model_breakdown", "input_check"},
        )
        self.assertIn(body["prediction"], {"FAKE", "REAL"})
        self.assertIsInstance(body["confidence"], float)
        self.assertIn(body["confidence_level"], LEVELS)
        self.assertIsInstance(body["is_uncertain"], bool)
        self.assertIsInstance(body["models_agree"], bool)
        self.assertIsInstance(body["text"], str)
        self.assertEqual(set(body["ensemble_probabilities"]), {"fake", "real"})
        self.assertEqual(set(body["model_breakdown"]), {"format_model", "content_model"})
        for part in body["model_breakdown"].values():
            self.assertEqual(set(part), {"prediction", "fake_prob", "real_prob"})
            self.assertIn(part["prediction"], {"FAKE", "REAL"})
        self.assert_input_check(body["input_check"])
        self.assertEqual(body["input_check"]["status"], "ok")
        self.assertEqual(body["input_check"]["word_count"], len(sample_text("wire_report").split()))

    def test_simple_fields_and_types(self):
        response = self.client.post("/api/predict/simple", json={"text": sample_text("wire_report")})
        body = strict_json(response.data)
        self.assertEqual(set(body), {"text", "prediction", "confidence", "confidence_score", "input_check"})
        self.assertAlmostEqual(body["confidence_score"], body["confidence"] / 100)
        self.assert_input_check(body["input_check"])

    def test_every_golden_response_is_strict_json_with_a_valid_input_check(self):
        for name, record in GOLDEN["inputs"].items():
            for route in ("/api/predict", "/api/predict/simple"):
                with self.subTest(input=name, route=route):
                    body = strict_json(self.client.post(route, json={"text": record["text"]}).data)
                    self.assert_input_check(body["input_check"])

    def test_refused_texts_still_score_with_a_status(self):
        body = strict_json(self.client.post("/api/predict", json={"text": sample_text("headline_only")}).data)
        self.assertEqual(body["input_check"]["status"], "no_signal")
        self.assertIn(body["prediction"], {"FAKE", "REAL"})


class HealthAndModelsTests(unittest.TestCase):
    def setUp(self):
        self.client = client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = strict_json(response.data)
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["service"], "fake-news-detector")
        self.assertEqual(body["models"], {"format_model": "available", "content_model": "available"})
        self.assertEqual(body["sklearn_version"], "1.7.0")

    def test_models(self):
        body = strict_json(self.client.get("/api/models").data)
        expected = {
            "format_model": ("RandomForestClassifier", 15000, [1, 3], 0.6, 65.52),
            "content_model": ("LogisticRegression", 12000, [1, 2], 0.4, 49.76),
        }
        for key, (kind, n, grams, weight, empty) in expected.items():
            with self.subTest(model=key):
                part = body[key]
                self.assertIsInstance(part["description"], str)
                self.assertIs(part["available"], True)
                self.assertIs(part["loaded"], True)
                self.assertEqual(part["type"], kind)
                self.assertEqual(part["n_features"], n)
                self.assertEqual(part["ngram_range"], grams)
                self.assertEqual(part["weight"], weight)
                self.assertEqual(part["empty_text_fake_prob"], empty)
        self.assertIs(body["ensemble"]["available"], True)
        self.assertIsInstance(body["ensemble"]["description"], str)
        self.assertEqual(body["ensemble"]["weights"], {"format_model": 0.6, "content_model": 0.4})
        self.assertEqual(body["no_signal_points"], 3.0)
        self.assertEqual(body["coverage_floor"], 0.5)


class LoadOnceTests(unittest.TestCase):
    def test_four_files_are_read_once_however_many_predictions_run(self):
        with mock.patch("joblib.load", wraps=joblib.load) as counter:
            loaded = models_module.load_models()
            self.assertEqual(counter.call_count, 4)
            with models_module.use_models(loaded):
                c = client()
                for i in range(10):
                    self.assertEqual(c.post("/api/predict", json={"text": f"The council met on day {i}."}).status_code, 200)
            self.assertEqual(counter.call_count, 4)

    def test_loading_raises_no_version_warning(self):
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = models_module.load_models()
        self.assertTrue(loaded.ready)
        self.assertFalse([w for w in caught if issubclass(w.category, InconsistentVersionWarning)])


if __name__ == "__main__":
    unittest.main()
