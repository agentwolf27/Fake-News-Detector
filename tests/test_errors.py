"""Bad requests get a 4xx with a named code; nothing leaks Python internals."""

import time
import unittest
from unittest import mock

from tests._support import client, parse_page, serving, strict_json
from utils.preprocess import MAX_TEXT_BYTES, clean_text

ROUTES = ("/api/predict", "/api/predict/simple")
SECRET = "boom-secret /Users/someone/app.py"
LEAKS = ("Traceback", "/Users/", "Werkzeug", "werkzeug", "object has no attribute", SECRET, "Exception")


class ApiErrorTests(unittest.TestCase):
    def setUp(self):
        self.client = client()

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status, response.data[:200])
        body = strict_json(response.data)
        self.assertEqual(body["code"], code)
        self.assertIsInstance(body["error"], str)
        text = response.get_data(as_text=True)
        for leak in LEAKS:
            self.assertNotIn(leak, text)
        return body

    def test_client_mistakes(self):
        cases = [
            ("malformed json", dict(data=b"{bad", content_type="application/json"), 400, "bad_json"),
            ("json array", dict(data=b'["text"]', content_type="application/json"), 400, "bad_json"),
            ("json string", dict(data=b'"text"', content_type="application/json"), 400, "bad_json"),
            ("deep nesting", dict(data=b"[" * 100000, content_type="application/json"), 400, "bad_json"),
            ("empty body", dict(data=b"", content_type="application/json"), 400, "bad_json"),
            ("empty object", dict(json={}), 400, "text_missing"),
            ("other field", dict(json={"body": "x"}), 400, "text_missing"),
            ("number", dict(json={"text": 5}), 400, "text_not_string"),
            ("null", dict(json={"text": None}), 400, "text_not_string"),
            ("list", dict(json={"text": ["a"]}), 400, "text_not_string"),
            ("whitespace", dict(json={"text": " \n\t "}), 400, "text_empty"),
            ("form body", dict(data={"text": "hello"}), 415, "json_required"),
            ("no content type", dict(data=b'{"text": "hello"}'), 415, "json_required"),
            ("digits only", dict(json={"text": "12345 67890"}), 422, "no_known_words"),
            ("non-latin", dict(json={"text": "東京の天気"}), 422, "no_known_words"),
            ("link only", dict(json={"text": "https://example.com/news/story"}), 422, "url_only"),
        ]
        for route in ROUTES:
            for name, kwargs, status, code in cases:
                with self.subTest(route=route, case=name):
                    body = self.assert_error(self.client.post(route, **kwargs), status, code)
                    if status == 422:
                        self.assertIn("input_check", body)
                        self.assertEqual(body["input_check"]["status"], code)

    def test_text_size_limit_is_inclusive(self):
        at_limit = "a" * MAX_TEXT_BYTES
        over = "a" * (MAX_TEXT_BYTES + 1)
        for route in ROUTES:
            with self.subTest(route=route):
                self.assertNotEqual(self.client.post(route, json={"text": at_limit}).status_code, 413)
                body = self.assert_error(self.client.post(route, json={"text": over}), 413, "too_large")
                self.assertEqual(body["text_bytes"], MAX_TEXT_BYTES + 1)
                self.assertEqual(body["limit_bytes"], MAX_TEXT_BYTES)

    def test_multibyte_text_is_measured_in_bytes(self):
        text = "é" * (MAX_TEXT_BYTES // 2 + 1)  # fewer characters than the limit, more bytes
        self.assert_error(self.client.post("/api/predict", json={"text": text}), 413, "too_large")

    def test_body_over_one_megabyte(self):
        big = b'{"text": "' + b"a" * (1024 * 1024 + 10) + b'"}'
        self.assert_error(self.client.post("/api/predict", data=big, content_type="application/json"), 413, "too_large")

    def test_models_missing(self):
        with serving():
            for route in ROUTES:
                with self.subTest(route=route):
                    body = self.assert_error(self.client.post(route, json={"text": "The council met."}), 503, "model_unavailable")
                    self.assertEqual(body["reason"], "fake_news_model.pkl not found")
            health = self.client.get("/health")
            self.assertEqual(health.status_code, 503)
            self.assertEqual(strict_json(health.data)["status"], "unavailable")

    def test_unexpected_failure(self):
        with mock.patch("app.app.ensemble_result", side_effect=RuntimeError(SECRET)):
            with self.assertLogs("fake_news_detector", level="ERROR"):
                body = self.assert_error(self.client.post("/api/predict", json={"text": "The council met on Monday."}),
                                         500, "internal_error")
        self.assertRegex(body["request_id"], r"^[0-9a-f]{32}$")

    def test_unknown_route_and_wrong_method(self):
        self.assert_error(self.client.get("/api/nope"), 404, "not_found")
        self.assert_error(self.client.get("/api/predict"), 405, "method_not_allowed")
        self.assert_error(self.client.delete("/health"), 405, "method_not_allowed")

    def test_debugger_console_is_not_served(self):
        self.assertEqual(self.client.get("/console").status_code, 404)

    def test_html_stripping_is_linear(self):
        started = time.perf_counter()
        clean_text("<" * 256000)
        clean_text("<a" * 128000)
        self.assertLess(time.perf_counter() - started, 1.0)


class PageErrorTests(unittest.TestCase):
    def setUp(self):
        self.client = client()

    def page(self, response, status, code):
        self.assertEqual(response.status_code, status)
        html = response.get_data(as_text=True)
        page = parse_page(html)
        self.assertEqual(page.states, ["error"])
        self.assertEqual(page.codes, [code])
        for leak in LEAKS:
            self.assertNotIn(leak, html)
        return page

    def test_empty_submit(self):
        self.page(self.client.post("/", data={"text": "   "}), 400, "text_empty")
        self.page(self.client.post("/", data={}), 400, "text_empty")

    def test_too_much_text(self):
        page = self.page(self.client.post("/", data={"text": "a" * (MAX_TEXT_BYTES + 1)}), 413, "too_large")
        self.assertIn(f"This text is {MAX_TEXT_BYTES + 1:,} bytes", " ".join(page.result_text))

    def test_models_missing(self):
        with serving():
            page = self.page(self.client.post("/", data={"text": "The council met."}), 503, "model_unavailable")
        self.assertIn("fake_news_model.pkl not found", " ".join(page.result_text))

    def test_unexpected_failure(self):
        with mock.patch("app.app.present.scored", side_effect=RuntimeError(SECRET)):
            with self.assertLogs("fake_news_detector", level="ERROR"):
                page = self.page(self.client.post("/", data={"text": "The council met on Monday."}), 500, "internal_error")
        self.assertRegex(" ".join(page.result_text), r"Logged as request [0-9a-f]{32}; no details are shown here\.")

    def test_not_found_page(self):
        response = self.client.get("/nope")
        self.assertEqual(response.status_code, 404)
        page = parse_page(response.get_data(as_text=True))
        self.assertEqual(page.states, ["not_found"])
        self.assertEqual(page.headlines, ["Nothing at /nope"])
        long = self.client.get("/" + "x" * 200)
        self.assertEqual(parse_page(long.get_data(as_text=True)).headlines, ["Nothing at /" + "x" * 79])

    def test_unknown_sample(self):
        response = self.client.get("/?sample=nope")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(parse_page(response.get_data(as_text=True)).states, ["not_found"])


if __name__ == "__main__":
    unittest.main()
