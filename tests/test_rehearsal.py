"""What the recording showed is still what the code serves.

docs/rehearsal.json is written by `scripts/rehearse.py --record` right after a take. If any page,
asset or API response the take showed would now render differently, the take is stale.
"""

import hashlib
import json
import unittest

from tests._support import ROOT, client

RECORD = ROOT / "docs" / "rehearsal.json"


def render(c, shot):
    if shot["method"] == "POST":
        body = (ROOT / shot["body_file"]).read_bytes()
        return c.post(shot["path"], data=body, content_type="application/json")
    return c.get(shot["path"])


class RehearsalTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(RECORD.is_file(), "no rehearsal recorded yet: run scripts/rehearse.py --record after the take")
        self.record = json.loads(RECORD.read_text(encoding="utf-8"))
        self.client = client()

    def test_the_record_covers_the_shot_list(self):
        paths = [s["path"] for s in self.record["shots"]]
        for path in ["/", "/?sample=wire_report", "/?sample=sensational_post", "/?sample=sensational_dateline",
                     "/?sample=headline_only", "/static/style.css", "/static/app.js", "/api/models", "/api/predict"]:
            self.assertIn(path, paths)
        self.assertRegex(self.record["window"], r"^\d{3,4}x\d{3,4}$")

    def test_current_code_serves_the_same_bytes(self):
        for shot in self.record["shots"]:
            with self.subTest(path=shot["path"]):
                response = render(self.client, shot)
                self.assertEqual(response.status_code, shot["status"])
                self.assertEqual(hashlib.sha256(response.data).hexdigest(), shot["sha256"])

    def test_pages_render_the_same_twice(self):
        for shot in self.record["shots"]:
            with self.subTest(path=shot["path"]):
                self.assertEqual(render(self.client, shot).data, render(self.client, shot).data)

    def test_recorded_states_are_the_ones_the_shot_list_needs(self):
        states = {s["path"]: s.get("state") for s in self.record["shots"]}
        self.assertEqual(states["/"], "first_load")
        self.assertEqual(states["/?sample=wire_report"], "verdict")
        self.assertEqual(states["/?sample=sensational_post"], "verdict")
        self.assertEqual(states["/?sample=sensational_dateline"], "disagreement")
        self.assertEqual(states["/?sample=headline_only"], "no_signal")


if __name__ == "__main__":
    unittest.main()
