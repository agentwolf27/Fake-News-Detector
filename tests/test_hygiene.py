"""No test is skipped, and every module PLAN.md names actually contains tests."""

import re
import unittest

from tests._support import ROOT

# Built from pieces so this file does not match itself: calls to the skip decorators and
# methods, the skip exception, and the expected-failure decorator.
SKIP_USES = ["sk" + r"ip(?:If|Unless|Test)?\s*\(", "Sk" + "ipTest", "expected" + "Failure"]


class HygieneTests(unittest.TestCase):
    def test_no_test_is_skipped(self):
        pattern = re.compile(r"\b(?:" + "|".join(SKIP_USES) + r")")
        self.assertIsNotNone(pattern.search("@unittest." + "sk" + "ip('x')"))
        self.assertIsNotNone(pattern.search("self." + "sk" + "ipTest('x')"))
        for path in sorted((ROOT / "tests").glob("*.py")):
            with self.subTest(file=path.name):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    def test_every_module_named_in_the_plan_has_tests(self):
        plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        done = plan.split("## Done looks like", 1)[1].split("\n## ", 1)[0]
        modules = sorted(set(re.findall(r"\btests\.(test_\w+)", done)))
        self.assertGreaterEqual(len(modules), 9)
        loader = unittest.TestLoader()
        for name in modules:
            with self.subTest(module=name):
                self.assertTrue((ROOT / "tests" / f"{name}.py").is_file(), name)
                self.assertGreater(loader.loadTestsFromName(f"tests.{name}").countTestCases(), 0)


if __name__ == "__main__":
    unittest.main()
