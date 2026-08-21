import copy
import gzip
import importlib
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).parent))
run_regression = importlib.import_module("run_regression")


class RegressionRunnerTest(unittest.TestCase):
    def test_parses_ordered_command_records(self):
        with tempfile.TemporaryDirectory() as temp_name:
            trace = Path(temp_name) / "commands.trace"
            trace.write_text("120 gh_read 3 1 2 4 1a 2b\n")

            self.assertEqual(run_regression.parse_command_trace(trace), [{
                "cycle": 120,
                "kind": "gh_read",
                "channel": 3,
                "rank": 1,
                "bankgroup": 2,
                "bank": 4,
                "row": 26,
                "column": 43,
            }])

    def test_command_comparison_reports_first_different_cycle(self):
        expected = {"3": [{"cycle": 120, "kind": "gh_read"}]}
        actual = copy.deepcopy(expected)
        actual["3"][0]["cycle"] = 121

        with self.assertRaisesRegex(
                AssertionError,
                "channel 3.*index 0.*expected cycle 120.*actual cycle 121"):
            run_regression.compare_commands(actual, expected)

    def test_energy_values_allow_only_approved_tolerance(self):
        run_regression.compare_value(
            {"total_energy": 1.0}, {"total_energy": 1.0000005}, [])
        with self.assertRaisesRegex(AssertionError, "energy mismatch"):
            run_regression.compare_value(
                {"total_energy": 1.0}, {"total_energy": 1.00001}, [])

    def test_golden_output_is_byte_stable_and_readable(self):
        value = {"b": 2, "a": {"cycle": 4}}
        with tempfile.TemporaryDirectory() as temp_name:
            first = Path(temp_name) / "first.json.gz"
            second = Path(temp_name) / "second.json.gz"
            run_regression.write_golden(first, value)
            run_regression.write_golden(second, value)

            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(run_regression.read_golden(first), value)
            self.assertEqual(gzip.decompress(first.read_bytes()).decode("utf-8"), (
                '{\n  "a": {\n    "cycle": 4\n  },\n  "b": 2\n}\n'))


if __name__ == "__main__":
    unittest.main()
