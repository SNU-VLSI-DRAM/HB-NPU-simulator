import copy
import contextlib
import gzip
import importlib
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).parent))
run_regression = importlib.import_module("run_regression")


def regression_result(cycle=4000):
    return {
        "terminated": True,
        "termination_cycle": cycle,
        "commands": {str(channel): [] for channel in range(8)},
        "stats": {"0": {"num_cycles": cycle}},
    }


@contextlib.contextmanager
def replace_runner_globals(**replacements):
    original = {name: getattr(run_regression, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(run_regression, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(run_regression, name, value)


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

    def test_run_case_collects_stats_and_all_eight_command_streams(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            trace = temp_dir / "input.trace"
            trace.write_text("input\n")
            binary = temp_dir / "fake_simulator.py"
            binary.write_text("""#!/usr/bin/env python3
import json
from pathlib import Path
import sys

output_dir = Path(sys.argv[7])
(output_dir / "dramsim3.json").write_text(json.dumps({"0": {"num_cycles": 4001}}))
for channel in range(8):
    (output_dir / ("dramsim3ch_{}cmd.trace".format(channel))).write_text(
        "7 act {} 0 0 0 0x1 0x2\\n".format(channel))
print("Turn off PIM")
""")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

            result = run_regression.run_case(binary, trace)

            self.assertEqual(result["termination_cycle"], 4001)
            self.assertTrue(result["terminated"])
            self.assertEqual(result["commands"]["0"], [{
                "cycle": 7, "kind": "act", "channel": 0, "rank": 0,
                "bankgroup": 0, "bank": 0, "row": 1, "column": 2,
            }])
            self.assertEqual(result["commands"]["7"][0]["channel"], 7)

    def test_run_case_rejects_failed_or_unterminated_simulation(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            trace = temp_dir / "input.trace"
            binary = temp_dir / "fake_simulator.py"
            binary.write_text("""#!/usr/bin/env python3
from pathlib import Path
import sys

mode = Path(sys.argv[5]).read_text().strip()
if mode == "failure":
    print("simulator failed")
    raise SystemExit(9)
print("simulation ended without marker")
""")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

            trace.write_text("failure\n")
            with self.assertRaisesRegex(AssertionError, "simulator failed"):
                run_regression.run_case(binary, trace)

            trace.write_text("unterminated\n")
            with self.assertRaisesRegex(AssertionError, "simulation did not terminate"):
                run_regression.run_case(binary, trace)

    def test_ordinary_quick_comparison_rejects_changes_without_writing_golden(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            trace = temp_dir / "case.trace"
            trace.write_text("case\n")
            golden_dir = temp_dir / "tests" / "regression" / "golden"
            golden_dir.mkdir(parents=True)
            golden = golden_dir / "case.json.gz"
            run_regression.write_golden(golden, regression_result())
            before = golden.read_bytes()

            with replace_runner_globals(
                    QUICK_CASES={"case": trace},
                    ROOT=temp_dir,
                    run_case=lambda binary, trace: regression_result(4001)):
                with self.assertRaisesRegex(AssertionError, "mismatch at stats/0/num_cycles"):
                    run_regression.run_quick(Path("unused"), False)

            self.assertEqual(golden.read_bytes(), before)

    def test_update_quick_rejects_unstable_baseline_without_creating_golden(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            trace = temp_dir / "case.trace"
            trace.write_text("case\n")
            results = iter([regression_result(), regression_result(4001), regression_result(4001)])

            with replace_runner_globals(
                    QUICK_CASES={"case": trace},
                    ROOT=temp_dir,
                    run_case=lambda binary, trace: next(results)):
                with self.assertRaisesRegex(AssertionError, "unstable baseline: case"):
                    run_regression.run_quick(Path("unused"), True)

            self.assertFalse(
                (temp_dir / "tests" / "regression" / "golden" / "case.json.gz").exists())

    def test_refresh_quick_rejects_cycle_at_trefi_boundary(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            trace = temp_dir / "refresh.trace"
            trace.write_text("refresh\n")

            with replace_runner_globals(
                    QUICK_CASES={"refresh_boundary": trace},
                    ROOT=temp_dir,
                    run_case=lambda binary, trace: regression_result(3900)):
                with self.assertRaisesRegex(AssertionError, "refresh fixture did not cross tREFI"):
                    run_regression.run_quick(Path("unused"), False)


if __name__ == "__main__":
    unittest.main()
