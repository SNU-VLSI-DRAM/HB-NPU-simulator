import copy
import contextlib
import gzip
import importlib
import json
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

    def test_rank_indexed_energy_values_allow_approved_tolerance(self):
        for field in ("act_stb_energy", "pre_stb_energy", "sref_energy"):
            with self.subTest(field=field):
                run_regression.compare_value(
                    {field: {"0": 195492.01}},
                    {field: {"0": 195492.0}},
                    [],
                )

    def test_rank_indexed_energy_values_reject_outside_tolerance(self):
        for field in ("act_stb_energy", "pre_stb_energy", "sref_energy"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(
                        AssertionError, "energy mismatch at {}/0".format(field)):
                    run_regression.compare_value(
                        {field: {"0": 195493.0}},
                        {field: {"0": 195492.0}},
                        [],
                    )

    def test_nested_non_energy_numbers_remain_exact(self):
        with self.assertRaisesRegex(
                AssertionError, "mismatch at stats/0/num_cycles"):
            run_regression.compare_value(
                {"stats": {"0": {"num_cycles": 4000.0000005}}},
                {"stats": {"0": {"num_cycles": 4000.0}}},
                [],
            )

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

    def test_opt27b_manifest_selects_exact_ordered_46_kernel_window(self):
        expected = [
            ("prompt/createQKV", "traces/OPT-2.7B_128_1024_32/prompt/createQKV"),
            ("prompt/QK", "traces/OPT-2.7B_128_1024_32/prompt/QK"),
            ("prompt/SV", "traces/OPT-2.7B_128_1024_32/prompt/SV"),
            ("prompt/Wo", "traces/OPT-2.7B_128_1024_32/prompt/Wo"),
            ("prompt/L1", "traces/OPT-2.7B_128_1024_32/prompt/L1"),
            ("prompt/L2", "traces/OPT-2.7B_128_1024_32/prompt/L2"),
            ("decode/createQKV", "traces/OPT-2.7B_128_1024_32/decode/createQKV"),
            ("decode/WS/createQKV", "traces/OPT-2.7B_128_1024_32/decode/WS/createQKV"),
            ("decode/Wo", "traces/OPT-2.7B_128_1024_32/decode/Wo"),
            ("decode/WS/Wo", "traces/OPT-2.7B_128_1024_32/decode/WS/Wo"),
            ("decode/L1", "traces/OPT-2.7B_128_1024_32/decode/L1"),
            ("decode/WS/L1", "traces/OPT-2.7B_128_1024_32/decode/WS/L1"),
            ("decode/L2", "traces/OPT-2.7B_128_1024_32/decode/L2"),
            ("decode/WS/L2", "traces/OPT-2.7B_128_1024_32/decode/WS/L2"),
            ("decode/QKV/QK_0128", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0128"),
            ("decode/QKV/SV_0128", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0128"),
            ("decode/QKV/QK_0129", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0129"),
            ("decode/QKV/SV_0129", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0129"),
            ("decode/QKV/QK_0130", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0130"),
            ("decode/QKV/SV_0130", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0130"),
            ("decode/QKV/QK_0131", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0131"),
            ("decode/QKV/SV_0131", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0131"),
            ("decode/QKV/QK_0132", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0132"),
            ("decode/QKV/SV_0132", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0132"),
            ("decode/QKV/QK_0133", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0133"),
            ("decode/QKV/SV_0133", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0133"),
            ("decode/QKV/QK_0134", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0134"),
            ("decode/QKV/SV_0134", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0134"),
            ("decode/QKV/QK_0135", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0135"),
            ("decode/QKV/SV_0135", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0135"),
            ("decode/QKV/QK_0136", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0136"),
            ("decode/QKV/SV_0136", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0136"),
            ("decode/QKV/QK_0137", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0137"),
            ("decode/QKV/SV_0137", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0137"),
            ("decode/QKV/QK_0138", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0138"),
            ("decode/QKV/SV_0138", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0138"),
            ("decode/QKV/QK_0139", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0139"),
            ("decode/QKV/SV_0139", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0139"),
            ("decode/QKV/QK_0140", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0140"),
            ("decode/QKV/SV_0140", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0140"),
            ("decode/QKV/QK_0141", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0141"),
            ("decode/QKV/SV_0141", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0141"),
            ("decode/QKV/QK_0142", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0142"),
            ("decode/QKV/SV_0142", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0142"),
            ("decode/QKV/QK_0143", "traces/OPT-2.7B_128_1024_32/decode/QKV/QK_0143"),
            ("decode/QKV/SV_0143", "traces/OPT-2.7B_128_1024_32/decode/QKV/SV_0143"),
        ]

        actual = [
            (name, trace.relative_to(run_regression.ROOT).as_posix())
            for name, trace in run_regression.opt27b_cases()
        ]
        self.assertEqual(actual, expected)

    def test_long_summary_hashes_canonical_ordered_commands_and_discards_them(self):
        result = regression_result(42)
        result["commands"] = {
            "0": [
                {
                    "cycle": 3, "kind": "pre", "channel": 0, "rank": 0,
                    "bankgroup": 1, "bank": 2, "row": 16, "column": 4,
                },
                {
                    "cycle": 7, "kind": "act", "channel": 0, "rank": 1,
                    "bankgroup": 3, "bank": 0, "row": 32, "column": 8,
                },
            ],
        }
        result["stats"] = {"0": {"total_energy": 1.0}}

        summary = run_regression.summarize_long_case(result)
        self.assertEqual(summary, {
            "terminated": True,
            "termination_cycle": 42,
            "command_sha256": {
                "0": "a6466bb992be4ea1a0957ca3a06f5cb74f205c5922541489d65bfccdde6d1536",
            },
            "stats": {"0": {"total_energy": 1.0}},
        })

        reversed_result = copy.deepcopy(result)
        reversed_result["commands"]["0"].reverse()
        self.assertEqual(
            run_regression.summarize_long_case(reversed_result)["command_sha256"]["0"],
            "d156b547b2e25613ff3b84488b228be9510e509dad287a03905185ac44f6daf5",
        )
        self.assertNotEqual(
            summary["command_sha256"]["0"],
            run_regression.summarize_long_case(reversed_result)["command_sha256"]["0"],
        )

        reordered_keys = copy.deepcopy(result)
        reordered_keys["commands"]["0"] = [
            dict(reversed(list(command.items())))
            for command in reordered_keys["commands"]["0"]
        ]
        self.assertEqual(
            run_regression.summarize_long_case(reordered_keys)["command_sha256"]["0"],
            summary["command_sha256"]["0"],
        )

    def test_ordinary_opt27b_comparison_rejects_changes_without_writing_golden(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            golden_path = temp_dir / "tests/regression/golden/opt27b_128_16_32.json"
            golden_path.parent.mkdir(parents=True)
            expected = {
                "model": "OPT-2.7B", "input_tokens": 128,
                "output_tokens": 16, "batch_size": 32,
                "kernels": {"prompt/QK": {"terminated": True,
                            "termination_cycle": 42,
                            "command_sha256": {"0": "expected"},
                            "stats": {"0": {"total_energy": 1.0}}}},
            }
            golden_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
            before = golden_path.read_bytes()
            actual = copy.deepcopy(expected)
            actual["kernels"]["prompt/QK"]["command_sha256"]["0"] = "actual"

            with replace_runner_globals(ROOT=temp_dir, run_opt27b_once=lambda binary: actual):
                with self.assertRaisesRegex(
                        AssertionError,
                        "kernels/prompt/QK/command_sha256/0.*expected.*actual"):
                    run_regression.run_opt27b(Path("unused"), False)

            self.assertEqual(golden_path.read_bytes(), before)

    def test_update_opt27b_rejects_unstable_baseline_without_creating_golden(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp_dir = Path(temp_name)
            stable = {
                "model": "OPT-2.7B", "input_tokens": 128,
                "output_tokens": 16, "batch_size": 32, "kernels": {},
            }
            unstable = copy.deepcopy(stable)
            unstable["output_tokens"] = 17
            results = iter([stable, unstable, unstable])

            with replace_runner_globals(
                    ROOT=temp_dir, run_opt27b_once=lambda binary: next(results)):
                with self.assertRaisesRegex(AssertionError, "unstable OPT-2.7B baseline"):
                    run_regression.run_opt27b(Path("unused"), True)

            self.assertFalse(
                (temp_dir / "tests/regression/golden/opt27b_128_16_32.json").exists())


if __name__ == "__main__":
    unittest.main()
