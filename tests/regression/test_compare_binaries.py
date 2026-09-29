import json
from pathlib import Path
import stat
import tempfile
import unittest

from compare_binaries import first_difference, compare_outputs, run_comparison


class BinaryComparisonTest(unittest.TestCase):
    def test_first_difference(self):
        self.assertIsNone(first_difference(["a"], ["a"]))
        self.assertEqual(first_difference(["a", "b"], ["a", "c"]), (1, "b", "c"))
        self.assertEqual(first_difference(["a"], ["a", "b"]), (1, None, "b"))
        self.assertEqual(first_difference(["a", "b"], ["a"]), (1, "b", None))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.left = self.root / "left"
        self.right = self.root / "right"
        for directory in (self.left, self.right):
            directory.mkdir()
            self.write_outputs(directory)

    def write_outputs(self, directory, cycles=100, energy=1.0):
        stats = {str(ch): {"num_cycles": cycles, "total_energy": energy} for ch in range(8)}
        (directory / "dramsim3.json").write_text(json.dumps(stats))
        for ch in range(8):
            (directory / f"dramsim3ch_{ch}cmd.trace").write_text(f"20 gh_read {ch} 0 0 1 0 0\n")
            (directory / f"dramsim3ch_{ch}pim.trace").write_text(f"20 {ch} GH_READ 0 0:1\n")
        (directory / "stdout.log").write_text("")

    def test_identical_bounded_runs_are_not_reported_as_complete(self):
        result = compare_outputs(self.left, self.right)
        self.assertFalse(result["completed"])
        self.assertEqual(result["cycles"], 100)
        self.assertEqual(result["physical_records"], 8)
        self.assertEqual(result["logical_records"], 8)

    def test_first_physical_cycle_difference_is_reported(self):
        (self.right / "dramsim3ch_3cmd.trace").write_text("21 gh_read 3 0 0 1 0 0\n")
        with self.assertRaisesRegex(AssertionError, "ch_3cmd.trace.*index 0.*20 gh_read.*21 gh_read"):
            compare_outputs(self.left, self.right)

    def test_logical_issue_length_is_exact(self):
        (self.right / "dramsim3ch_0pim.trace").write_text("")
        with self.assertRaisesRegex(AssertionError, "ch_0pim.trace.*index 0"):
            compare_outputs(self.left, self.right)

    def test_missing_channel_output_is_failure(self):
        (self.right / "dramsim3ch_7pim.trace").unlink()
        with self.assertRaisesRegex(AssertionError, "missing.*ch_7pim.trace"):
            compare_outputs(self.left, self.right)

    def test_energy_tolerance_does_not_apply_to_cycles(self):
        self.write_outputs(self.right, energy=1.0000005)
        compare_outputs(self.left, self.right)
        self.write_outputs(self.right, cycles=101)
        with self.assertRaisesRegex(AssertionError, "num_cycles"):
            compare_outputs(self.left, self.right)

    def test_one_sided_completion_is_failure(self):
        (self.right / "stdout.log").write_text("100 Output Exhausted: Array0. Turn off PIM mode.\n")
        with self.assertRaisesRegex(AssertionError, "completion"):
            compare_outputs(self.left, self.right)

    def test_complete_pair_is_identified(self):
        for d in (self.left, self.right):
            (d / "stdout.log").write_text("100 Output Exhausted: Array0. Turn off PIM mode.\n")
        self.assertTrue(compare_outputs(self.left, self.right)["completed"])

    def make_binary(self, extra=""):
        binary = self.root / "simulator.py"
        binary.write_text("""#!/usr/bin/env python3
import json, pathlib, sys, time
output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
cycles = int(sys.argv[sys.argv.index('-c') + 1])
""" + extra + """
(output / 'dramsim3.json').write_text(json.dumps({str(c): {'num_cycles': cycles} for c in range(8)}))
for c in range(8):
    for suffix in ('cmd', 'pim'):
        (output / f'dramsim3ch_{c}{suffix}.trace').write_text('')
""")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def run_pair(self, extra="", timeout=2):
        binary = self.make_binary(extra)
        trace, config = self.root / "input.trace", self.root / "config.ini"
        trace.touch(); config.touch()
        return run_comparison(binary, binary, trace, config, self.root / "results", 123, timeout)

    def test_runner_passes_cycle_limit_and_retains_result(self):
        result = self.run_pair()
        self.assertEqual(result["cycles"], 123)
        self.assertTrue(result["passed"])
        self.assertTrue((self.root / "results" / "result.json").is_file())

    def test_nonzero_process_is_not_a_prefix_success(self):
        with self.assertRaisesRegex(AssertionError, "exit code 2"):
            self.run_pair("sys.exit(2)\n")
        result = json.loads((self.root / "results" / "result.json").read_text())
        self.assertFalse(result["passed"])

    def test_timeout_is_not_a_prefix_success(self):
        with self.assertRaisesRegex(AssertionError, "timeout"):
            self.run_pair("time.sleep(2)\n", timeout=0.05)

    def test_existing_artifacts_are_not_overwritten(self):
        target = self.root / "results"
        target.mkdir()
        (target / "keep").write_text("original")
        with self.assertRaises(FileExistsError):
            self.run_pair()
        self.assertEqual((target / "keep").read_text(), "original")


if __name__ == "__main__":
    unittest.main()
