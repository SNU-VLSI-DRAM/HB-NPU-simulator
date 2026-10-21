"""Run the actual OPT entry point without touching repository traces/results."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import compile_kernel, load_workload, write_trace
from run_demo import simulate_kernel


class OptEvaluationTest(unittest.TestCase):
    binary = None

    def test_run_demo_generates_and_evaluates_prefill_decode_and_attention(self):
        with tempfile.TemporaryDirectory(prefix="hbnpu-opt-cli-") as tmp:
            tmp = Path(tmp)
            for name in ("models", "workloads", "configs"):
                (tmp / name).symlink_to(ROOT / name)
            (tmp / "build").mkdir()
            (tmp / "build/dramsim3main").symlink_to(self.binary)
            result = subprocess.run(
                [sys.executable, str(ROOT / "run_demo.py"), "-m", "OPT-2.7B",
                 "-i", "128", "-o", "16", "-b", "32"],
                cwd=tmp, capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            book = openpyxl.load_workbook(tmp / "result.xlsx", data_only=True)
            sheet = book["OPT-2.7B_128_16_32"]
            for row in (2, 3, 4, 5):
                for column in ("B", "C", "D", "E", "F", "G", "H"):
                    self.assertGreater(sheet[f"{column}{row}"].value, 0)
            self.assertGreater(sheet["I4"].value, 0)
            book.close()
            logs = list((tmp / "logs").glob("*.log"))
            self.assertEqual(len(logs), 2)
            launches = [line for log in logs for line in log.read_text().splitlines()
                        if line.startswith("/tmp/hbnpu-opt-")]
            self.assertEqual(len(launches), 46)
            self.assertFalse((tmp / "traces").exists())

    def test_capped_kernel_is_not_reported_as_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = ROOT / "configs/HBNPU_8Gb.ini"
            trace = tmp / "kernel.trace"
            write_trace(compile_kernel(load_workload(
                ROOT / "workloads/OPT-2.7B/decode/createQKV", config)), trace)
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                with simulate_kernel(trace, config, tmp / "run.log",
                                     binary=self.binary, cycles=1):
                    self.fail("incomplete kernel exposed result statistics")


if __name__ == "__main__":
    OptEvaluationTest.binary = Path(sys.argv.pop(1)).resolve()
    unittest.main()
