import importlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
decode = importlib.import_module("gen_workload_decode")
prompt = importlib.import_module("gen_workload_prompt")
trace_generator = importlib.import_module("gen_LLM_trace")


class WorkloadFormatTest(unittest.TestCase):
    def test_rejects_legacy_header_without_overwriting_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            workload = Path(folder) / "workload"
            trace = Path(folder) / "trace"
            workload.write_text("1, 1, 2048, 8, 2, 8, 1\n80, 2560, 8\n")
            trace.write_text("existing trace\n")
            with self.assertRaisesRegex(ValueError, "five-field header"):
                trace_generator.gen_pim_trace(workload, trace, -1)
            self.assertEqual(trace.read_text(), "existing trace\n")

    def test_row_override_matches_decode_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / "trace"
            trace_generator.gen_pim_trace(
                ROOT / "workloads/OPT-2.7B/decode/createQKV", trace, 2133)
            actual = [line.split() for line in trace.read_text().splitlines()]
            fixture = ROOT / "tests/regression/fixtures/decode_gemv.trace"
            self.assertEqual(actual, [line.split() for line in
                                      fixture.read_text().splitlines()])

    def test_rejects_additional_workload_tuple(self):
        with tempfile.TemporaryDirectory() as folder:
            workload = Path(folder) / "workload"
            trace = Path(folder) / "trace"
            workload.write_text("2048, 8, 2, 8, 1\n80, 2560, 8\n80, 2560, 8\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                trace_generator.gen_pim_trace(workload, trace, -1)
            self.assertFalse(trace.exists())

    def test_generators_write_five_field_headers(self):
        with tempfile.TemporaryDirectory() as folder:
            decode.gen_actual_workload(folder, "gemv", 2048, 8,
                                       80, 2560, 8, 2, 8)
            decode.gen_actual_workload_ws(folder, "ws", 2048, 8,
                                          128, 128, 128, 2)
            prompt.gen_actual_workload_ws(folder, "prompt", 2048, 8,
                                          128, 128, 128, 2)
            self.assertEqual((Path(folder) / "gemv").read_text(),
                             "2048, 8, 2, 8, 1\n80, 2560, 8\n")
            for name in ("ws", "prompt"):
                self.assertEqual((Path(folder) / name).read_text(),
                                 "2048, 8, 2, 1, 0\n128, 128, 128\n")

    def test_file_mode_encodes_compact_transactions(self):
        cases = [
            ("2048, 8, 2, 1, 0\n128, 128, 128\n",
             [0x2c0e, 0x200, 0x8800000402, 0x8800000404, 0x1]),
            ("2048, 8, 2, 8, 1\n80, 2560, 8\n",
             [0x2ece, 0x64800000140, 0x64800000a02, 0x204, 0x1]),
        ]
        with tempfile.TemporaryDirectory() as folder:
            workload = Path(folder) / "workload"
            trace = Path(folder) / "trace"
            for contents, addresses in cases:
                with self.subTest(workload=contents):
                    workload.write_text(contents)
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "gen_LLM_trace.py"),
                         "-f", "False", "-w", str(workload), "-t", str(trace)],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(
                        trace.read_text(),
                        "".join("{}\tPIM\t{}\n".format(hex(addr), cycle)
                                for cycle, addr in enumerate(addresses)),
                    )


if __name__ == "__main__":
    unittest.main()
