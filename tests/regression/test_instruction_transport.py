"""Exercise the actual trace CPU and simulator, including EOF under backpressure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import KernelConfig, compile_kernel, encode_packet, write_trace


class TransportTest(unittest.TestCase):
    binary = None

    def test_tile_submission_really_executes_row_operations(self):
        result, stats, _, _ = self.run_program()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(stats["end_completed"])
        self.assertEqual(stats.get("row_operations_completed"), 4)
        self.assertEqual(stats.get("row_operations_launched"), 4)

    def run_program(
        self,
        *,
        period=1,
        end=True,
        truncated=False,
        capacity="1",
        mode="tile_instructions",
        stress=False,
        cycles=100000,
        newline=True
    ):
        with tempfile.TemporaryDirectory(prefix="hbnpu-transport-") as folder:
            folder = Path(folder)
            config = folder / "config.ini"
            trace = folder / "program"
            config.write_text(
                (ROOT / "configs/HBM2_8Gb_x128_pim.ini").read_text()
                + "\n[hbnpu]\nexecution_mode = {}\ninstruction_capacity = {}\n".format(
                    mode, capacity
                )
            )
            c = KernelConfig(
                256 if stress else 64,
                128,
                2048 if stress else 256,
                64,
                2133,
                0,
                2133,
                0,
                2,
                1,
                8,
                16,
                32,
            )
            packets = list(compile_kernel(c))
            if not end:
                packets = packets[:-1]
            write_trace(packets, trace, word_period=period)
            lines = trace.read_text().splitlines()
            if truncated:
                lines = lines[:3]
            trace.write_text("\n".join(lines) + ("\n" if newline else ""))
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            result = subprocess.run(
                [
                    str(self.binary),
                    str(config),
                    "-c",
                    str(cycles),
                    "-t",
                    str(trace),
                    "-o",
                    str(folder),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            path = folder / "hbnpu_instruction.json"
            return (
                result,
                json.loads(path.read_text()) if path.exists() else None,
                len(lines),
                len(packets),
            )

    def test_full_fifo_retries_last_word_without_newline(self):
        result, stats, words, packets = self.run_program(stress=True, newline=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(stats, result.stdout)
        self.assertTrue(stats["end_completed"])
        self.assertEqual(stats["admitted_words"], words)
        self.assertEqual(stats["decoded_instructions"], packets)
        self.assertEqual(stats["retired_instructions"], packets)
        self.assertGreater(stats["cpu_backpressure_cycles"], 0)
        self.assertLessEqual(stats["raw_high_watermark"], 32)
        self.assertEqual(stats["slot_high_watermark"], 1)

    def test_slow_upload_is_measured_without_changing_completion(self):
        result, stats, words, packets = self.run_program(period=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNotNone(stats)
        self.assertTrue(stats["end_completed"])
        self.assertGreater(stats["instruction_starvation_cycles"], 0)
        self.assertEqual(stats["admitted_words"], words)
        self.assertGreaterEqual(stats["end_cycle"], stats["backend_drain_cycle"])

    def test_missing_end_and_partial_config_report_incomplete(self):
        for partial in (False, True):
            result, stats, _, _ = self.run_program(
                end=False, truncated=partial, cycles=10000
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(stats)
            self.assertFalse(stats["end_completed"])
            self.assertEqual(stats["partial_packet"], partial)
            self.assertNotIn("Turn off PIM", result.stdout)

    def test_invalid_mode_and_capacity_fail(self):
        for mode, cap in [
            ("wrong", "16"),
            ("tile_instructions", "0"),
            ("tile_instructions", "-1"),
            ("tile_instructions", "oops"),
        ]:
            result, _, _, _ = self.run_program(mode=mode, capacity=cap, cycles=1)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    TransportTest.binary = Path(sys.argv.pop(1)).resolve()
    unittest.main()
