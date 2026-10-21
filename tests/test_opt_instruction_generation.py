"""OPT CLI must emit instruction packets, not matrix-configuration transactions."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shlex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import decode_packet


class OptInstructionsTest(unittest.TestCase):
    def test_default_model_output_preserves_supplied_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for name in ("models", "workloads", "configs"):
                (tmp / name).symlink_to(ROOT / name)
            paper = tmp / "traces/OPT-2.7B_128_1_32/prompt/createQKV"
            paper.parent.mkdir(parents=True)
            paper.write_text("supplied paper trace\n")
            result = subprocess.run(
                [sys.executable, str(ROOT / "gen_LLM_trace.py"),
                 "-m", "OPT-2.7B", "-i", "128", "-o", "1", "-b", "32"],
                cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(paper.read_text(), "supplied paper trace\n")
            generated = tmp / "output/traces/OPT-2.7B_128_1_32/prompt/createQKV"
            self.assertEqual(int(generated.read_text().split()[0], 16) >> 56, 0xB7)

    def test_existing_trace_is_preserved_without_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "program.trace"
            output.write_text("preserve this input\n")
            command = [
                sys.executable, str(ROOT / "gen_LLM_trace.py"), "-f", "False",
                "-w", str(ROOT / "workloads/OPT-2.7B/decode/createQKV"),
                "-s", str(ROOT / "configs/HBNPU_8Gb.ini"), "-t", str(output),
            ]
            result = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(output.read_text(), "preserve this input\n")
            subprocess.run(command + ["--overwrite"], cwd=ROOT,
                           check=True, capture_output=True)
            self.assertEqual(int(output.read_text().split()[0], 16) >> 56, 0xB7)

    def test_paper_model_batch_matrix_generates_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            for line in (ROOT / "run_models.sh").read_text().splitlines():
                if not line.strip():
                    continue
                args = shlex.split(line)[2:]
                # All 24 model/input/batch combinations; one decode step each.
                args[args.index("-o") + 1] = "1"
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "gen_LLM_trace.py"),
                         "-p", tmp] + args, cwd=ROOT, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            programs = [p for p in Path(tmp).rglob("*") if p.is_file()]
            self.assertEqual(len(programs), 24 * 16)
            for program in programs:
                with program.open() as stream:
                    self.assertEqual(int(stream.readline().split()[0], 16) >> 56, 0xB7)

    def test_file_generation_emits_config_load_gemm_store_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "program.trace"
            subprocess.run([
                sys.executable, str(ROOT / "gen_LLM_trace.py"), "-f", "False",
                "-w", str(ROOT / "workloads/OPT-2.7B/decode/createQKV"),
                "-s", str(ROOT / "configs/HBNPU_8Gb.ini"), "-t", str(output),
            ], cwd=ROOT, check=True, capture_output=True)
            words = [int(line.split()[0], 16)
                     for line in output.read_text().splitlines()]
            self.assertEqual(words[0] >> 56, 0xB7,
                             "OPT generator emitted the removed transaction format")
            opcodes = []
            while words:
                length = (words[0] >> 40) & 0xFF
                packet = decode_packet(words[:length + 1])
                opcodes.append(packet.opcode)
                words = words[length + 1:]
            self.assertEqual(opcodes[0], 1)
            self.assertEqual(opcodes[-1], 4)
            self.assertTrue({2, 5, 6}.issubset(opcodes))


if __name__ == "__main__":
    unittest.main()
