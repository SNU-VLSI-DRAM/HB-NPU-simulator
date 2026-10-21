"""Execute OPT instruction programs and validate data streams and row timing.

Five-record fixtures are decoded only as test input metadata; the simulator
receives CONFIG/STORE/LOAD_INPUTS/TILE_GEMM/END packets in every run.
"""
import argparse
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import compile_kernel, write_trace
import run_regression as reference
from compare_instruction_execution import (
    source_context, run_one, validate_timing, validate_rows, compare_data_streams,
)


def run_case(binary, trace, hardware=reference.CONFIG, fused=False):
    context, start, _ = source_context(trace, hardware)
    with tempfile.TemporaryDirectory(prefix="hbnpu-tile-regression-") as tmp:
        tmp = Path(tmp)
        program = tmp / "program.trace"
        write_trace(compile_kernel(context, fused=fused), program, start_cycle=start)
        commands, issues, complete, cycles, metrics, rows = run_one(
            Path(binary).resolve(), hardware, program, tmp / "run", 10000000, 120)
        if not complete:
            raise AssertionError("incomplete instruction program: " + str(trace))
        errors = validate_timing(commands, issues, hardware)
        errors += validate_rows(commands, issues, rows, hardware, complete)
        if errors:
            raise AssertionError(str(errors[:3]))
        if metrics["decoded_instructions"] != metrics["retired_instructions"]:
            raise AssertionError("instruction retirement mismatch")
        return dict(commands=commands, pim_issues=issues, terminated=complete,
                    termination_cycle=cycles, metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--suite", choices=("quick", "opt27b"), required=True)
    args = parser.parse_args()
    cases = (reference.QUICK_CASES.items() if args.suite == "quick"
             else reference.opt27b_cases())
    for name, trace in cases:
        hardware = (reference.CONFIG if args.suite == "quick"
                    else ROOT / "configs/HBNPU_8Gb.ini")
        actual = run_case(args.binary, trace, hardware)
        if args.suite == "quick":
            golden = reference.read_golden(
                ROOT / "tests/regression/golden" / (name + ".json.gz"))
            # Timing intentionally changes with instruction submission/row locking.
            expected = golden["commands"]
        else:
            expected = run_case(args.binary, trace, hardware, fused=True)["commands"]
        comparison = compare_data_streams(expected, actual["commands"], True, True)
        if not comparison["data_equal"]:
            raise AssertionError(str(comparison))
        print("PASS " + name, flush=True)


if __name__ == "__main__":
    main()
