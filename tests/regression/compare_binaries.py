"""Compare bounded simulator executions exactly; never updates golden files."""

import argparse
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import subprocess

from run_regression import compare_value


def first_difference(expected, actual):
    for index, (left, right) in enumerate(zip_longest(expected, actual)):
        if left != right:
            return index, left, right
    return None


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_outputs(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    names = [f"dramsim3ch_{ch}{kind}.trace" for ch in range(8) for kind in ("cmd", "pim")]
    for directory in (baseline, candidate):
        for name in names + ["dramsim3.json", "stdout.log"]:
            if not (directory / name).is_file():
                raise AssertionError("missing output: " + str(directory / name))
    counts = {"cmd": 0, "pim": 0}
    hashes = {}
    for name in names:
        with (baseline / name).open() as left, (candidate / name).open() as right:
            mismatch = first_difference(left, right)
        if mismatch is not None:
            index, expected, actual = mismatch
            raise AssertionError(f"first difference {name}, index {index}: expected {expected!r}, actual {actual!r}")
        kind = "cmd" if name.endswith("cmd.trace") else "pim"
        with (baseline / name).open() as stream:
            counts[kind] += sum(1 for _ in stream)
        hashes[name] = fingerprint(baseline / name)
    expected_stats = json.loads((baseline / "dramsim3.json").read_text())
    actual_stats = json.loads((candidate / "dramsim3.json").read_text())
    for stats in (expected_stats, actual_stats):
        for ch in range(8):
            if str(ch) not in stats or "num_cycles" not in stats[str(ch)]:
                raise AssertionError(f"missing channel {ch} cycle statistics")
    compare_value(actual_stats, expected_stats, ["stats"])
    marker = "Output Exhausted: Array0. Turn off PIM mode."
    completed = marker in (baseline / "stdout.log").read_text()
    if completed != (marker in (candidate / "stdout.log").read_text()):
        raise AssertionError("completion mismatch between baseline and candidate")
    return {"completed": completed, "cycles": expected_stats["0"]["num_cycles"],
            "physical_records": counts["cmd"], "logical_records": counts["pim"],
            "trace_sha256": hashes}


def run_comparison(baseline, candidate, trace, config, output, cycles=100000, timeout=60):
    baseline, candidate, trace, config, output = (
        Path(p).resolve() for p in (baseline, candidate, trace, config, output))
    if cycles <= 0 or timeout <= 0:
        raise ValueError("cycles and timeout must be positive")
    for path in (baseline, candidate, trace, config):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    # Refuse reuse: a prior successful trace must never mask a failed rerun.
    output.mkdir(parents=True, exist_ok=False)
    result = {"passed": False, "baseline": str(baseline), "candidate": str(candidate),
              "trace": str(trace), "config": str(config), "cycle_limit": cycles,
              "baseline_sha256": fingerprint(baseline), "candidate_sha256": fingerprint(candidate),
              "input_sha256": fingerprint(trace), "config_sha256": fingerprint(config)}
    try:
        for label, binary in (("baseline", baseline), ("candidate", candidate)):
            directory = output / label
            directory.mkdir()
            command = [str(binary), str(config), "-c", str(cycles), "-t", str(trace),
                       "-o", str(directory) + os.sep]
            env = dict(os.environ)
            env.pop("LD_LIBRARY_PATH", None)  # use each isolated build's own RPATH
            with (directory / "stdout.log").open("w") as log:
                try:
                    run = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, timeout=timeout)
                except subprocess.TimeoutExpired as error:
                    raise AssertionError(f"{label} timeout after {timeout}s") from error
            if run.returncode != 0:
                raise AssertionError(f"{label} exit code {run.returncode}; see {directory / 'stdout.log'}")
        result.update(compare_outputs(output / "baseline", output / "candidate"))
        result["passed"] = True
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "trace", "config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cycles", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    result = run_comparison(**vars(args))
    status = "complete" if result["completed"] else "bounded, not complete"
    print(f"PASS {args.trace}: {result['cycles']} cycles ({status}); "
          f"{result['physical_records']} physical / {result['logical_records']} logical records")


if __name__ == "__main__":
    main()
