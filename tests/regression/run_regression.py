#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/HBM2_8Gb_x128_pim.ini"
QUICK_CASES = {
    "ws_gemm_prompt": ROOT / "tests/regression/fixtures/ws_gemm_prompt.trace",
    "decode_gemv": ROOT / "tests/regression/fixtures/decode_gemv.trace",
    "attention_qk": ROOT / "tests/regression/fixtures/attention_qk.trace",
    "attention_sv": ROOT / "tests/regression/fixtures/attention_sv.trace",
    "refresh_boundary": ROOT / "tests/regression/fixtures/refresh_boundary.trace",
}


def opt27b_cases():
    base = ROOT / "traces/OPT-2.7B_128_1024_32"
    cases = []
    for name in ("createQKV", "QK", "SV", "Wo", "L1", "L2"):
        cases.append(("prompt/" + name, base / "prompt" / name))
    for name in ("createQKV", "Wo", "L1", "L2"):
        cases.append(("decode/" + name, base / "decode" / name))
        cases.append(("decode/WS/" + name, base / "decode" / "WS" / name))
    for sequence in range(128, 144):
        for operation in ("QK", "SV"):
            filename = "{}_{:04d}".format(operation, sequence)
            cases.append(("decode/QKV/" + filename,
                          base / "decode" / "QKV" / filename))
    return cases


def parse_command_trace(path):
    commands = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise AssertionError("invalid command trace line: " + line)
        commands.append({
            "cycle": int(fields[0]),
            "kind": fields[1],
            "channel": int(fields[2]),
            "rank": int(fields[3]),
            "bankgroup": int(fields[4]),
            "bank": int(fields[5]),
            "row": int(fields[6], 16),
            "column": int(fields[7], 16),
        })
    return commands


def run_case(binary, trace):
    with tempfile.TemporaryDirectory(prefix="hbnpu-regression-") as temp_name:
        output_dir = Path(temp_name)
        completed = subprocess.run(
            [
                str(binary), str(CONFIG), "-c", "10000000",
                "-t", str(trace), "-o", str(output_dir) + os.sep,
            ],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            raise AssertionError(completed.stdout)
        if "Turn off PIM" not in completed.stdout:
            raise AssertionError("simulation did not terminate: " + str(trace))

        stats_path = output_dir / "dramsim3.json"
        command_paths = [
            output_dir / "dramsim3ch_{}cmd.trace".format(channel)
            for channel in range(8)
        ]
        missing = [path for path in [stats_path] + command_paths if not path.exists()]
        if missing:
            raise AssertionError(
                "missing regression output; configure with -DCMD_TRACE=ON: " +
                ", ".join(str(path) for path in missing)
            )
        stats = json.loads(stats_path.read_text())
        commands = {
            str(channel): parse_command_trace(command_paths[channel])
            for channel in range(8)
        }
        return {
            "terminated": True,
            "termination_cycle": int(stats["0"]["num_cycles"]),
            "commands": commands,
            "stats": stats,
        }


def energy_close(actual, expected):
    return abs(actual - expected) <= (
        1e-9 + 1e-6 * max(abs(actual), abs(expected))
    )


def compare_commands(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError("command channel mismatch")
    for channel in sorted(actual, key=int):
        if len(actual[channel]) != len(expected[channel]):
            raise AssertionError("command count mismatch in channel " + channel)
        for index, (actual_command, expected_command) in enumerate(
                zip(actual[channel], expected[channel])):
            if actual_command != expected_command:
                raise AssertionError(
                    "first command mismatch: channel {}, index {}, "
                    "expected cycle {}, actual cycle {}, expected {!r}, actual {!r}".format(
                        channel,
                        index,
                        expected_command["cycle"],
                        actual_command["cycle"],
                        expected_command,
                        actual_command,
                    )
                )


def compare_value(actual, expected, path):
    key = path[-1] if path else ""
    if path == ["commands"]:
        compare_commands(actual, expected)
        return
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError("key mismatch at " + "/".join(path))
        for child_key in sorted(actual):
            compare_value(actual[child_key], expected[child_key], path + [child_key])
        return
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("length mismatch at " + "/".join(path))
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            compare_value(actual_item, expected_item, path + [str(index)])
        return
    if "energy" in key and isinstance(actual, (int, float)):
        if not energy_close(float(actual), float(expected)):
            raise AssertionError("energy mismatch at " + "/".join(path))
        return
    if actual != expected:
        raise AssertionError(
            "mismatch at {}: expected {!r}, actual {!r}".format(
                "/".join(path), expected, actual
            )
        )


def write_golden(path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with path.open("wb") as raw_file:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file:
            gzip_file.write(payload)


def read_golden(path):
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


def run_quick(binary, update_golden):
    golden_dir = ROOT / "tests/regression/golden"
    for name, trace in QUICK_CASES.items():
        actual = run_case(binary, trace)
        if name == "refresh_boundary" and actual["termination_cycle"] <= 3900:
            raise AssertionError("refresh fixture did not cross tREFI")
        golden_path = golden_dir / (name + ".json.gz")
        if update_golden:
            repeated = [actual, run_case(binary, trace), run_case(binary, trace)]
            if repeated[0] != repeated[1] or repeated[0] != repeated[2]:
                raise AssertionError("unstable baseline: " + name)
            write_golden(golden_path, actual)
        else:
            expected = read_golden(golden_path)
            compare_value(actual, expected, [])
        print("PASS " + name)


def summarize_long_case(full_result):
    command_sha256 = {}
    for channel, commands in full_result["commands"].items():
        canonical = json.dumps(
            commands, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        command_sha256[channel] = hashlib.sha256(canonical).hexdigest()
    return {
        "terminated": full_result["terminated"],
        "termination_cycle": full_result["termination_cycle"],
        "command_sha256": command_sha256,
        "stats": full_result["stats"],
    }


def run_opt27b_once(binary):
    kernels = {}
    for name, trace in opt27b_cases():
        kernels[name] = summarize_long_case(run_case(binary, trace))
    return {
        "model": "OPT-2.7B",
        "input_tokens": 128,
        "output_tokens": 16,
        "batch_size": 32,
        "kernels": kernels,
    }


def run_opt27b(binary, update_golden):
    golden_path = ROOT / "tests/regression/golden/opt27b_128_16_32.json"
    actual = run_opt27b_once(binary)
    if update_golden:
        second = run_opt27b_once(binary)
        third = run_opt27b_once(binary)
        if actual != second or actual != third:
            raise AssertionError("unstable OPT-2.7B baseline")
        golden_path.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n"
        )
    else:
        expected = json.loads(golden_path.read_text())
        compare_value(actual, expected, [])
    print("PASS opt27b_128_16_32")


parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True, type=Path)
parser.add_argument("--suite", choices=("quick", "opt27b"), required=True)
parser.add_argument("--update-golden", action="store_true")


def main():
    args = parser.parse_args()
    if args.suite == "quick":
        run_quick(args.binary, args.update_golden)
    if args.suite == "opt27b":
        run_opt27b(args.binary, args.update_golden)


if __name__ == "__main__":
    main()
