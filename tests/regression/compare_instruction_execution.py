"""Compare explicit instruction execution with legacy data streams and timing.

Unlike compare_binaries.py, this accepts cycle changes, never data changes.
"""
import argparse
import collections
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys

import run_regression as regression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import KernelConfig, compile_kernel, load_workload, write_trace

DATA = {"gh_read", "gh_read_p", "lh_read", "lh_read_p", "pim_write", "pim_write_p"}
KINDS = {
    "GH_READ": "gh_read",
    "GH_READ_PRECHARGE": "gh_read_p",
    "LH_READ": "lh_read",
    "LH_READ_PRECHARGE": "lh_read_p",
    "GH_WRITE": "pim_write",
    "GH_WRITE_PRECHARGE": "pim_write_p",
    "LH_WRITE": "pim_write",
    "LH_WRITE_PRECHARGE": "pim_write_p",
    "PIM_ACT": "pim_activate",
    "GANG_ACT": "pim_activate",
    "PIM_PRE": "precharge",
    "GANG_PRE": "precharge",
}


def compare_data_streams(reference, candidate, reference_complete, candidate_complete):
    def streams(commands):
        result = collections.defaultdict(list)
        for channel in commands.values():
            for c in channel:
                if c["kind"] in DATA:
                    key = tuple(c[k] for k in ("channel", "rank", "bankgroup", "bank"))
                    result[key].append((c["kind"], c["row"], c["column"]))
        return result

    left, right = streams(reference), streams(candidate)
    complete = reference_complete and candidate_complete
    report = dict(
        data_equal=True,
        full_completion=complete,
        first_difference=None,
        reference_count=sum(map(len, left.values())),
        candidate_count=sum(map(len, right.values())),
        per_bank_counts={},
    )
    for bank in sorted(left.keys() | right.keys()):
        a, b = left[bank], right[bank]
        report["per_bank_counts"][str(bank)] = [len(a), len(b)]
        limit = max(len(a), len(b)) if complete else min(len(a), len(b))
        for i in range(limit):
            av = a[i] if i < len(a) else None
            bv = b[i] if i < len(b) else None
            if av != bv:
                report["data_equal"] = False
                if report["first_difference"] is None:
                    report["first_difference"] = dict(
                        bank=bank, index=i, reference=av, candidate=bv
                    )
                break
    return report


def validate_timing(commands, issues, config):
    """Replay PIM bank state and timing constraints from src/timing.cc.

    This auditor covers the HBM PIM-only workload profile, explicit/automatic
    precharge, rank/bank refresh, channel issue width, cadence and tFAW. It
    rejects unknown command kinds rather than silently claiming validation.
    """
    ini = configparser.ConfigParser()
    ini.read(config)
    t = ini["timing"]
    d = ini["dram_structure"]
    get = lambda name, default: int(t.get(name, default))
    banks = int(d["banks_per_group"])
    groups = int(d["bankgroups"])
    burst = (int(d["BL"]) // 2) if int(d["BL"]) else 0
    al = get("AL", 0)
    wl = al + get("CWL", 12)
    rl = al + get("CL", 12)
    rp = get("tRP", 14)
    ras = get("tRAS", 34)
    rtp = al + get("tRTP", 5)
    wr = wl + burst + get("tWR", 16)
    cadence = max(burst, get("tCCD_L", 2))
    errors = []

    def fail(channel, cycle, message):
        if len(errors) < 30:
            errors.append("channel {} cycle {}: {}".format(channel, cycle, message))

    for channel, trace in commands.items():
        states = {}
        windows = collections.defaultdict(collections.deque)

        def state(rank, bg, bank):
            return states.setdefault(
                (rank, bg, bank),
                dict(row=None, act=0, pre=0, ref=0, **{kind: 0 for kind in DATA}),
            )

        effects = collections.defaultdict(list)
        for c in trace:
            effects[c["cycle"]].append(c)
        issue_cycles = set()
        next_global = 0
        for issue in issues.get(channel, []):
            cycle = issue["cycle"]
            kind = issue["kind"]
            targets = [tuple(b) for b in issue["targets"]]
            if cycle in issue_cycles:
                fail(channel, cycle, "multiple logical issues")
            issue_cycles.add(cycle)
            width = (
                groups * banks
                if kind.startswith("LH_")
                else 4
                if kind.startswith("GANG_")
                else 1
            )
            actual = effects[cycle]
            if (
                kind not in KINDS
                or len(targets) != width
                or len(set(targets)) != width
                or len(actual) != width
                or {(c["bankgroup"], c["bank"]) for c in actual} != set(targets)
                or any(
                    c["kind"] != KINDS.get(kind) or c["rank"] != issue["rank"]
                    for c in actual
                )
            ):
                fail(channel, cycle, "logical broadcast/effect mismatch")
            if kind.startswith("GH_"):
                if cycle < next_global:
                    fail(channel, cycle, "GH channel cadence")
                next_global = cycle + max(1, get("tCCD_S", 1))
        last = -1
        for c in trace:
            cycle = c["cycle"]
            kind = c["kind"]
            rank = c["rank"]
            bg = c["bankgroup"]
            bank = c["bank"]
            if cycle < last:
                fail(channel, cycle, "unordered trace")
            last = cycle
            if kind in DATA or kind == "pim_activate":
                if cycle not in issue_cycles:
                    fail(channel, cycle, "missing logical PIM issue")
            if kind in ("refresh", "refresh_bank"):
                targets = (
                    [(bg, bank)]
                    if kind == "refresh_bank"
                    else [(g, b) for g in range(groups) for b in range(banks)]
                )
                for g, b in targets:
                    s = state(rank, g, b)
                    if s["row"] is not None or cycle < s["ref"]:
                        fail(channel, cycle, "refresh before bank ready")
                    delay = (
                        get("tRFCb", get("tRFC", 260))
                        if kind == "refresh_bank"
                        else get("tRFC", 260)
                    )
                    s["act"] = max(s["act"], cycle + delay)
                    s["ref"] = max(s["ref"], cycle + delay)
                continue
            if not (0 <= bg < groups and 0 <= bank < banks):
                fail(channel, cycle, "bank address outside topology")
                continue
            s = state(rank, bg, bank)
            if kind in ("pim_activate", "activate"):
                if s["row"] is not None or cycle < s["act"]:
                    fail(channel, cycle, "ACT before bank ready")
                live = windows[rank]
                while live and live[0] + get("tFAW", 30) <= cycle:
                    live.popleft()
                live.append(cycle)
                if len(live) > 4:
                    fail(channel, cycle, "tFAW exceeded")
                s["row"] = c["row"]
                s["act"] = max(s["act"], cycle + ras + rp)
                s["pre"] = max(s["pre"], cycle + ras)
                for data in DATA:
                    s[data] = max(
                        s[data],
                        cycle
                        + get(
                            "tRCDWR" if data.startswith("pim_write") else "tRCDRD", 14
                        ),
                    )
            elif kind == "precharge":
                if s["row"] is None or cycle < s["pre"]:
                    fail(channel, cycle, "PRE before bank ready")
                s["row"] = None
                s["act"] = max(s["act"], cycle + rp)
                s["ref"] = max(s["ref"], cycle + rp)
            elif kind in DATA:
                if s["row"] != c["row"] or cycle < s[kind]:
                    fail(
                        channel,
                        cycle,
                        "data row/timing mismatch at bank {}:{}".format(bg, bank),
                    )
                writing = kind.startswith("pim_write")
                pre_delay = wr if writing else rtp
                if kind.endswith("_p"):
                    s["row"] = None
                    s["act"] = max(s["act"], cycle + pre_delay + rp)
                    s["ref"] = max(s["ref"], cycle + pre_delay + rp)
                else:
                    s["pre"] = max(s["pre"], cycle + pre_delay)
                    for data in DATA:
                        target_write = data.startswith("pim_write")
                        if writing == target_write:
                            delay = cadence
                        elif writing:
                            delay = (
                                wl
                                + burst
                                + get("tWTR_L" if data.endswith("_p") else "tWTR_S", 0)
                            )
                        else:
                            delay = rl + burst - wl + get("tRTRS", 0)
                        s[data] = max(s[data], cycle + delay)
            else:
                fail(channel, cycle, "unsupported audit command " + kind)
    return errors


def legacy_context(trace, config):
    """Test-only launch extraction; never consumes observed DRAM commands."""
    records = [
        line.split() for line in Path(trace).read_text().splitlines() if line.strip()
    ]
    if len(records) != 5 or any(row[1] != "PIM" for row in records):
        raise ValueError("comparison requires a five-record legacy launch")
    words = [int(row[0], 16) for row in records]
    flags = words[0]
    ini = configparser.ConfigParser()
    ini.read(config)
    d = ini["dram_structure"]
    bl = int(d["BL"]) or 4
    c = KernelConfig(
        (words[1] >> 3) & 0xFFFFFFFF,
        (words[3] >> 3) & 0xFFFFFFFF,
        (words[2] >> 3) & 0xFFFFFFFF,
        1 << ((flags >> 10) & 15),
        words[3] >> 35,
        words[1] >> 35,
        words[2] >> 35,
        (flags >> 9) & 1,
        1 << ((flags >> 3) & 7),
        1 << ((flags >> 6) & 7),
        int(ini["system"]["channels"]),
        int(d["bankgroups"]) * int(d["banks_per_group"]),
        int(d["columns"]) * 2 // bl,
    )
    return c, int(records[0][2])


def source_context(trace, config):
    reference, start = legacy_context(trace, config)
    relative = Path(trace).resolve().relative_to(ROOT)
    if relative.parts[0] != "traces":
        return reference, start, "synthetic legacy launch"
    model, inputs, outputs, batch = relative.parts[1].split("_")
    tail = list(relative.parts[2:])
    if tail[0] == "prompt":
        tail.insert(1, format(int(inputs), "04d"))
    if tail[:2] == ["decode", "WS"]:
        model_fields = next(
            line.split()
            for line in (ROOT / "models").read_text().splitlines()
            if line.split()[0] == model
        )
        tail[-1] = format(int(batch) // int(model_fields[-1]), "04d") + "_" + tail[-1]
    workload = ROOT / "workloads" / model / Path(*tail)
    context = load_workload(workload, config, reference.output_row)
    if context != reference:
        raise ValueError("workload CONFIG differs from legacy launch: " + str(workload))
    return context, start, str(workload)


def run_one(binary, config, trace, output, cycles, timeout):
    output.mkdir()
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    with (output / "run.log").open("w") as log:
        result = subprocess.run(
            [
                str(binary),
                str(config),
                "-t",
                str(trace),
                "-c",
                str(cycles),
                "-o",
                str(output),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    if result.returncode:
        raise RuntimeError("simulator failed: " + str(output / "run.log"))
    channels = int(configparser_value(config, "system", "channels"))
    commands = {
        str(ch): regression.parse_command_trace(
            output / ("dramsim3ch_{}cmd.trace".format(ch))
        )
        for ch in range(channels)
    }
    issues = {
        str(ch): regression.parse_pim_issue_trace(
            output / ("dramsim3ch_{}pim.trace".format(ch))
        )
        for ch in range(channels)
    }
    stats = json.loads((output / "dramsim3.json").read_text())
    metrics_path = output / "hbnpu_instruction.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    complete = metrics.get(
        "end_completed", "Turn off PIM" in (output / "run.log").read_text()
    )
    return commands, issues, complete, int(stats["0"]["num_cycles"]), metrics


def configparser_value(path, section, key):
    config = configparser.ConfigParser()
    config.read(path)
    return config[section][key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-binary", type=Path, required=True)
    parser.add_argument("--candidate-binary", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=regression.CONFIG)
    parser.add_argument("--suite", choices=("quick", "opt27b"), default="quick")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cycles", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir()
    args.config = args.config.resolve()
    config = configparser.ConfigParser()
    config.read(args.config)
    config["hbnpu"] = {
        "execution_mode": "tile_instructions",
        "instruction_capacity": "16",
    }
    instruction_config = args.output_dir / "instruction.ini"
    with instruction_config.open("w") as out:
        config.write(out)
    cases = (
        json.loads(args.manifest.read_text())
        if args.manifest
        else [
            dict(name=n, trace=str(p), cycles=args.cycles)
            for n, p in (
                regression.QUICK_CASES.items()
                if args.suite == "quick"
                else regression.opt27b_cases()
            )
        ]
    )
    reports = []
    for index, case in enumerate(cases):
        directory = args.output_dir / str(index)
        directory.mkdir()
        report = dict(name=case["name"], passed=False)
        try:
            context, start, source = source_context(Path(case["trace"]), args.config)
            report["source"] = source
            program = directory / "instructions.trace"
            write_trace(compile_kernel(context), program, start)
            cycles = case.get("cycles", args.cycles)
            a = run_one(
                args.baseline_binary.resolve(),
                args.config,
                Path(case["trace"]),
                directory / "reference",
                cycles,
                args.timeout,
            )
            b = run_one(
                args.candidate_binary.resolve(),
                instruction_config,
                program,
                directory / "candidate",
                cycles,
                args.timeout,
            )
            report.update(compare_data_streams(a[0], b[0], a[2], b[2]))
            report.update(
                reference_complete=a[2],
                candidate_complete=b[2],
                reference_cycles=a[3],
                candidate_cycles=b[3],
                cycle_delta=b[3] - a[3],
                instruction_metrics=b[4],
            )
            report["reference_timing_errors"] = validate_timing(a[0], a[1], args.config)
            report["candidate_timing_errors"] = validate_timing(b[0], b[1], args.config)
            report["passed"] = (
                report["data_equal"]
                and not report["candidate_timing_errors"]
                and not report["reference_timing_errors"]
            )
        except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
            report["error"] = str(error)
        reports.append(report)
        (args.output_dir / "report.json").write_text(
            json.dumps(reports, indent=2) + "\n"
        )
        print(
            case["name"],
            "PASS" if report["passed"] else "FAIL",
            "complete" if report.get("full_completion") else "incomplete",
            "delta",
            report.get("cycle_delta"),
            report.get("error", ""),
            flush=True,
        )
    return 0 if all(r["passed"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
