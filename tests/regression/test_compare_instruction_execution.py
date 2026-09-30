import copy
from pathlib import Path
import unittest
from compare_instruction_execution import compare_data_streams, validate_timing

CONFIG = Path(__file__).resolve().parents[2] / "configs/HBM2_8Gb_x128_pim.ini"


def command(cycle, kind, bank=0, row=0, column=0):
    return dict(
        cycle=cycle,
        kind=kind,
        channel=0,
        rank=0,
        bankgroup=0,
        bank=bank,
        row=row,
        column=column,
    )


def issue(cycle, kind, targets):
    return dict(cycle=cycle, channel=0, rank=0, kind=kind, targets=targets)


class ComparisonTest(unittest.TestCase):
    def test_cycle_shift_does_not_hide_data_changes(self):
        reference = {"0": [command(i * 2, "gh_read", column=i) for i in range(8)]}
        candidate = copy.deepcopy(reference)
        for c in candidate["0"]:
            c["cycle"] += 10
        self.assertTrue(
            compare_data_streams(reference, candidate, True, True)["data_equal"]
        )
        for field in ("kind", "row", "column"):
            changed = copy.deepcopy(candidate)
            changed["0"][3][field] = "pim_write" if field == "kind" else 99
            report = compare_data_streams(reference, changed, True, True)
            self.assertFalse(report["data_equal"])
            self.assertEqual(report["first_difference"]["index"], 3)
        for data in (
            candidate["0"][:-1],
            candidate["0"] + [candidate["0"][-1]],
            list(reversed(candidate["0"])),
        ):
            self.assertFalse(
                compare_data_streams(reference, {"0": data}, True, True)["data_equal"]
            )

    def test_matching_prefix_is_not_full_completion(self):
        reference = {"0": [command(i * 2, "gh_read", column=i) for i in range(8)]}
        report = compare_data_streams(
            reference, {"0": reference["0"][:6]}, False, False
        )
        self.assertTrue(report["data_equal"])
        self.assertFalse(report["full_completion"])
        self.assertIsNone(report["first_difference"])
        self.assertEqual(report["reference_count"], 8)
        self.assertEqual(report["candidate_count"], 6)

    def test_timing_and_issue_mutations_are_detected(self):
        commands = {
            "0": [
                command(0, "pim_activate"),
                command(14, "gh_read"),
                command(34, "precharge"),
                command(48, "pim_activate"),
                command(62, "gh_read_p"),
            ]
        }
        issues = {
            "0": [
                issue(0, "PIM_ACT", [[0, 0]]),
                issue(14, "GH_READ", [[0, 0]]),
                issue(34, "PIM_PRE", [[0, 0]]),
                issue(48, "PIM_ACT", [[0, 0]]),
                issue(62, "GH_READ_PRECHARGE", [[0, 0]]),
            ]
        }
        self.assertEqual(validate_timing(commands, issues, CONFIG), [])
        for idx, cycle in ((1, 13), (2, 20), (3, 47)):
            bad = copy.deepcopy(commands)
            bad["0"][idx]["cycle"] = cycle
            bi = copy.deepcopy(issues)
            bi["0"][idx]["cycle"] = cycle
            self.assertTrue(validate_timing(bad, bi, CONFIG))
        bad = copy.deepcopy(commands)
        bad["0"][1]["row"] = 9
        self.assertTrue(validate_timing(bad, issues, CONFIG))
        bad = copy.deepcopy(issues)
        bad["0"][1]["kind"] = "LH_READ"
        self.assertTrue(validate_timing(commands, bad, CONFIG))
        bad = copy.deepcopy(issues)
        bad["0"].append(bad["0"][1])
        self.assertTrue(validate_timing(commands, bad, CONFIG))

    def test_faw_counts_each_gang_bank_effect(self):
        commands = {"0": []}
        issues = {"0": []}
        for i in range(5):
            c = command(i, "pim_activate", bank=i % 4)
            c["bankgroup"] = i // 4
            commands["0"].append(c)
            issues["0"].append(issue(i, "PIM_ACT", [[i // 4, i % 4]]))
        errors = validate_timing(commands, issues, CONFIG)
        self.assertTrue(any("tFAW" in e for e in errors), errors)

    def test_auto_precharge_and_refresh_waits_are_checked(self):
        commands = {
            "0": [
                command(0, "pim_activate"),
                command(14, "gh_read_p"),
                command(48, "pim_activate"),
            ]
        }
        issues = {
            "0": [
                issue(0, "PIM_ACT", [[0, 0]]),
                issue(14, "GH_READ_PRECHARGE", [[0, 0]]),
                issue(48, "PIM_ACT", [[0, 0]]),
            ]
        }
        self.assertEqual(validate_timing(commands, issues, CONFIG), [])
        commands["0"][-1]["cycle"] = 47
        issues["0"][-1]["cycle"] = 47
        self.assertTrue(
            any(
                "ACT before bank ready" in e
                for e in validate_timing(commands, issues, CONFIG)
            )
        )
        commands["0"][-1] = command(48, "refresh")
        commands["0"].append(command(308, "pim_activate"))
        issues["0"][-1]["cycle"] = 308
        self.assertEqual(validate_timing(commands, issues, CONFIG), [])
        commands["0"][-1]["cycle"] = 307
        issues["0"][-1]["cycle"] = 307
        self.assertTrue(
            any(
                "ACT before bank ready" in e
                for e in validate_timing(commands, issues, CONFIG)
            )
        )


if __name__ == "__main__":
    unittest.main()
