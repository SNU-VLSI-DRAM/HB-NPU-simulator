import copy
import configparser
from pathlib import Path
import tempfile
import unittest
from compare_instruction_execution import (
    compare_data_streams,
    validate_timing,
    validate_rows,
)

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
    def test_row_predictability_checks_physical_issues_not_just_duration(self):
        commands, issues = {}, {}
        for ch in range(8):
            commands[str(ch)], issues[str(ch)] = [], []
            for beat in range(4):
                c = command(100 + beat, "gh_read", bank=1, row=9, column=30 + beat // 2)
                c["channel"] = ch
                c["bankgroup"] = (beat % 2) * 2
                commands[str(ch)].append(c)
                i = issue(100 + beat, "GH_READ", [[(beat % 2) * 2, 1]])
                i["channel"] = ch
                issues[str(ch)].append(i)
        rows = [
            dict(
                id=0,
                request_cycle=0,
                start_cycle=100,
                last_issue_cycle=103,
                vectors=2,
                operations_per_vector=2,
                vector_period=2,
                row=9,
                first_column=30,
            )
        ]
        self.assertEqual(validate_rows(commands, issues, rows, CONFIG), [])
        for field, value in (
            ("cycle", 104),
            ("row", 10),
            ("column", 29),
            ("kind", "precharge"),
        ):
            changed = copy.deepcopy(commands)
            changed["3"][1][field] = value
            self.assertTrue(validate_rows(changed, issues, rows, CONFIG))
        shifted = copy.deepcopy(issues)
        shifted["3"][1]["cycle"] += 1
        self.assertTrue(validate_rows(commands, shifted, rows, CONFIG))
        for field, value in (
            ("vectors", 3),
            ("vector_period", 3),
            ("last_issue_cycle", 104),
        ):
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            self.assertTrue(validate_rows(commands, issues, changed, CONFIG))

    def test_bank_refresh_cross_bank_constraints(self):
        for bg, bank in ((0, 1), (1, 0)):
            for first, second, earliest in (
                ("refresh_bank", "pim_activate", 260),
                ("pim_activate", "refresh_bank", 48),
                ("refresh_bank", "refresh_bank", 3900),
            ):
                for cycle in (earliest - 1, earliest):
                    with self.subTest(bg=bg, first=first, second=second, cycle=cycle):
                        a, b = command(0, first), command(cycle, second, bank=bank)
                        b["bankgroup"] = bg
                        logical = [
                            issue(c["cycle"], "PIM_ACT", [[c["bankgroup"], c["bank"]]])
                            for c in (a, b)
                            if c["kind"] == "pim_activate"
                        ]
                        errors = validate_timing({"0": [a, b]}, {"0": logical}, CONFIG)
                        self.assertEqual(bool(errors), cycle < earliest, errors)

    def test_bank_refresh_matches_existing_same_bank_application(self):
        # ChannelState does not apply the same_rank[REFRESH_BANK] tRFCb table.
        commands = {"0": [command(0, "refresh_bank"), command(1, "pim_activate")]}
        self.assertEqual(
            validate_timing(commands, {"0": [issue(1, "PIM_ACT", [[0, 0]])]}, CONFIG),
            [],
        )

    def test_omitted_timing_defaults_reject_early_and_accept_legal_commands(self):
        ini = configparser.ConfigParser()
        ini.read(CONFIG)
        for name in ("tCCD_L", "tRP", "tRAS"):
            ini.remove_option("timing", name)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "defaults.ini"
            with config.open("w") as stream:
                ini.write(stream)
            for second in (16, 20):
                commands = {
                    "0": [
                        command(0, "pim_activate"),
                        command(14, "gh_read"),
                        command(second, "gh_read"),
                    ]
                }
                logical = {
                    "0": [
                        issue(0, "PIM_ACT", [[0, 0]]),
                        issue(14, "GH_READ", [[0, 0]]),
                        issue(second, "GH_READ", [[0, 0]]),
                    ]
                }
                self.assertEqual(
                    bool(validate_timing(commands, logical, config)), second < 20
                )
            commands = {
                "0": [
                    command(0, "pim_activate"),
                    command(24, "precharge"),
                    command(34, "pim_activate"),
                ]
            }
            logical = {
                "0": [
                    issue(0, "PIM_ACT", [[0, 0]]),
                    issue(24, "PIM_PRE", [[0, 0]]),
                    issue(34, "PIM_ACT", [[0, 0]]),
                ]
            }
            self.assertEqual(validate_timing(commands, logical, config), [])

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
