"""Check global-I/O issue bandwidth using real simulator command traces."""
import collections
from pathlib import Path
import sys
import unittest

import run_regression as regression


class GlobalReadTimingTest(unittest.TestCase):
    binary = None

    @classmethod
    def setUpClass(cls):
        cls.result = regression.run_case(
            cls.binary, regression.QUICK_CASES["ws_gemm_prompt"])
        cls.local_result = regression.run_case(
            cls.binary, regression.QUICK_CASES["decode_gemv"])
        cls.refresh_result = regression.run_case(
            cls.binary, regression.QUICK_CASES["refresh_boundary"])

    def test_no_channel_issues_two_global_reads_in_one_cycle(self):
        for channel, commands in self.result["commands"].items():
            with self.subTest(channel=channel):
                cycles = collections.Counter(
                    c["cycle"] for c in commands
                    if c["kind"] in ("gh_read", "gh_read_p"))
                self.assertTrue(cycles)
                self.assertEqual(max(cycles.values()), 1)

    def test_weight_and_input_banks_sustain_one_global_read_per_cycle(self):
        for channel, commands in self.result["commands"].items():
            for bank in (0, 1):  # Weight loading and input streaming.
                with self.subTest(channel=channel, bank=bank):
                    reads = [c for c in commands
                             if c["kind"] == "gh_read" and c["bank"] == bank][:8]
                    self.assertEqual(len(reads), 8)
                    self.assertEqual([c["cycle"] - reads[0]["cycle"] for c in reads],
                                     list(range(8)))
                    self.assertEqual([c["bankgroup"] for c in reads],
                                     [0, 2, 0, 2, 0, 2, 0, 2])
                    self.assertEqual([c["column"] for c in reads],
                                     [0, 0, 1, 1, 2, 2, 3, 3])

    def test_output_writes_interleave_instead_of_issuing_bank_pairs_together(self):
        # Catches the former read-only channel gate: two ready output banks
        # must consume successive slots, including write-with-precharge.
        for channel, commands in self.result["commands"].items():
            with self.subTest(channel=channel):
                writes = [c for c in commands if c["kind"] in
                          ("pim_write", "pim_write_p", "gh_write", "gh_write_p")]
                self.assertGreaterEqual(len(writes), 8)
                self.assertEqual(max(collections.Counter(
                    c["cycle"] for c in writes).values()), 1)
                self.assertEqual([c["cycle"] - writes[0]["cycle"]
                                  for c in writes[:8]], list(range(8)))
                self.assertEqual([c["bankgroup"] for c in writes[:8]],
                                 [0, 2, 0, 2, 0, 2, 0, 2])
                self.assertEqual([c["bank"] for c in writes[:8]], [3] * 8)
                self.assertEqual([c["column"] for c in writes[:8]],
                                 [0, 0, 1, 1, 2, 2, 3, 3])

    def test_global_reads_and_writes_share_one_channel_slot(self):
        for channel, commands in self.result["commands"].items():
            with self.subTest(channel=channel):
                transfers = [c for c in commands if c["kind"] in
                             ("gh_read", "gh_read_p", "pim_write", "pim_write_p",
                              "gh_write", "gh_write_p")]
                self.assertTrue(transfers)
                self.assertEqual(max(collections.Counter(
                    c["cycle"] for c in transfers).values()), 1)

    def test_one_logical_issue_has_exactly_its_declared_bank_effects(self):
        physical_kind = {
            "GH_READ": "gh_read", "GH_READ_PRECHARGE": "gh_read_p",
            "GH_WRITE": "pim_write", "GH_WRITE_PRECHARGE": "pim_write_p",
            "LH_READ": "lh_read", "LH_READ_PRECHARGE": "lh_read_p",
            "LH_WRITE": "pim_write", "LH_WRITE_PRECHARGE": "pim_write_p",
            "GANG_ACT": "pim_activate", "GANG_PRE": "precharge",
            "PIM_ACT": "pim_activate", "PIM_PRE": "precharge",
        }
        for name, result in (("global", self.result), ("local", self.local_result),
                             ("refresh", self.refresh_result)):
            with self.subTest(case=name):
                self.assertTrue("pim_issues" in result, "missing logical PIM trace")
                for channel, issues in result["pim_issues"].items():
                    self.assertTrue(issues)
                    self.assertEqual(max(collections.Counter(
                        issue["cycle"] for issue in issues).values()), 1)
                    effects = collections.defaultdict(list)
                    for command in result["commands"][channel]:
                        effects[command["cycle"]].append(command)
                    for issue in issues:
                        self.assertEqual(issue["channel"], int(channel))
                        self.assertIn(issue["kind"], physical_kind)
                        targets = [tuple(bank) for bank in issue["targets"]]
                        self.assertEqual(len(targets), len(set(targets)))
                        expected_width = (16 if issue["kind"].startswith("LH_")
                                          else 4 if issue["kind"].startswith("GANG_")
                                          else 1)
                        self.assertEqual(len(targets), expected_width)
                        commands = effects[issue["cycle"]]
                        self.assertEqual(len(commands), expected_width)
                        self.assertEqual({(c["bankgroup"], c["bank"])
                                          for c in commands}, set(targets))
                        self.assertEqual({c["kind"] for c in commands},
                                         {physical_kind[issue["kind"]]})
                        self.assertEqual({c["rank"] for c in commands},
                                         {issue["rank"]})
                    issue_cycles = {issue["cycle"] for issue in issues}
                    for command in result["commands"][channel]:
                        if command["kind"] in ("pim_activate", "pim_write", "pim_write_p",
                                               "gh_read", "gh_read_p", "lh_read", "lh_read_p"):
                            self.assertIn(command["cycle"], issue_cycles)

    def test_local_read_prepares_all_banks_with_four_bank_gangs(self):
        self.assertTrue("pim_issues" in self.local_result, "missing logical PIM trace")
        for channel, issues in self.local_result["pim_issues"].items():
            with self.subTest(channel=channel):
                first_local = next(i for i, issue in enumerate(issues)
                                   if issue["kind"].startswith("LH_READ"))
                gangs = [issue for issue in issues[:first_local]
                         if issue["kind"] == "GANG_ACT"]
                self.assertEqual(len(gangs), 4)
                self.assertEqual({tuple(bank) for gang in gangs
                                  for bank in gang["targets"]},
                                 {(group, bank) for group in range(4)
                                  for bank in range(4)})

    def test_activation_window_counts_bank_effects_not_gang_commands(self):
        # The regression configuration has tFAW=30 cycles. A gang contributes
        # four activations, not one; normal ACT and PIM ACT share the window.
        for result in (self.result, self.local_result, self.refresh_result):
            for channel, commands in result["commands"].items():
                windows = collections.defaultdict(collections.deque)
                for command in commands:
                    if command["kind"] not in ("activate", "pim_activate"):
                        continue
                    live = windows[command["rank"]]
                    cycle = command["cycle"]
                    while live and live[0] + 30 <= cycle:
                        live.popleft()
                    live.append(cycle)
                    self.assertLessEqual(len(live), 4,
                        "tFAW exceeded in channel {} at cycle {}".format(channel, cycle))


if __name__ == "__main__":
    GlobalReadTimingTest.binary = Path(sys.argv.pop(1)).resolve()
    unittest.main()
