"""Canonical HB-NPU geometry and original head-packed attention row coverage.

This is not a GQA lowering test: the original decode representation packs
sixteen independent heads. Keep it distinct from single-KV-group programs.
"""
import configparser
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import compile_kernel, load_workload, write_trace
from compare_instruction_execution import run_one, validate_rows, validate_timing

HARDWARE = ROOT / 'configs/HBNPU_8Gb.ini'


class CanonicalHardwareTest(unittest.TestCase):
    binary = None

    def test_geometry_and_capacity(self):
        ini = configparser.ConfigParser()
        self.assertEqual(ini.read(HARDWARE), [str(HARDWARE)])
        d, s = ini['dram_structure'], ini['system']
        self.assertEqual((int(d['device_width']), int(s['bus_width'])), (64, 64))
        self.assertEqual(int(d['BL']), 4)
        self.assertEqual(int(d['columns']), 128)
        self.assertEqual(int(d['columns']) * 2 // int(d['BL']), 64)
        self.assertEqual(int(s['bus_width']) // 8 * int(d['BL']), 32)
        self.assertEqual((int(s['channel_size']), int(s['channels'])), (256, 8))
        self.assertEqual(s['address_mapping'], 'rorabgbacoch')

    def test_original_head_packing_consumes_full_rows(self):
        for operation, mcf, ucf, expected_m, expected_n, expected_k in (
            ('QK', 1, 16, 2048, 16, 128),
            ('SV', 16, 1, 128, 1, 2048),
        ):
            with self.subTest(operation=operation), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                workload = base / 'workload'
                # gen_workload_decode's sixteen-head packing at d=128,L=2048.
                workload.write_text(f'2048, 8, {mcf}, {ucf}, 1\n2048, 2048, 1\n')
                context = load_workload(workload, HARDWARE)
                self.assertEqual((context.m, context.n, context.k),
                                 (expected_m, expected_n, expected_k))
                self.assertEqual(context.columns_per_burst, 64)
                hardware = configparser.ConfigParser()
                hardware.read(HARDWARE)
                hardware['hbnpu'] = dict(execution_mode='tile_instructions',
                                         row_lockstep='true', gh_read_delay='true')
                config = base / 'hardware.ini'
                with config.open('w') as stream:
                    hardware.write(stream)
                trace = base / 'instructions.trace'
                write_trace(compile_kernel(context), trace)
                commands, issues, complete, _, metrics, rows = run_one(
                    self.binary, config, trace, base / 'output', 100000, 60)
                self.assertTrue(complete)
                self.assertTrue(metrics['end_completed'])
                self.assertEqual(validate_timing(commands, issues, config), [])
                self.assertEqual(validate_rows(commands, issues, rows, config, complete), [])
                for channel in range(8):
                    for bankgroup in range(4):
                        for bank in range(4):
                            reads = [c for c in commands[str(channel)]
                                     if c['bankgroup'] == bankgroup and c['bank'] == bank
                                     and c['kind'] in ('lh_read', 'lh_read_p')]
                            self.assertEqual(len(reads), 2048)
                            self.assertEqual([c['column'] for c in reads], list(range(64)) * 32)
                            self.assertEqual([c['row'] for c in reads],
                                             [r for r in range(32) for _ in range(64)])
                            self.assertEqual([c['column'] for c in reads if c['kind'] == 'lh_read_p'],
                                             [63] * 32)


if __name__ == '__main__':
    CanonicalHardwareTest.binary = Path(sys.argv.pop(1)).resolve()
    unittest.main()
