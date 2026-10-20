"""Offline translation: literal wire vectors and real legacy workload parity."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hbnpu_instruction import (
    KernelConfig,
    Packet,
    encode_packet,
    decode_packet,
    load_workload,
    compile_kernel,
    write_trace,
)
import gen_LLM_trace as legacy

CONFIG = ROOT / "configs/HBM2_8Gb_x128_pim.ini"


def kernel(**kwargs):
    values = dict(
        m=64,
        n=128,
        k=256,
        m_tile=2048,
        input_row=2133,
        weight_row=0,
        output_row=2133,
        df=0,
        mcf=2,
        ucf=1,
        channels=8,
        banks_per_channel=16,
        columns_per_burst=32,
    )
    values.update(kwargs)
    return KernelConfig(**values)


class TranslationTest(unittest.TestCase):
    def test_default_program_separates_resident_load_and_streamed_gemm(self):
        packets = list(compile_kernel(kernel()))
        self.assertEqual([p.opcode for p in packets], [1, 2, 5, 6, 5, 6, 4])
        self.assertEqual([p.sequence for p in packets], list(range(7)))
        self.assertEqual(
            [p.payload for p in packets[2:6]], [(0, 0), (0, 0), (0, 128), (0, 128)]
        )
        self.assertEqual(encode_packet(packets[2]), (0xB715020000000002, 0, 0))
        self.assertEqual(encode_packet(packets[3]), (0xB716020000000003, 0, 0))

    def test_golden_words_and_store_first_profile(self):
        golden = json.loads(
            (ROOT / "tests/regression/fixtures/instruction_words.json").read_text()
        )
        for item in golden:
            words = tuple(int(w, 16) for w in item["words"])
            self.assertEqual(encode_packet(decode_packet(words)), words)
        packets = list(compile_kernel(kernel(), fused=True))
        self.assertEqual([p.opcode for p in packets], [1, 2, 3, 3, 4])
        self.assertEqual([p.sequence for p in packets], list(range(5)))
        self.assertEqual(
            encode_packet(packets[0]), tuple(int(w, 16) for w in golden[0]["words"])
        )
        self.assertEqual(packets[3].payload, (0, 128))
        self.assertEqual(encode_packet(Packet(4, 3, ())), (0xB714000000000003,))

    def test_partial_tiles_enumerate_m_n_k_without_simulation(self):
        packets = list(compile_kernel(kernel(m=65, n=129, k=130, m_tile=64)))
        self.assertEqual(
            [p.payload for p in packets if p.opcode == 6],
            [
                (0, 0),
                (0, 128),
                (128 << 32, 0),
                (128 << 32, 128),
                (64, 0),
                (64, 128),
                ((128 << 32) | 64, 0),
                ((128 << 32) | 64, 128),
            ],
        )
        self.assertEqual(
            [p.payload for p in packets if p.opcode == 2],
            [(0,), (1 << 32,), (64,), ((1 << 32) | 64,)],
        )

    def test_real_workload_transforms_match_legacy_generator(self):
        cases = [
            "prompt/0128/createQKV",
            "prompt/0128/QK",
            "decode/createQKV",
            "decode/WS/0032_L2",
            "decode/QKV/QK_0128",
            "decode/QKV/SV_0128",
        ]
        with tempfile.TemporaryDirectory() as temp:
            for name in cases:
                for row in (-1, 2133):
                    with self.subTest(name=name, row=row):
                        source = ROOT / "workloads/OPT-2.7B" / name
                        old = Path(temp) / "legacy"
                        legacy.gen_pim_trace(source, old, row)
                        words = [
                            int(line.split()[0], 16)
                            for line in old.read_text().splitlines()
                        ]
                        c = load_workload(source, CONFIG, row)
                        self.assertEqual(
                            (c.m, c.k, c.n),
                            tuple((w >> 3) & 0xFFFFFFFF for w in words[1:4]),
                        )
                        self.assertEqual(
                            (c.weight_row, c.output_row, c.input_row),
                            tuple(w >> 35 for w in words[1:4]),
                        )
                        self.assertEqual(c.columns_per_burst, 32)
                        self.assertEqual(next(compile_kernel(c)).opcode, 1)

    def test_lh_output_geometry_and_unsupported_group_count(self):
        packets = list(
            compile_kernel(kernel(df=1, mcf=2, ucf=8, m=256, n=128, m_tile=128))
        )
        self.assertEqual([p.payload for p in packets if p.opcode == 2], [(0,), (2,)])
        with self.assertRaises(ValueError):
            list(compile_kernel(kernel(df=1, mcf=2, ucf=8, m=256, n=256, m_tile=128)))

    def test_invalid_fields_and_wire_overflow(self):
        for changes in [
            dict(m=0),
            dict(n=-1),
            dict(k=2**31),
            dict(m_tile=3),
            dict(input_row=2**64),
            dict(mcf=3),
            dict(channels=0),
            dict(banks_per_channel=8),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                list(compile_kernel(kernel(**changes)))
        for packet in [Packet(4, 2**32, ()), Packet(9, 0, ()), Packet(3, 0, (0,))]:
            with self.assertRaises(ValueError):
                encode_packet(packet)
        for words in [
            (0xB721070000000000,),
            (0xB714000100000000,),
            (0xB714000000000000, 1),
            (0xB711070000000000,),
        ]:
            with self.assertRaises(ValueError):
                decode_packet(words)

    def test_trace_word_cycles_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "program"
            write_trace([Packet(2, 0, (0,))], out, 7, 3)
            self.assertEqual(
                out.read_text(), "0xb712010000000000\tPIM\t7\n0x0\tPIM\t10\n"
            )
            with self.assertRaises(FileExistsError):
                write_trace([], out)
            with self.assertRaises(ValueError):
                write_trace([], Path(temp) / "bad", word_period=0)


if __name__ == "__main__":
    unittest.main()
