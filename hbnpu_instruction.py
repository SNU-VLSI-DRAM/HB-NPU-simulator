"""Version-1 HB-NPU packets and offline tile enumeration (no simulator dependency)."""
import ast
import configparser
from dataclasses import dataclass
from pathlib import Path

LENGTHS = {1: 7, 2: 1, 3: 2, 4: 0}
INT_MAX = (1 << 31) - 1


@dataclass(frozen=True)
class Packet:
    opcode: int
    sequence: int
    payload: tuple


@dataclass(frozen=True)
class KernelConfig:
    m: int
    n: int
    k: int
    m_tile: int
    input_row: int
    weight_row: int
    output_row: int
    df: int
    mcf: int
    ucf: int
    channels: int
    banks_per_channel: int
    columns_per_burst: int


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("{} outside [{}, {}]".format(name, low, high))
    return value


def _power(value):
    return value > 0 and value & (value - 1) == 0


def encode_packet(packet):
    if packet.opcode not in LENGTHS or len(packet.payload) != LENGTHS[packet.opcode]:
        raise ValueError("opcode/payload length")
    _integer(packet.sequence, 0, (1 << 32) - 1, "sequence")
    for word in packet.payload:
        _integer(word, 0, (1 << 64) - 1, "payload word")
    header = (
        0xB710000000000000
        | packet.opcode << 48
        | len(packet.payload) << 40
        | packet.sequence
    )
    return (header,) + tuple(packet.payload)


def decode_packet(words):
    if not words:
        raise ValueError("missing header")
    for word in words:
        _integer(word, 0, (1 << 64) - 1, "word")
    header = words[0]
    if header >> 56 != 0xB7 or (header >> 52) & 15 != 1 or (header >> 32) & 255:
        raise ValueError("magic/version/reserved header")
    opcode = (header >> 48) & 15
    if (
        opcode not in LENGTHS
        or (header >> 40) & 255 != LENGTHS[opcode]
        or len(words) != LENGTHS[opcode] + 1
    ):
        raise ValueError("opcode/payload length")
    return Packet(opcode, header & 0xFFFFFFFF, tuple(words[1:]))


def _geometry(c):
    for name in ("m", "n", "k", "m_tile"):
        _integer(getattr(c, name), 1, INT_MAX, name)
    _integer(c.channels, 1, INT_MAX // 16, "channels")
    _integer(c.columns_per_burst, 1, 65535, "columns per burst")
    for name in ("input_row", "weight_row", "output_row"):
        _integer(getattr(c, name), 0, (1 << 64) - 1, name)
    for name in ("mcf", "ucf"):
        _integer(getattr(c, name), 1, 255, name)
    if not all(_power(v) for v in (c.m_tile, c.mcf, c.ucf)):
        raise ValueError("tile and interleaving factors must be powers of two")
    if c.banks_per_channel != 16 or not (
        (c.df == 0 and c.mcf * c.ucf == 2) or (c.df == 1 and c.mcf * c.ucf == 16)
    ):
        raise ValueError("unsupported dataflow/interleaving geometry")
    kt = min(c.channels * 16, c.k)
    nm = (c.m - 1) // c.m_tile + 1
    nn = (c.n - 1) // 128 + 1
    nk = (c.k - 1) // kt + 1
    _integer(c.m * c.mcf, 1, INT_MAX, "output M product")
    mo = max(1, c.m * c.mcf // 128) if c.df else c.m
    ot = c.m_tile // 128 * c.mcf if c.df else c.m_tile
    _integer(ot, 1, INT_MAX, "output tile")
    no = 1 if c.df else nn
    if nm * nn != ((mo - 1) // ot + 1) * no:
        raise ValueError("output/compute group count")
    spans = (
        nm * c.m_tile * nk,
        nn * min(c.n, 128 if c.df else 64) * nk,
        ((mo - 1) // ot + 1) * ot * nn,
    )
    for value in (nm * c.m_tile, nk * kt, nn * 128) + spans:
        _integer(value, 1, INT_MAX, "address offset product")
    return kt, mo, ot, no, spans


def load_workload(workload, config, row_addr=-1):
    lines = Path(workload).read_text().splitlines()
    if len(lines) < 2 or any(line.strip() for line in lines[2:]):
        raise ValueError("expected one header and exactly one (M,K,N) workload")
    header = ast.literal_eval(lines[0])
    dims = ast.literal_eval(lines[1])
    if not isinstance(header, (tuple, list)) or len(header) != 5:
        raise ValueError("expected five-field header")
    if not isinstance(dims, (tuple, list)) or len(dims) != 3:
        raise ValueError("expected (M,K,N) tuple")
    tile, post_delay, mcf, ucf, df = header
    m, k, n = dims
    for name, value in zip(
        ("M", "K", "N", "tile", "mcf", "ucf"), (m, k, n, tile, mcf, ucf)
    ):
        _integer(value, 1, INT_MAX, name)
    _integer(df, 0, 1, "dataflow")
    _integer(row_addr, -1, (1 << 64) - 1, "row override")
    placement = (
        (k * n // 1024 + 1 if df == 0 else k * m // 1024 + 1)
        if row_addr == -1
        else row_addr
    )
    ini = configparser.ConfigParser()
    if not ini.read(config):
        raise ValueError("cannot read hardware configuration")
    dram = ini["dram_structure"]
    protocol = dram["protocol"]
    if protocol not in ("HBM", "HBM2"):
        raise ValueError("v1 tile translator supports HBM geometry")
    bl = int(dram["BL"]) or 4
    columns = int(dram["columns"]) * 2
    if (
        columns % bl
        or int(dram["banks_per_group"]) != 4
        or int(dram["bankgroups"]) != 4
    ):
        raise ValueError("unsupported bank/column geometry")
    c = KernelConfig(
        m // 2 if df == 0 else max(1, m // mcf),
        n if df == 0 else n * ucf,
        k if df == 0 else max(1, k // ucf),
        tile,
        placement if df == 0 else 0,
        0 if df == 0 else placement,
        placement,
        df,
        mcf,
        ucf,
        int(ini["system"]["channels"]),
        16,
        columns // bl,
    )
    *_, spans = _geometry(c)
    rows = int(dram["rows"])
    for base, span in zip((c.input_row, c.weight_row, c.output_row), spans):
        if base >= rows or (span - 1) // c.columns_per_burst >= rows - base:
            raise ValueError("row extent exceeds hardware")
    return c


def compile_kernel(c):
    kt, mo, ot, no, _ = _geometry(c)
    config = (
        c.m | c.n << 32,
        c.k | c.m_tile << 32,
        c.input_row,
        c.weight_row,
        c.output_row,
        c.df | c.mcf << 8 | c.ucf << 16,
        c.channels | c.banks_per_channel << 32 | c.columns_per_burst << 48,
    )
    sequence = 0
    yield Packet(1, sequence, config)
    group = 0
    for m in range(0, c.m, c.m_tile):
        for n in range(0, c.n, 128):
            sequence += 1
            _integer(sequence, 0, (1 << 32) - 1, "sequence")
            yield Packet(2, sequence, ((group // no) * ot | (group % no) << 32,))
            for k in range(0, c.k, kt):
                sequence += 1
                _integer(sequence, 0, (1 << 32) - 1, "sequence")
                yield Packet(3, sequence, (m | n << 32, k))
            group += 1
    sequence += 1
    _integer(sequence, 0, (1 << 32) - 1, "sequence")
    yield Packet(4, sequence, ())


def write_trace(packets, output, start_cycle=0, word_period=1, *, overwrite=False):
    _integer(start_cycle, 0, (1 << 64) - 1, "start cycle")
    _integer(word_period, 1, (1 << 64) - 1, "word period")
    with Path(output).open("w" if overwrite else "x") as out:
        cycle = start_cycle
        for packet in packets:
            for word in encode_packet(packet):
                _integer(cycle, 0, (1 << 64) - 1, "submission cycle")
                out.write("{}\tPIM\t{}\n".format(hex(word), cycle))
                cycle += word_period
