# HB-NPU Tile Instruction Format

Instruction mode submits explicit tile descriptors to the existing executor.
It is selected by `[hbnpu] execution_mode = tile_instructions`; without that
setting, the simulator uses the legacy configuration-transaction format.
The two formats must not be mixed. The translator is `gen_tile_trace.py` and
requires only Python 3.7 or newer and its standard library.

## Words and packets

A word is one unsigned 64-bit value carried by a `0x... PIM cycle` trace record.
A packet is one instruction: a header word followed by its payload words.
The cycle is the earliest CPU submission opportunity, not the tile execution
time. Payload words are never interpreted as legacy configuration transactions.

| Header bits | Meaning |
| --- | --- |
| 63:56 | Magic `0xB7` |
| 55:52 | Version `1` |
| 51:48 | Opcode: CONFIG=1, STORE=2, TILE_BLOCK=3, END=4 |
| 47:40 | Number of payload words |
| 39:32 | Reserved, must be zero |
| 31:0 | Sequence number, starting at zero, without gaps or wrap |

CONFIG has seven payload words (eight transactions total):

1. M in low 32 bits, N in high 32 bits.
2. K in low 32 bits, M tile size in high 32 bits.
3. Input base row.
4. Resident-operand base row.
5. Output base row.
6. df in bits 7:0, mcf in 15:8, ucf in 23:16; remaining bits zero.
7. Channels in bits 31:0, banks/channel in 47:32, columns/burst in 63:48.

The topology is the simulator's normalized geometry. For the supplied HBM
configuration, columns are doubled for prefetch, giving 32 columns/burst.
Kernel coordinates use the same transformed M/N/K layout as legacy traces.
CONFIG is immutable and may occur only once.

STORE has one payload word: output M origin in low 32 bits and output N-tile
index in high 32 bits. TILE_BLOCK has two payload words: M/N origins packed
low/high 32 bits, followed by K origin in low 32 bits with high bits zero.
Extents and boundary flags are derived from those explicit coordinates.
END has no payload. For example, END at sequence 3 is `0xb714000000000003`.

## Supported program order

Programs contain one kernel and follow this profile:

```text
CONFIG
STORE output group 0
TILE_BLOCK group 0, K block 0
TILE_BLOCK group 0, remaining K blocks
STORE output group 1
TILE_BLOCK group 1, all K blocks
...
END
```

STORE dispatch reserves the output descriptor; actual writes wait for final-K
data readiness. Compute dispatch does not wait for that STORE to retire.
This order avoids placing a required store behind a compute instruction that
cannot progress until the store finishes.

Version 1 supports the existing HBM layout with four groups of four banks,
mc=mcf*ucf equal to 2 for df=0 or 16 for df=1, and matching compute/output group
counts. Tile/interleaving factors must be powers of two. Unsupported layouts,
topology mismatches, bad sequence/order, reserved bits, and unsafe integer or
row-address extents are rejected. This is a timing/command simulator, not a
numerical matrix evaluator.

## Queue and cycle behavior

The CPU can submit at most one word per tick and retains a rejected word.
The raw FIFO holds 32 words. Instruction storage defaults to 16 packet slots
and is configurable using `instruction_capacity`; every slot can hold the
largest packet, eight encoded words. Partial assembly reserves one slot and
finishes in that slot even when storage is otherwise full.

Ingress consumes at most one word per memory tick. A packet completed at tick t
is eligible for dispatch at t+1. At most one packet dispatches per tick, in order.
Dispatch frees the packet slot while the executor retains a bounded active
descriptor; it does not imply that computation or output has completed.

Tick order is refresh/completion checks, ingress, dispatch, executor progress,
then controllers. CPU submission follows the memory tick. Refresh can pause
bank work while upload and free descriptor slots continue to accept work.
END alone signals successful completion, after the backend drains. Missing END
or a partial packet at the cycle limit produces an incomplete result.

## Translator and diagnostics

`gen_tile_trace.py -w WORKLOAD -s CONFIG -t TRACE` compiles one existing
workload without running DRAMsim. Optional `--row-addr` matches the legacy base
row override; `--start-cycle` and `--word-period` control upload opportunities.
Existing destinations are refused unless `--overwrite` is supplied.

Instruction mode writes `hbnpu_instruction.json` alongside ordinary statistics.
It reports admitted words; decoded, dispatched and retired instructions; raw
and packet-slot high watermarks; CPU capacity stalls; compute/output resource
stalls; partial-packet and instruction-starvation cycles; backend drain and END
cycles; and incomplete state. A cycle timestamp is valid only when its matching
`backend_drained` or `end_completed` flag is true. Stall categories can overlap;
do not add them as an exclusive latency breakdown.

## Comparing executions

Use `tests/regression/compare_instruction_execution.py` with separate baseline
and candidate source trees/builds. For example:

```bash
python3 tests/regression/compare_instruction_execution.py \
  --baseline-binary /path/to/reference/build/dramsim3main \
  --candidate-binary build/dramsim3main --suite opt27b \
  --cycles 10000000 --output-dir /tmp/hbnpu-instruction-comparison-new
```

The tool compares per-bank data kind/row/column order and checks logical issue
widths, GH cadence, bank/row timing, and tFAW. Physical ACT/PRE scheduling and
refresh counts may differ, but their timing must remain legal. It reports raw
cycle deltas without shifting cycles to hide instruction delivery delays.
For capped runs it compares only common prefixes and reports both operation
counts; a matching prefix is not a completed kernel. The original
`compare_binaries.py` continues to require exact command cycles in legacy mode.
