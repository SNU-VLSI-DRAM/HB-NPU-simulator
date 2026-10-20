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
| 51:48 | Opcode: CONFIG=1, STORE=2, TILE_BLOCK=3, END=4, LOAD_INPUTS=5, TILE_GEMM=6 |
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

The topology is the simulator's normalized geometry. HBM column normalization
doubles the INI column count before division by BL: the canonical
`HBNPU_8Gb.ini` gives 64 addressable bursts per row, while the historical
`HBM2_8Gb_x128_pim.ini` fixture gives 32. The field named `columns/burst`
carries this addressable-burst count; it is not the number of bytes per command.
Kernel coordinates use the same transformed M/N/K layout as legacy traces.
CONFIG is immutable and may occur only once.

STORE has one payload word: output M origin in low 32 bits and output N-tile
index in high 32 bits. TILE_BLOCK, LOAD_INPUTS and TILE_GEMM each have two
payload words: M/N origins packed
low/high 32 bits, followed by K origin in low 32 bits with high bits zero.
Extents and boundary flags are derived from those explicit coordinates.
END has no payload. For example, END at sequence 3 is `0xb714000000000003`.

## Supported program order

Programs contain one kernel and follow this profile:

```text
CONFIG
STORE output group 0
LOAD_INPUTS group 0, K block 0
TILE_GEMM group 0, K block 0
LOAD_INPUTS / TILE_GEMM pairs for remaining K blocks
STORE output group 1
LOAD_INPUTS / TILE_GEMM pairs for all K blocks
...
END
```

LOAD_INPUTS reads the resident operand from CONFIG's resident base row, using
the existing GH weight-loading stage. TILE_GEMM consumes that resident tile
and streams the other operand from CONFIG's input base row through GH or LH.
Both packets must specify the same M/N/K origins. For example, load sequence 2
and GEMM sequence 3 for origins (0,0,0) are:

```text
0xb715020000000002 PIM 10
0x0 PIM 11
0x0 PIM 12
0xb716020000000003 PIM 13
0x0 PIM 14
0x0 PIM 15
```

GEMM dispatch may occur before the resident load finishes; execution waits for
all addressed channels to complete resident loading. With `gh_read_delay`
enabled (the instruction-mode default), this includes operand arrival after
the final GH read, not just command issue. See the
[GH arrival model](gh-streaming-timing.md#gh-operand-arrival).
Load retirement occurs at resident-tile readiness. Without a matching GEMM, the resident tile
remains ready but no streamed reads or computation start. A missing load,
duplicate load, mismatched tile, or END in place of GEMM is rejected before
resource waits can hide the error. A truncated upload remains incomplete.

There is one active resident tile, with no extra buffering, prefetch or reuse
across tiles. Stage-shift timing remains in effect. Row mode reserves the
channel during input streaming, so competing output traffic waits outside
that interval; total kernel latency can change.
For compatibility, a fused TILE_BLOCK can replace any complete LOAD_INPUTS /
TILE_GEMM pair; it cannot appear between the two. Its wire encoding and fused
semantics are unchanged; row mode applies to its streaming stage too.
Older instruction-mode binaries reject the new opcodes; use
`--fused` when generating traces for those binaries.

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

`gen_gqa_workload.py` emits this same version-1 instruction encoding. Its
version-2 JSON manifest is a separate metadata format, not a new ISA version.
For multi-group programs, the manifest's `members` describes each real
batch/KV group and its bank/column mapping. The existing LH store protocol is
fixed-footprint; it does not establish byte-accurate materialization of every
logical output. See the [GQA scope and limitations](../README.md#dense-qwen2mistral-matrix-workloads).

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

Tick order is completion checks and row coordination, refresh checks, ingress,
dispatch, executor progress, then controllers. CPU submission follows the
memory tick. Refresh can pause bank preparation while upload and free
descriptor slots continue to accept work.
END alone signals successful completion, after the backend drains. Missing END
or a partial packet at the cycle limit produces an incomplete result.

## Row boundary synchronization

`row_lockstep` defaults to true in tile-instruction mode and false in legacy
mode. The CPU packet format is unchanged: the HB-NPU executor decomposes each
tile's streamed operand into internal descriptors that stop at a physical
row boundary or the end of the tile. Resident loading remains separate.
One row descriptor's command metadata is buffered; no operand buffer is added.
Set `[hbnpu] row_lockstep = false` for the preceding per-vector scheduling path.
Row mode requires tile instructions and self-refresh disabled.

Before START, controllers drain existing PIM work, perform legal PRE/ACT
operations (including gang preparation for LH), and wait until every target
can read. The coordinator checks every channel and the next refresh deadline.
The reserved budget includes the final read offset, a conservative
`max(tRAS, AL+tRTP) + tRP + banks + 1` recovery/precharge margin. Refresh is not
postponed or disabled. Submission also requires a feasible post-refresh window:
the row budget plus conservative refresh recovery, residual tFAW, activation
waves, tRCDRD and one coordination cycle must fit the configured interval.
An interval failing this check is rejected rather than waiting forever. The
bound can reject aggressive configurations that permit only a favorable
initial launch. Otherwise, insufficient remaining time delays launch.

Once all channels are ready, START is common to all of them. Each controller
reserves its command path through the final beat, including any cadence gaps.
For the supported two-bank GH layout, banks 1 and 9 alternate at tCCD_S spacing;
each next vector begins after `max(burst_cycle, tCCD_L, 2*tCCD_S)` cycles.
LH broadcasts a vector to all banks every `max(burst_cycle, tCCD_L)` cycles.
With the supplied HBM configuration this gives successive GH beats one cycle
apart and LH broadcasts two cycles apart. Every actual issue is rechecked:
unexpected refresh, a missed beat, or channel skew raises a predictability
violation instead of rescheduling the stream. Tile accumulation and final-K
output readiness remain owned by the enclosing tile executor.

The contract is fixed controller-read issue timing relative to START, not
fixed latency from CPU submission or a model of physical PVT variation.
`hbnpu_rows.jsonl` records each completed row's request, START, last issue,
address range and cadence. `hbnpu_instruction.json` includes launched/completed
row counts and boundary-wait cycles. An unfinished final row in a capped run
is not reported as a completed predictable interval. The comparison tool
independently checks the row records against physical command and logical
issue traces, including exact beat cycles and all-channel alignment.

The `[pim-row-boundary]` C++ tests check admission at one cycle below, exactly
at, and one cycle above the refresh budget, with banks already ready. They
also check progress resets and command addresses across consecutive full rows
and a one-vector tail, for both GH and LH streaming.

## Translator and diagnostics

`gen_tile_trace.py -w WORKLOAD -s CONFIG -t TRACE` compiles one existing
workload without running DRAMsim. Optional `--row-addr` matches the legacy base
row override; `--start-cycle` and `--word-period` control upload opportunities.
Existing destinations are refused unless `--overwrite` is supplied. The default
is split load/GEMM packets; `--fused` emits compatible TILE_BLOCK packets.

Instruction mode writes `hbnpu_instruction.json` alongside ordinary statistics.
It reports admitted words; decoded, dispatched and retired instructions; raw
and packet-slot high watermarks; CPU capacity stalls; load/compute/output resource
stalls; partial-packet and instruction-starvation cycles; backend drain and END
cycles; and incomplete state. A cycle timestamp is valid only when its matching
`backend_drained` or `end_completed` flag is true. Stall categories can overlap;
do not add them as an exclusive latency breakdown. Load-resource stalls mean
the previous tile still owns the resident slot; a ready load waiting for an
unavailable GEMM packet contributes to instruction starvation.

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
