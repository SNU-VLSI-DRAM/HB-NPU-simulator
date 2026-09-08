# HB-NPU refactoring, input migration, and channel-command updates

This document summarizes the work merged from
`refactor/behavior-preserving-hbnpu` into local `master`. The original reference
state remains available as the `baseline/pre-hbnpu-refactor-2026-09` tag. The
work began at
`502b5275` and was carried out in three distinct stages. The first stage
preserved execution behavior; the later stages deliberately changed the input
format and corrected timing. They should not be treated as one timing-neutral
refactor.

## Summary of the changes

| Stage | Main changes | Compatibility / timing |
|---|---|---|
| Behavior-preserving refactor | Split clock-tick phases, extract transaction decoding, group configuration/execution state, encapsulate controller queues, add regression baselines | Existing command addresses, order, and cycles retained at this checkpoint |
| Single-array cleanup | Remove spatial cut configuration and partition selection from code, workload headers, and transaction addresses | Breaking input-format change; remaining decoded payloads and transaction cycles preserved |
| Timing and channel operations | Correct GH read/write interleaving and GH array period; introduce explicit broadcasts and gangs; issue one logical PIM command per channel/cycle | Intentional command-cycle changes; tested per-bank data-operation streams preserved |

## 1. Structural refactoring

The large HB-NPU control path was separated into named stages for transaction
completion, refresh checks, PIM transaction processing, command scheduling,
dispatch, and channel-controller ticking. The matrix/tile geometry is still
scheduled in `dram_system.cc`.

The resulting ownership boundaries are:

| Component | Responsibility |
|---|---|
| `src/pim_transaction.{h,cc}` | Decode configuration, workload, and launch transactions |
| `src/pim_config.h` | Kernel/dataflow configuration, including an explicit initialized flag |
| `src/pim_execution_state.h` | One array's dimensions, iterators, stage counters, and pending-vector state |
| `src/pim_operation.h` | Logical PIM opcode, target banks, release cycle, and operation validation |
| `src/pim_command_batch.h` | Preserve operation boundaries and source classification during dispatch |
| `src/dram_system.cc` | Generate desired data operations and advance kernel progress |
| `src/controller.cc` | Queue and arbitrate operations, issue one eligible logical PIM command, collect bank-level activity |
| `src/channel_state.cc` | Check bank prerequisites and aggregate activation-window readiness |
| `src/timing.cc` | Bank timing relationships and PIM transfer cadence |

Configuration and execution state are separate: configuration describes the
kernel, while execution state changes as work proceeds. Tests also cover launch
before initialization and missing M/K/N dimensions.

## 2. Breaking change: remove spatial cuts from both formats

The old `vcut`/`hcut` fields, next-cut metadata, partition state, partition IDs,
and launch masks were removed rather than retained as unused placeholders.
Execution now explicitly targets one full array. The earlier partition-state
header is replaced by `pim_execution_state.h`.

Workload files contain exactly two records:

```text
tile_M, post_delay, mcf, ucf, df
M, K, N
```

Trace lines retain `hex_address PIM earliest_submission_cycle`, but the address
bit layout is compacted. The canonical launch address is now `0x1`. The
configuration transaction is followed by three dimension/base-row transactions
and launch. Transaction timestamps are earliest submission times, not a fixed
DRAM-command execution trace; queue backpressure and bank timing still apply.

The migration covers 12,840 workload headers, 27,984 generated trace files, five
regression input fixtures, and the root address-trace sample. This generated
data accounts for almost all of the large changed-file count. Remaining decoded
fields, base-row allocation rules, and transaction timestamps were preserved.
Generator parsing now uses `ast.literal_eval` and validates the two-record,
five-field-header format before writing output.

**External old traces must be regenerated.** There is no version tag or automatic
legacy-layout detection; old address bits have different meanings to the new
decoder. Removing two workload-header fields does not convert an existing trace.
`post_delay` remains in the workload format but is still not encoded or used by
the trace generator. See [the exact format and examples](../pim-trace-format.md).

## 3. GH/LH clocks and interleaving

For the supplied `tCK=1 ns`, `tCCD_S=1`, `tCCD_L=2` configuration:

- LH streaming uses bank-level I/O and a 2 ns array period (500 MHz).
- GH streaming uses global I/O and a 1 ns array period (1 GHz).
- With `df=0, mc=2` (`mcf=2, ucf=1`), two banks alternate at 1 ns intervals;
  each bank retains its 2 ns cadence. This applies to both reads and writes.

The first timing correction covered GH reads and their stage delays. The
subsequent audit found that output `PIM_WRITE` commands still bypassed the
global-I/O gate: both output banks could write in the same cycle. The current
implementation shares one global transfer slot across reads, writes, and their
auto-precharge variants, across all three PIM source queues.

`in_cnt` and `out_cnt` retain their geometry and activation-overlap subtraction,
using `tCCD_S` for GH and `tCCD_L` for LH. They now start from the corresponding
data-vector issue completion event, not from enqueueing an operation or issuing
a prerequisite. The equations and triggering details are documented in the
[timing guide](../gh-streaming-timing.md).

## 4. One logical issue, possibly many bank effects

The new controller contract distinguishes a channel command from its effects:

| Logical operation | Banks affected by one issue |
|---|---|
| LH_READ / LH_WRITE | All banks in the channel/rank |
| GH_READ / GH_WRITE | One bank |
| GANG_ACT / GANG_PRE | Exactly four banks simultaneously |
| Single-bank PIM ACT / PRE | One bank |

The controller issues at most one logical PIM command per channel per cycle,
including prerequisites. Channels remain independent. The normal-DRAM command
path and its HBM dual-command option are not replaced by this PIM policy.

Broadcast readiness is checked without mutation before any member changes state.
Each `GANG_ACT` consumes four bank activations in the rank's rolling `tFAW`
window, shared with ordinary and single-bank PIM activation. PIM deliberately
ignores `tRRD`, as agreed for this hardware model; normal-DRAM timing tables retain
their `tRRD` constraints. `GANG_PRE` checks every bank's recovery constraints and
starts each bank's `tRP`; it does not consume activation-window slots.

LH preparation can make partial progress through gangs and single-bank
remainders. GH activates only its addressed bank. Existing selected-bank output
stores remain GH writes: one bank for `df=1`, two interleaved banks for `df=0`.
They are not expanded into all-bank stores.

Pending data intents survive activation, precharge, and refresh. Iterators
advance only after the associated channel queues have issued their data, and
kernel termination waits for the final output transfer. Tests and review also
caught a self-refresh stall: pending PIM work now prevents its target rank from
sleeping or wakes an already-sleeping rank, respecting `tCKESR` and `tXS`.

## 5. Traces, testing, and measured effects

With command tracing enabled, each channel emits:

- `dramsim3ch_<channel>cmd.trace`: expanded bank effects, retaining the existing
  eight-field format and physical opcode spellings.
- `dramsim3ch_<channel>pim.trace`: logical issue cycle, channel, opcode, rank, and
  target set. A single gang record corresponds to four bank-effect records.

Physical writes retain the names `pim_write` / `pim_write_p`; the logical trace
identifies GH_WRITE versus LH_WRITE. Bank-level activity and energy accounting
still counts every affected bank rather than charging a broadcast as one bank.

The final validation set includes 36 C++ test cases (480 assertions), seven
real-trace timing tests, workload-format tests, regression-runner tests, five
quick fixtures, and a 46-kernel OPT-2.7B window. All six CTest groups pass.
The 51 workload runs retain identical per-bank data-operation addresses and
order relative to the pre-channel-command snapshot. This is command-level
equivalence, not a floating-point GEMM numerical-accuracy test.

The latest channel-command change makes all 36 LH OPT kernels faster and the
10 GH kernels 0.74–2.52% slower relative to the GH-read-corrected checkpoint.
The sum across those independent kernel runs changes from 162,956 to 163,935
cycles (+0.60%). It is not a separate full-model pipeline simulation.

A subsequent comparison directly against original `master` (`502b5275`) covers
OPT-2.7B, 6.7B, 13B, and 66B, two input-length/batch scenarios, both decode
dataflows, and early/middle/late attention contexts. All 159 completed kernel
pairs preserve per-bank data-operation streams (28,629,536 bank effects per
version), with no violations of the checked candidate timing invariants. The
largest long-prefill OPT-66B L2 run hits the 45-second candidate execution limit;
a separate 20,000-cycle comparison matches 115,232 common-prefix bank effects.
It does not establish that kernel's full completion. See the
[full comparison report and per-kernel results](2026-09-original-master-llm-comparison.md).
These measurements use the default `HBM2_8Gb_x128.ini`, not the earlier regression
configuration, and must not be mixed with the checkpoint numbers above.

Representative history, showing why the comparison checkpoint matters:

| Kernel | Before GH clock/read correction | GH clock/read corrected | Channel commands and GH writes corrected |
|---|---:|---:|---:|
| prompt/createQKV | 9,844 | 8,605 | 8,742 |
| prompt/L1 | 29,675 | 25,870 | 26,208 |
| prompt/L2 | 31,000 | 28,596 | 28,939 |
| decode/WS/createQKV | 6,691 | 5,487 | 5,625 |
| decode/createQKV | 1,556 | 1,556 | 1,511 |

Corrected scheduling is not expected to reproduce the old over-issued command
cycles. Keep the simulator revision, configuration, and input-format revision
together when comparing experiments with historical paper results.

Quick goldens compare physical and logical records exactly. The long suite
compares their separate hashes and statistics. Energy fields alone retain the
existing floating-point tolerance. Golden regeneration is explicit and requires
three stable runs; ordinary tests never rewrite expectations, and first-difference
reporting remains enabled. CMake propagates trace-related compile definitions
to library consumers to prevent public-class layout mismatches.

```bash
cmake -S . -B build -DCMD_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## Remaining scope and limitations

Arbitrary matrix shapes are not newly guaranteed. In particular,
`df=1, M=64, K=128, N=256` reaches an output-completion assertion in both the
saved pre-change binary and the updated implementation. That existing geometry
limitation remains documented, along with other
[legacy observations](known-legacy-issues.md). No new cross-bank read/write
bus-turnaround model is introduced here.
