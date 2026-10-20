# GH and LH streaming timing

## Hardware interpretation

The canonical configuration is `configs/HBNPU_8Gb.ini`, with
`tCK=1 ns`, `tCCD_S=1`, and `tCCD_L=2`. The historical regression fixture
`configs/HBM2_8Gb_x128_pim.ini` shares those three timings but has different
bus width, row geometry, CL, and other timing parameters. Results from the two
profiles are not interchangeable.
Timing parameters in the simulator are cycle counts, not absolute nanoseconds.

| Streaming path | I/O resource | Array period | Array frequency |
|---|---|---|---|
| LH (`df=1`) | Bank-level local I/O | `tCCD_L * tCK = 2 ns` | 500 MHz |
| GH (`df=0`, weight-stationary) | Shared global I/O within a channel | `tCCD_S * tCK = 1 ns` | 1 GHz |

Prefill uses `mcf=2, ucf=1` for two-bank interleaving. Each bank can contribute
once per 2 ns, while their combined GH stream supplies one transfer every 1 ns:

```text
Time:   t    t+1 ns  t+2 ns  t+3 ns
Bank:   A      B       A       B
```

This is temporal bank interleaving, not the removed spatial array partitioning.
LH reads can operate across independent banks in parallel. Independent channels
also retain their own global-I/O slots. GH weight loading uses the same shared
GH slot even in a kernel whose later streaming phase uses LH; that does not
change the LH streaming array period.

## GH operand arrival

`[hbnpu] gh_read_delay` defaults to `true` in tile-instruction mode and `false`
in legacy mode. An explicit setting overrides either default. With it enabled,
GH_READ and GH_READ_PRECHARGE operands arrive at the actual data-command issue
cycle plus `read_delay = AL + CL + burst_cycle`. This is a whole-burst-ready
model for the global I/O path, not an additional interval between commands.
For example, GH commands issued at cycles 100 and 101 with a 16-cycle read
delay make their operands available at 116 and 117.

The controller records arrival deadlines separately from command queues.
Resident loads continue issuing at the existing cadence, then wait for the
last return across all channels before becoming ready. This applies to
resident loading in both dataflows because both use GH reads. For streaming,
issued vectors remain pipelined while their arrivals are consumed in order;
a two-bank GH vector is usable after both bank operands arrive. The existing
array-stage counters start from operand arrival and retain their ACT-overlap
subtractions. Row reservation and its trace still describe command issue,
not the later operand-arrival window. In-flight arrivals and array counters
continue during refresh in this mode; refresh still blocks new DRAM work.

LH column data uses the direct per-bank hybrid-bonding path, modeled with no
additional read-return latency. That is an abstraction, not zero physical
propagation time. Existing next-tick issue feedback still applies. GH writes
and normal CPU read/write timing are unchanged. This option does not add a
separate numerical GEMM, VPU, or physical store-completion model.

## Channel operations and gang timing

The scheduler submits explicit `PimOperation` data intents with their target
banks. The controller selects at most one eligible logical PIM operation per
channel per cycle, considering weight, input, and output queues in that priority
order. Bank updates belonging to one broadcast are effects of that command,
not additional command issues.

- GH_READ and GH_WRITE target one bank. All global transfers share one
  per-channel deadline, including their auto-precharge variants. Each bank's
  transfer cadence is also enforced, so two ready banks alternate instead of
  supplying simultaneous writes or repeated same-bank transfers at 1 ns.
- LH_READ and LH_WRITE target all banks in the channel/rank. Every bank must be
  ready before any data effect of the broadcast occurs.
- GANG_ACT activates exactly four banks simultaneously. It checks all target
  banks and reserves four bank activations in the rank's rolling `tFAW` window,
  shared with normal and single-bank PIM ACT. A recent single ACT can block a
  full gang; simultaneous expired entries cease consuming the window together.
- PIM activation intentionally does not enforce `tRRD`. Normal-DRAM activation
  timing is unchanged. With `tFAW=30`, four full gangs can prepare a closed
  16-bank channel at cycles 0, 30, 60, and 90, assuming no preceding activity or
  other constraints; the read also waits for the final bank's `tRCDRD`.
- GANG_PRE precharges four banks simultaneously, after every member satisfies
  its activation/read/write recovery constraints. It does not consume `tFAW`
  slots; each bank subsequently observes `tRP`.
- GH prepares only its target bank. LH preparation uses compatible groups of
  four in flattened-bank order, with single-bank prerequisites for remainders.
  Already-ready banks are not reactivated.

The existing output stores remain global-I/O transfers: one selected bank for
`df=1`, and two interleaved banks for `df=0, mc=2`. Their physical trace names
remain `pim_write`/`pim_write_p` for compatibility, while their logical names are
GH_WRITE/GH_WRITE_PRECHARGE. They are not converted into all-bank LH stores.

Queued data intents survive prerequisite issue and refresh. The scheduler
advances a vector's data iterator only after all its channel queues have issued
the data, not when a vector is enqueued or its banks are activated. It can submit
the next pair in the tick that observes completion, retaining continuous GH
interleaving. Kernel termination likewise waits for the final output transfer.
Refresh preparation gets an issue opportunity without silently discarding
queued work. With self-refresh enabled, pending PIM work prevents its target
rank from sleeping and wakes a rank that is already asleep. Other idle ranks
can still sleep; exit respects `tCKESR`, and subsequent activation respects
`tXS`. Instruction/transaction submission remains separate from this
command-level progress.

Existing per-bank read/write recovery rules remain. This correction does not
introduce a new cross-bank read/write bus-turnaround model. The scheduler no
longer depends on the earlier read-only readiness-lookahead mechanism.

For the fixed 128x128 array profile, each bank group has 16x8 PEs; eight
groups vertically and sixteen horizontally cover the array. The modeled
result pipeline uses one MAC cycle, four local reduction cycles (16 inputs),
and three cross-bank reduction cycles (eight inputs): **eight array cycles**.
Both reductions are trees; the vertical reduction is not eight serial
bank-to-bank accumulation steps. Horizontal input distribution is assumed
aligned and hidden by the result path, not added again to this latency.
This is a declared pipeline assumption, not RTL-derived timing.

The stage counters select the array period by **streaming dataflow**:

```text
array_period = (df == 0) ? tCCD_S : tCCD_L
out_cnt = max(1, array_period * (1 + 4 + 3) - tRCDWR)
in_cnt  = max(1, array_period * max(128 / mc, 16) - tRCDRD)
mc = mcf * ucf
```

`out_cnt` starts on the first vector of the final-K block; `in_cnt` starts
after its block's last vector. With GH read latency enabled, these refer to
operand consumption after arrival; otherwise they retain next-tick issue
feedback. `out_cnt` is also decremented during its initialization tick.
The activation-overlap subtraction and minimum one-tick countdown are retained.
With the supplied timing, eight array cycles mean 8 ns for GH and 16 ns for LH;
the preparation countdowns are respectively 1 and 2 simulator cycles before
the initialization-tick decrement. Expiration grants output eligibility, not
an independently checked per-result data-ready event. Output submission still
requires streaming to have finished. The separate `in_cnt` equation is unchanged.
This result-pipeline correction applies in both execution modes, independently
of the GH read-return-latency option.

## Trace formats

With `CMD_TRACE=ON`, each channel produces two traces:

- `dramsim3ch_<channel>cmd.trace`: the existing expanded per-bank effects,
  preserving the eight-field physical trace format and per-bank activity/energy.
- `dramsim3ch_<channel>pim.trace`: logical issues in the format
  `cycle channel opcode rank bankgroup:bank[,bankgroup:bank...]`.

For example, `30 0 GANG_ACT 0 0:0,0:1,0:2,0:3` is one issued operation with four
physical bank effects at cycle 30. Multiple physical entries in the same cycle
are therefore valid for gangs and LH broadcasts, but never for two GH transfers.
Quick regression goldens compare both streams exactly. The 46-kernel summary
records separate hashes for physical effects and logical issues.

## Channel-command correction: validation and performance

The baseline for this change is the preceding GH-read timing correction, not
the original simultaneous-read implementation. That baseline still issued two
output writes in the same channel/cycle (8430, 8430, 8432, 8432 in the prefill
fixture). The shared global-I/O gate corrects writes as well as reads.

All five quick fixtures and all 46 OPT kernels preserve their per-bank data
operation sequences and addresses. The audit finds no shared-GH-slot collisions,
multiple logical PIM issues per channel/cycle, or activation-window violations.

| Kernel | Before channel-command correction | Corrected |
|---|---:|---:|
| prompt/createQKV | 8,605 | 8,742 |
| prompt/QK | 541 | 545 |
| prompt/SV | 541 | 545 |
| prompt/Wo | 11,345 | 11,484 |
| prompt/L1 | 25,870 | 26,208 |
| prompt/L2 | 28,596 | 28,939 |
| decode/WS/createQKV | 5,487 | 5,625 |
| decode/WS/Wo | 5,673 | 5,766 |
| decode/WS/L1 | 16,811 | 17,204 |
| decode/WS/L2 | 17,024 | 17,335 |

The 10 GH kernels take 0.74–2.52% more cycles. All 36 LH kernels take fewer
cycles, by up to 2.89%. Summing the 46 independent kernel runs gives 162,956
before and 163,935 after (+0.60%); this sum is not a separate full-model pipeline
simulation. Gang preparation, single logical issue, and completion-based
progress intentionally change activation and phase timing. Matching the old
cycle trace is not a correctness requirement.

The then-default `HBM2_8Gb_x128.ini` configuration (now `HBNPU_8Gb.ini`)
was also checked with the prefill
and decode quick fixtures: 7,048 → 7,155 and 1,221 → 1,176 cycles respectively,
with unchanged per-bank data streams and no GH collisions.

Validation is not a claim that every arbitrary matrix shape is supported.
A synthetic `df=1, M=64, K=128, N=256` workload reaches the output-exhaustion
assertion in both the saved pre-change binary and this implementation. That
existing geometry/completion limitation is not repaired by the command-issue
change; the 51-case regression set does not exercise it.

## Historical GH-read-only correction

The original reference traces issued two GH reads in the same channel and cycle
(for example at cycles 19, 19, 21, 21). That behavior was faithfully preserved by
the earlier format-only refactor, but it did not implement bank interleaving.
This timing correction changes those records to 19, 20, 21, 22 and recalculates
the GH stage delays. Exact old timing equivalence is therefore not a goal here.

The earlier read-only correction measured these termination cycles:

| Kernel | Before correction | Corrected |
|---|---:|---:|
| prompt/createQKV | 9,844 | 8,605 |
| prompt/QK | 539 | 541 |
| prompt/SV | 539 | 541 |
| prompt/Wo | 11,290 | 11,345 |
| prompt/L1 | 29,675 | 25,870 |
| prompt/L2 | 31,000 | 28,596 |
| decode/WS/createQKV | 6,691 | 5,487 |
| decode/WS/Wo | 6,793 | 5,673 |
| decode/WS/L1 | 20,855 | 16,811 |
| decode/WS/L2 | 20,985 | 17,024 |

At that earlier checkpoint, all 36 other kernels retained their complete
original command streams and stats.
Seven GH kernels improve; three increase by less than 0.5%. Sustained GH
bandwidth is one read per ns, but removing simultaneous issue can change phase
and row-boundary timing, so identical latency is not promised for every shape.
These are simulator cycle measurements, not a new measured hardware result.

Checks cover:

- Per-channel GH exclusivity, alternating banks without steady-state bubbles,
  and shared weight/input slots, including read-with-precharge commands.
- Unchanged LH parallelism and two-cycle spacing.
- Dataflow-dependent stage counters and actual-issue rechecks after lookahead.
- No early activation from the lookahead path.
- Independent channel issue slots and spacing with a different `tCCD_S` value.
- Equal per-bank data-operation sequences for all five quick fixtures; equal
  GH/LH read and PIM-write counts for all 46 kernels.

The tests also exposed a build-layout mismatch: `CMD_TRACE` and `ADDR_TRACE`
control members in public C++ classes, so their definitions must be shared by
the library and its consumers. CMake now propagates both definitions publicly.

That checkpoint updated the WS/prefill goldens and left the three LH quick
goldens unchanged. The channel-command correction changes both GH and LH
schedules and adds logical trace records. Regression comparisons still use
exact command order, addresses, and cycles, with the existing tolerance only
for floating-point energy fields. Run:

```bash
cmake --build build -j4
ctest --test-dir build --output-on-failure
```
