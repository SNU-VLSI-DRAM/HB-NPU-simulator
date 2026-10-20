# Canonical hardware configuration revalidation

This records the original configuration audit. The subsequent
[head/batch-first GQA implementation and canonical trace/roofline results](2026-10-gqa-hybrid-roofline.md)
supersede the GQA limitations recorded below; the original-reference results
remain unchanged.

## Configuration and reference provenance

The canonical file is `configs/HBNPU_8Gb.ini`, renamed from
`configs/HBM2_8Gb_x128.ini`. All parsed settings are unchanged. Configuration
equivalence is checked using parsed settings, independently of line endings.

The immutable reference is original master
`502b52750778242c3a76845e40af32674e5baefd`, rebuilt from a Git archive with
command tracing enabled. This is not the master branch after refactoring was
merged. The audited candidate was the instruction implementation at
`402c2deb6fce362bf16d724da419d91f3d52649e` plus the then-uncommitted timing and
configuration changes. This is a recorded audit snapshot, not an assertion
that the commit alone reproduces the candidate.

| Parameter | Canonical configuration | Historical `_pim.ini` fixture |
|---|---:|---:|
| Device / bus width | 64 / 64 bits | 128 / 128 bits |
| INI columns | 128 | 64 |
| BL | 4 | 4 |
| Bytes per request | 32 | 64 |
| Addressable bursts per row | 64 | 32 |
| Channel capacity | 256 MiB | 1024 MiB |
| CL | 14 cycles | 4 cycles |
| tWTR_S / tWTR_L | 6 / 8 cycles | 0 / 0 cycles |
| Address mapping | rorabgbacoch | rorabgbachco |

These are simulator configuration values, not a claim that every modeled
parameter implements a complete physical HBM pseudo-channel interface.
The earlier roofline experiment using `_pim.ini` is not canonical HB-NPU
performance evidence. Historical fixture goldens have not been replaced with
candidate-generated results. The September original-master comparison report
did use the canonical file, but predates the current instruction pipeline.

## Comparison method

Original and current trace formats differ. Before each execution, the audit
independently decodes both launch formats and checks dimensions, dataflow,
packing factors, base rows, tile size, and submission timestamps. Current
instructions are compiled from the corresponding workload, not from observed
reference DRAM commands.

Both versions use identical canonical hardware parameters. The current path
enables tile instructions, GH read-delay accounting, and row lockstep. Timing
differences are recorded, not hidden or required to vanish. Complete kernels
must have identical ordered data opcodes and row/column addresses per bank.
For runs capped at 100,000 cycles, only the common per-bank prefix is compared;
unmatched suffixes and completion flags remain in the result. Candidate timing
and row-contract checks are independent additional checks.

The wide audit covers four OPT sizes (2.7B, 6.7B, 13B, 66B), two workload
scenarios, six prefill kernel families, decode linear kernels in both
dataflows, and QK/SV at three context positions per scenario. Results and
per-kernel cycle counts are retained in the audit directory below.

### Final results

- 160 kernel pairs attempted; all original/current launch semantics matched.
- 149 pairs completed in both versions with identical per-bank data streams.
- 11 pairs reached the 100,000-cycle cap in both versions with matching common
  per-bank prefixes. Their remaining suffixes and full completion are unverified.
- No candidate violation of the checked command-timing or row-contract rules.
- No simulator timeout, process error, or data-stream mismatch was observed.

The audit runner intentionally exits nonzero when any pair is incomplete;
its exit status was 1 because of those eleven capped pairs. This is not an
all-kernels-completed result. The separate full CTest run exited successfully.

[Per-kernel comparison and roofline estimates](2026-10-canonical-configuration-revalidation.csv)
record completion flags; capped kernels have no full-kernel roofline ratio.

## Corrected attention row-consumption result

The original decode generator packs sixteen independent heads/subarrays. This
must not be confused with the earlier GQA generator's single-group
output-column sharding. The current generator uses head/batch-first packing
with sharding only when fewer independent groups are available.

For head dimension 128 and context length 2048, the original packing yields:

| Operation | Streamed M | K | N | Full streamed rows per bank |
|---|---:|---:|---:|---:|
| QK | 2048 | 128 | 16 | 32 |
| SV | 128 | 2048 | 1 | 32 |

Fresh reference and candidate runs have matching data streams. Both consume
columns 0 through 63 for every streamed row. The new canonical integration
test checks this on all eight channels and all sixteen banks per channel,
including the position of read-with-precharge. It also checks completion,
command timing, and row contracts. The compact audit's displayed row runs
are channel-0/bank-0 examples; the integration test supplies the all-bank check.

The blanket claim that original QK/SV underutilize rows is therefore withdrawn.
Non-aligned dimensions can still produce valid tails: OPT-2.7B has head dimension
80, not 128, and a context such as 1151 is not a multiple of 64.

## Stream-only roofline check

Assuming two bytes per tensor element, each head-packed diagnostic streams
`16 * 128 * 2048 * 2 = 8,388,608` bytes. The actual LH command count gives the
same amount at 32 bytes per bank access. At eight channels, sixteen banks per
channel, and a two-cycle LH cadence, the ideal streaming rate is 2048 bytes per
cycle and the stream-only lower bound is 4096 cycles.
Including the sixteen heads' minimum query/score and output traffic gives
8,458,240 bytes, or a 4130-cycle ideal memory bound for either operation.
At the assumed 128x128 array compute ceiling, the compute bound is 512 cycles,
so this simple roofline remains memory-bound.

| Operation | Original cycles | Current cycles | Streaming slots | Row-boundary waiting | Remaining cycles |
|---|---:|---:|---:|---:|---:|
| QK | 8878 | 8883 | 4096 | 4647 | 140 |
| SV | 9467 | 9445 | 4096 | 4489 | 860 |

The waiting figures are the simulator's `row_boundary_wait_cycles`, including
bank preparation and refresh-related waiting; they are not pure tRCD costs.
The residual includes resident loads, output/drain, and sequencing overhead.
The 4096-cycle figure excludes those costs and is not a full kernel-latency
prediction. Full rows and a gap above the streaming lower bound coexist.

For example, current QK finishes row 0 at cycle 305 (`LH_READ_PRECHARGE`,
column 63), issues four `GANG_ACT` operations at 324, 354, 384, and 414,
then starts row 1 at cycle 428. The four activation groups are separated by
the configured 30-cycle tFAW window; the last group needs 14 cycles of
tRCDRD. Twenty-nine row contracts have a request-to-first-read delay of
122 cycles. Two longer delays coincide with refresh, observed at cycles
3921 and 7829. This is a concrete source of the gap despite full-row reads.

The per-kernel CSV additionally contains simple roofline estimates with
minimum input/output traffic and an assumed array compute ceiling. Decode
attention byte estimates account for sixteen separate heads, rather than
mistaking their packed dimensions for one dense matrix. These remain lower
bounds, not calibrated hardware accuracy estimates.

## Coverage and limits

After adding canonical coverage, CTest passed 14/14 targets. Most existing
targets retain the historical `_pim.ini` fixture; that green suite is not a
claim that every test has migrated to the canonical profile. The new
`hbnpu_canonical_hardware` target explicitly uses `HBNPU_8Gb.ini`.

At the time of this audit, the GQA allocator rejected the canonical 64-bit
profile and used single-group sharding. The subsequent GQA implementation
supports the canonical profile and head/batch-first packing; its independent
validation is documented in the [GQA report](2026-10-gqa-hybrid-roofline.md).
The OPT audit alone does not validate GQA. No numerical model inference is
executed by these tests.

## Reproduction artifacts

Local audit directory: `/tmp/hbnpu-correct-reference-54ZBIq`.

- `audit.py`, `attention.py`, `summarize.py`: comparison and analysis scripts.
- `manifest.json`: source revision and executable/library/configuration hashes.
- `reference-source/`, `reference-build/`: freshly archived original source/build.
- `candidate-snapshot/`: candidate sources, including untracked implementation
  files, executable, and shared library. `candidate.patch` alone is not a full
  snapshot.
- `report.json`, `comparison.csv`, `summary.json`: per-case results and estimates.
- `case-*/`, `head128-QK/`, `head128-SV/`: raw traces and simulator output.
- `final-suite.log`: the 14-target CTest run.

The temporary audit directory is local evidence, not a permanent repository
fixture. The canonical regression can be rerun without it:

```bash
cmake -S . -B build -DCMD_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
ctest --test-dir build -R hbnpu_canonical_hardware --output-on-failure
```
