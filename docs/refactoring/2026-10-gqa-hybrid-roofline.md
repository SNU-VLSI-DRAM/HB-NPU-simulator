# Head/batch-first GQA: trace and roofline analysis

Hardware: `configs/HBNPU_8Gb.ini`; 8 channels, 16 banks/channel,
64-bit bus, BL=4, 32 bytes/request, 64 requests/row, tCK=1 ns,
tCCD_L=2 cycles. These are the simulator's configured units, not a claim
about a different HBM organization. LH is used for K/V streaming.

## Implemented mapping

`hbnpu_gqa.lower_workloads` packs compatible independent KV groups/batch items
before splitting a group's streamed output-column dimension across banks.
For a pack of G groups:

- `ucf=G`, `mcf=16/G`, with G in {1,2,4,8,16}.
- Physical `M=padded_output_columns/(16/G)`, `K=padded_reduction_dimension`,
  `N=active_queries_per_group*G`.
- Each group owns a disjoint contiguous stripe of banks. Within that stripe,
  each bank owns a disjoint output-column range.
- Up to eight queries per group are processed together. Larger query groups
  use additional chunks, sharing the cache region but charging another stream.
- Six independent groups use packs of four and two. No duplicate/fake groups
  fill the spare banks; the smaller pack splits each group's columns further.
- Layer, operation, query position, shape, and causal prefix must match.
  Batch identities may differ. Projections retain single-record lowering.

At head dimension 128 and context 2048:

| Model / batch | Groups per pack | Banks/group | QK physical M,N,K | SV physical M,N,K |
|---|---:|---:|---|---|
| Qwen2.5-7B / 1 | 4 | 4 | 512,28,128 | 32,28,2048 |
| Qwen2.5-7B / 4 | 16 | 1 | 2048,112,128 | 128,112,2048 |
| Mistral-7B-v0.3 / 1 | 8 | 2 | 1024,32,128 | 64,32,2048 |
| Mistral-7B-v0.3 / 2 | 16 | 1 | 2048,64,128 | 128,64,2048 |

Only the fully head/batch-parallel case retains QK M=2048 and SV M=128.
Fallback sharding deliberately reduces M; this is not lost logical work.
`manifest.json` version 2 records every member's identity, bank stripe,
column range, and resident-vector range. Singular tensor/head fields are null
for packs; use `members`, not member 0 as a proxy for the program.

## What the traces establish

The experiment executes 16 packed programs: the eight aligned decode kernels
above, Qwen batch 1 QK/SV at context 143, and Qwen batch 2 QK/SV for each of
three prefill causal prefixes. Every program must complete within 100,000
cycles; incomplete runs fail rather than receiving a performance ratio.

Checks cover timing/row contracts and every channel/bank's stream count,
absolute row progression, and column progression. Issued LH bytes equal the
independently calculated padded K/V bytes. The comparison also runs the old
single-group lowering on the **same current binary and canonical hardware**;
it is not a comparison against original simulator timing or old goldens.

At context 2048, Qwen batch 1 streams 2 MiB, Mistral batch 1 streams 4 MiB,
and each sixteen-group case streams 8 MiB, for either QK or SV. Packing does
not change these total LH bytes relative to running the groups separately.

All aligned QK segments consume 64 columns. Qwen batch 1 SV consumes 32-column
segments because of its four-bank/group split. Mistral batch 1 and both
sixteen-group SV cases consume full 64-column segments. Bank parallelism does
not guarantee full rows when the fallback splits a small output dimension.

## Roofline and measured cycles

For G groups, q active queries/group, reduction K, and output columns O:

```
logical FLOPs         = 2 * G * q * K * O
minimum logical bytes = 2 * G * (K*O + q*K + q*O)
LH peak bandwidth     = 8 * 16 * 32 / 2 = 2048 bytes/cycle
assumed compute peak = 2 * 128 * 128 / 2 = 16384 FLOPs/cycle
logical roofline      = max(minimum bytes / bandwidth, FLOPs / compute peak)
issued-stream bound   = actual LH command bytes / bandwidth
```

The logical roofline is deliberately optimistic: all operands use the LH peak
in that estimate, though resident operands actually use GH. It excludes row
preparation, refresh, padding, repeated resident loads, and submission/drain.
It is not a fitted latency predictor or an error bar against measured silicon.

| Model / batch | Op | Logical roof cycles | Packed cycles | Separate-group cycles |
|---|---|---:|---:|---:|
| Qwen / 1 | QK | 1,083.5 | 2,537 | 2,408 |
| Qwen / 1 | SV | 1,083.5 | 5,321 | 12,992 |
| Qwen / 4 | QK | 4,334 | 15,000 | 9,632 |
| Qwen / 4 | SV | 4,334 | 14,101 | 51,968 |
| Mistral / 1 | QK | 2,116 | 5,431 | 4,768 |
| Mistral / 1 | SV | 2,116 | 5,798 | 22,640 |
| Mistral / 2 | QK | 4,232 | 12,496 | 9,536 |
| Mistral / 2 | SV | 4,232 | 11,336 | 45,280 |

These are independent, cold-start matrix programs. Summing separate runs
includes their separate setup/drain overhead, not steady-state fused execution.

### Where the gap comes from

The issued-stream lower bound is 1,024 / 2,048 / 4,096 cycles for 2 / 4 / 8 MiB.
For Qwen batch 1:

| Op | Stream slots | Reported row-boundary waits | Remaining cycles | Total |
|---|---:|---:|---:|---:|
| QK | 1,024 | 896 | 617 | 2,537 |
| SV | 1,024 | 2,011 | 2,286 | 5,321 |

The remainder is an accounting residual, not an independently measured stage:
it includes GH resident loads/arrival, compute-tile turnover, output stores,
and instruction/drain overhead. Row waits include preparation and refresh;
they must not all be called refresh time. The gang-ACT/tFAW cost remains a
major row-preparation constraint.

QK is slower than the separate-group baseline for these aligned contexts.
The current lowering retains M tiles of 128. Packing makes physical M larger:
Qwen batch 1 has four M tiles and reloads 28 resident vectors per tile,
112 loads/channel versus 28 total for four separate one-tile programs.
Qwen batch 4 has sixteen M tiles and 1,792 loads/channel versus 112 total
for sixteen separate programs. The larger contiguous stream does not remove
this repeated-load cost. No dataflow search or resident-reuse optimization was
introduced by this change.

SV improves substantially: packing reduces the number of short row contracts
and separate program launches without increasing total K/V stream bytes.
For Qwen batch 1, sixteen 32-column contracts replace sixty-four 8-column
contracts across separate groups. Total GH resident loads are unchanged.

At context 143, padding increases K/V traffic from 146,432 to 262,144 bytes
for Qwen batch 1. Its ideal logical roof is 78.91 cycles, while the actual
stream alone needs 128 cycles; QK/SV take 367/572 cycles. Tiny prefill prefixes
are even more dominated by fixed padding and setup, not by useful matrix size.

### Important output-model limitation

The existing LH output-store protocol remains fixed-footprint. The packed
Qwen context-2048 traces issue 16 QK stores/channel and one SV store/channel,
irrespective of the number of packed groups. These are **not byte-accurate
materialization of all logical outputs**. Packing reduces this abstract output
overhead versus separate programs, so the whole-program ratios above are not
isolated streaming speedups. The logical roofline includes logical outputs;
the simulator does not establish that complete output traffic was emitted.

No tensor values, softmax, RoPE, normalization, activations, cache repacking,
or end-to-end inference were executed. Trace correctness here means the
declared timing/address representation executed as intended, not numerical
GQA correctness or a calibrated full-model performance prediction.

## When splitting a single group helps

Fewer than sixteen independent groups leaves bank parallelism unused if each
group stays on one bank. Splitting is an occupancy optimization, not a
mathematical requirement for correct attention. Qwen TP=4/batch=1 has only one
local KV group (seven queries); Qwen TP=1/batch=1 has four; Mistral batch=1 has
eight. These exercise 16-, 4-, and 2-way column splitting, respectively.

The controlled **no-split baseline** uses the same real groups, but maps each
to one bank and zero-pads the remaining slots of a sixteen-group program.
Resident-query slots for those dummy groups are padded too. LH currently has
no active-bank mask. This is a concrete existing-ISA baseline, **not optimal
unsplit scheduling or hypothetical masked-bank hardware**. Both variants use
the same canonical hardware, 128-wide M tiles, and context 2048.

| Local groups | Case | Banks/group with splitting | QK cycles, no-split → split | QK speedup | SV cycles, no-split → split | SV speedup |
|---:|---|---:|---|---:|---|---:|
| 1 | Qwen TP=4, batch=1 | 16 | 15,000 → 602 | 24.92x | 14,101 → 3,248 | 4.34x |
| 4 | Qwen TP=1, batch=1 | 4 | 15,000 → 2,537 | 5.91x | 14,101 → 5,321 | 2.65x |
| 8 | Mistral TP=1, batch=1 | 2 | 12,496 → 5,431 | 2.30x | 11,336 → 5,798 | 1.96x |
| 16 | Qwen TP=1, batch=4 | 1 | 15,000 → 15,000 | 1.00x | 14,101 → 14,101 | 1.00x |

The sixteen-group case is a negative control: no splitting is needed and both
representations have identical timing/counts. In the other cases, issued K/V
bytes fall from 8 MiB to 0.5 / 2 / 4 MiB. At 2048 bytes/cycle, the issued-stream
bound falls from 4096 cycles to 256 / 1024 / 2048 cycles. The logical useful-work
roofline is unchanged between each pair; the padded baseline moves extra bytes.

QK's measured speedup can exceed the byte-reduction factor because the baseline
also loads dummy resident queries and repeats those loads across more M tiles.
For the one-group case, total GH resident commands fall from 14,336 to 56.
Do not interpret 24.92x as a pure sixteen-bank streaming speedup. SV pays for
short row segments even after splitting: one-group SV has sixteen 8-column
segments and 1,648 row-wait cycles, so it remains far above its 256-cycle stream
bound. Unlike the separate-program comparison, output-store counts are equal
within these controlled split/no-split pairs; the earlier output-model caveat
still applies.

Reproduce this additional comparison with:

```bash
python3 -B tests/regression/analyze_gqa_sharding.py \
  --binary build/dramsim3main --output /tmp/gqa-sharding-new
```

Recorded run: `/tmp/hbnpu-gqa-sharding-final-20261011`. All eight pairs
completed with timing, row-contract, byte-count, and all-bank address checks;
both sixteen-group controls matched exactly. The
[split/no-split CSV](2026-10-gqa-sharding.csv) records the measurements.

## Reproduce the main trace/roofline sweep

```bash
cmake -S . -B build -DCMD_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
ctest --test-dir build --output-on-failure
python3 -B tests/regression/analyze_gqa_roofline.py \
  --binary build/dramsim3main --output /tmp/gqa-roofline-new
```

The output must not exist. It contains manifests, instruction traces, raw
command/row traces, separate-group baselines, and JSON/CSV summaries.

Recorded run: `/tmp/hbnpu-gqa-hybrid-final-20261011` (local audit artifacts,
not a path supplied by the repository; rerun the commands above for new traces).
All 16 packed programs and their 144 separate-group baseline programs completed;
timing, row-contract, stream-byte, and absolute bank-address checks passed.
The [CSV results](2026-10-gqa-hybrid-roofline.csv) retain all cases, including
padding tails and individual prefill prefixes.

Final regression run: 14/14 CTest targets passed (188.66 seconds), including
canonical all-bank attention traces, a packed 4+2-group/9-query chunk case,
and the pre-existing OPT end-to-end regression. Tests using the historical
128-bit fixture remain historical-fixture tests, not canonical coverage.
