# Original-master comparison on real OPT workload traces

Reference: original local `master` at `502b52750778242c3a76845e40af32674e5baefd`.
Candidate: updated HB-NPU implementation; the tested binary fingerprints are recorded below.
No merge or branch-pointer change was performed.

[Per-kernel cycle and energy results](2026-09-original-master-llm-comparison.csv) are saved alongside this report.

## Scope and method

- Models: OPT-2.7B, OPT-6.7B, OPT-13B, OPT-66B.
- Scenarios: 128 input / 1024 output tokens, batch 32; 1024 input / 128 output tokens, batch 128.
- Six prefill kernel families; four decode linear kernels in both dataflows; QK/SV at first, middle and last decode context positions.
- Decode attention contexts: 128, 640, 1151 for the short-input scenario; 1024, 1088, 1151 for the long-input scenario.
- Full-kernel runs use a 45-second per-execution timeout. The initial eight-minute submission budget is followed by bounded two-minute and one-minute continuations to finish the same selected decode sample set.
- Matching checked-in traces are independently decoded in both formats: remaining kernel fields, dimensions, base rows and submission timestamps must be identical.
- Configuration: byte-identical `configs/HBM2_8Gb_x128.ini` (8 channels, 16 banks/channel; tCK=1, tCCD_S=1, tCCD_L=2, tFAW=30).
- Original source is built from a Git archive in this audit directory; candidate executable/library are copied before testing. Both use Debug and command tracing.
- Comparison: per-bank ordered (physical data opcode, row, column) counts and SHA-256; whole physical command traces including timestamps are hashed separately.
- Independent trace checks: rank tFAW activation count, shared GH read/write slot, data row open, candidate logical issue slot and opcode target width.
- Rank-maintenance records may have channel=-1; their channel is resolved from their trace filename.

## Results

- Completed pairs: 159; unsuccessful/incomplete pairs: 1; planned: 160.
- Distinct decoded trace inputs among completed pairs: 156 (some model-labelled attention kernels share a trace).
- Identical per-bank data streams: 159/159.
- Compared data-command bank effects per version: 28,629,536.
- Identical full physical command traces including cycles: 0/159.
- master observed trace violations: {'global_slot': 10412992}.
- updated observed trace violations: {}.

Cycle changes below are sums of independent kernel runs, not end-to-end model latency.

| Model | Pairs | Original cycles | Updated cycles | Cycle change | Energy change |
|---|---:|---:|---:|---:|---:|
| OPT-2.7B | 40 | 601,007 | 573,887 | -4.51% | -0.87% |
| OPT-6.7B | 40 | 1,065,574 | 1,004,760 | -5.71% | -1.29% |
| OPT-13B | 40 | 1,088,329 | 1,030,140 | -5.35% | -1.16% |
| OPT-66B | 39 | 1,408,742 | 1,304,696 | -7.39% | -1.83% |

| Phase | Pairs | Cycle change |
|---|---:|---:|
| Prefill | 47 | -2.95% |
| Decode linear, LH | 32 | -1.06% |
| Decode linear, WS/GH | 32 | -18.24% |
| Decode attention | 48 | -0.53% |

## Largest slowdowns

| Workload/kernel | Original | Updated | Change |
|---|---:|---:|---:|
| OPT-66B_128_1024_32/prompt/Wo | 32940 | 33625 | +2.08% |
| OPT-13B_128_1024_32/prompt/Wo | 18172 | 18538 | +2.01% |
| OPT-2.7B_128_1024_32/prompt/Wo | 9088 | 9271 | +2.01% |
| OPT-2.7B_1024_128_128/decode/WS/Wo | 9088 | 9271 | +2.01% |
| OPT-6.7B_128_1024_32/prompt/Wo | 14480 | 14767 | +1.98% |
| OPT-2.7B_128_1024_32/prompt/QK | 429 | 435 | +1.40% |
| OPT-6.7B_128_1024_32/prompt/QK | 429 | 435 | +1.40% |
| OPT-13B_128_1024_32/prompt/QK | 429 | 435 | +1.40% |
| OPT-66B_128_1024_32/prompt/QK | 429 | 435 | +1.40% |
| OPT-2.7B_128_1024_32/prompt/SV | 429 | 435 | +1.40% |

## Concrete timing difference (retained pilot traces)

For OPT-2.7B / input 128 / output 1024 / batch 32 / prompt createQKV, the first differing channel-0 physical record is the second activation: bankgroup 2, bank 0, row 0 activates at cycle 5 in master versus cycle 6 in the candidate. Master issues both addressed bank activations at cycle 5; the candidate sequences them one per cycle.

The first two output banks write simultaneously at cycle 8322 in master (next pair 8324). The candidate writes them at 7026 and 7027 (next pair 7028 and 7029): one transfer per cycle, two-cycle cadence per bank. The earlier output start also reflects the GH-stage clock correction, not just write serialization.

The retained pilot also passes explicit checks for 41,984 GH data records against the per-bank two-cycle cadence, and 42,656 logical-command target members against their physical bank-effect records. `check_pilot.py` reproduces these checks and the first-difference output.

## Limits

These are generated kernel traces from the real supported OPT workload set, not synthetic quick fixtures. They do not execute model weights or establish floating-point output accuracy. Each kernel starts from a fresh simulator; decode samples only three context positions rather than all generated tokens. The results do not cover full-pipeline/layer transitions, other model architectures, every batch/configuration, or arbitrary shapes.

Timing differences are expected because the candidate includes intentional GH-clock/interleaving and command-issue corrections in addition to refactoring. Exact bank streams do not prove that every timing rule is correct; the listed invariants are the ones explicitly checked.

## Reproduction

```bash
python3 /tmp/hbnpu-master-comparison-GzmU1g/compare_llm.py --budget 480 --timeout 45
python3 /tmp/hbnpu-master-comparison-GzmU1g/compare_llm.py --resume --budget 120 --timeout 45
python3 /tmp/hbnpu-master-comparison-GzmU1g/compare_llm.py --resume --budget 60 --timeout 45
python3 /tmp/hbnpu-master-comparison-GzmU1g/compare_prefix.py
python3 /tmp/hbnpu-master-comparison-GzmU1g/check_pilot.py
python3 /tmp/hbnpu-master-comparison-GzmU1g/summarize.py
```

The comparison script assumes the two existing workspace paths and preserved binaries in this audit directory. `manifest.json` contains all planned decoded kernel inputs; `results.json` includes per-bank counts/hashes, cycles, energy, command counts and checks for every attempted pair. `results.csv` is a compact per-kernel table. Large per-run traces are discarded after hashing; the pilot traces are retained.

## Binary fingerprints

- `reference-build/dramsim3main`: `e9abe864d7d9a8128a23c477a5e015974a05244493dbb4bbccd851e111d70ddf`
- `reference-source/libdramsim3.so`: `c1ba6b3bb20a00d6087371683acd3376ef305e333645fc1bcaf359efdbe130f2`
- `updated-bin/dramsim3main`: `a40a3a824b53ba2a8230b6c9703125b656374189792e3f7d4b476a1bc63f7e7b`
- `updated-bin/libdramsim3.so`: `80eb75a749a69a8a045130be6423fadda0e24521cd5e33aaf03b9e38eb14a97d`

## Incomplete/failed pairs

- OPT-66B_1024_128_128/prompt/L2: master=ok, updated=timeout.

## Bounded comparison of the timed-out kernel

- Case: `traces/OPT-66B_1024_128_128/prompt/L2`; both versions run for 20,000 simulated cycles.
- Equal common per-bank data prefixes: True; 115,232 bank effects compared.
- Updated observed trace violations: {}.
- This does not establish full-kernel completion or full-kernel cycle/energy equivalence. At a common cycle cutoff the versions can have issued different numbers of operations; only each bank's common prefix is compared. This case is excluded from the full-kernel performance tables.
- Raw partial traces and detailed counts are retained under `prefix-master/`, `prefix-updated/`, and `prefix-results.json` in the audit directory.

## Follow-up: 100,000-cycle comparison

- The same OPT-66B long-prefill L2 kernel was rerun to exactly 100,000 cycles on both preserved binaries.
- Matching common per-bank data prefixes: True; 575,280 bank effects compared.
- Master issued 575,280 data bank effects; the candidate issued 581,192. Additional candidate operations beyond each common prefix are not covered by the equivalence check.
- Candidate violations of the checked timing invariants: {}. Neither kernel completed within the cycle cutoff.
- The 20,000-cycle evidence is preserved. Extended raw traces and results are stored as `prefix-100000-master/`, `prefix-100000-updated/`, and `prefix-100000-results.json` in the audit directory.

```bash
python3 /tmp/hbnpu-master-comparison-GzmU1g/compare_prefix.py --cycles 100000
```
