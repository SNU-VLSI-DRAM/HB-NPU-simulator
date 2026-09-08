# Single-array workload and trace format

HB-NPU executes one kernel on the full array. Spatial multi-cut configuration,
partition IDs, next-cut fields, and launch masks have been removed from both
the simulator and the input formats.

## Workload files

Exactly two records are required:

```text
tile_M, post_delay, mcf, ucf, df
M, K, N
```

For example, `workloads/OPT-2.7B/decode/createQKV` contains:

```text
2048, 8, 2, 8, 1
80, 2560, 8
```

`tile_M`, `mcf`, and `ucf` retain their power-of-two encoding. `df=0` selects
the weight-stationary dataflow and `df=1` the TS-GEMM/GEMV dataflow.
`post_delay` is retained unchanged in the workload format; as before, the
trace generator does not encode or use it. The M/K/N dimension transforms
and base-row allocation rules are unchanged.

## PIM transaction addresses

Each trace line is still `hex_address PIM earliest_submission_cycle`.
Queue backpressure can delay submission; these are not scheduled DRAM-command
issue cycles. Bit ranges below are inclusive and numbered from bit 0 (LSB).

| Transaction | Bits | Meaning |
|---|---|---|
| Launch | 0 | `1`; canonical address is `0x1`, with no payload or mask |
| Configuration | 0 | `0` |
| Configuration | 1–2 | `3`, configuration marker |
| Configuration | 3–5 | `log2(mcf)` |
| Configuration | 6–8 | `log2(ucf)` |
| Configuration | 9 | `df` |
| Configuration | 10–13 | `log2(tile_M)` |
| Configuration | 14–18 | `kernel_size`, retained existing field |
| Configuration | 19–23 | `stride`, retained existing field |
| Workload | 0 | `0` |
| Workload | 1–2 | Dimension selector: `0=M`, `1=K`, `2=N` |
| Workload | 3–34 | Transformed dimension, as in the existing dataflow mapping |
| Workload | 35–56 | Base row |

Unused high bits are zero in generated traces. The decoder retains the existing
signed `int` representation of dimensions. These field widths describe the
encoding, not a guarantee that every encodable combination is supported by a
dataflow or hardware configuration.

The generator emits configuration at cycle 0, the three workload transactions
at cycles 1–3, and launch at cycle 4. Configuration initializes one
`PimExecutionState` and sets an explicit `configured` flag. Launch stays queued
until dataflow configuration and nonzero M, K, and N are present. Configuration
must precede workload records; the queue is processed in order.

For the example above, file mode with inferred base rows emits:

```text
0x2ece PIM 0
0x64800000140 PIM 1
0x64800000a02 PIM 2
0x204 PIM 3
0x1 PIM 4
```

Model mode can supply a different base row, so its workload addresses can differ
from this file-mode example.

## Migration and behavior checks

This is an intentional format break, not a backward-compatible extension.
There is no format-version tag or automatic legacy-trace detection. Do not
feed old traces to the new simulator: their addresses have different meanings.
Regenerate external workloads and traces using the updated generators. For
handwritten old single-array workloads, remove the first two `1, 1` header
entries, then regenerate the trace; do not merely reuse the old trace.

The repository migration rewrote 12,840 workload headers and 27,984 trace files,
plus five regression input fixtures and the root sample `dramsim3addr.trace`.
It preserved all remaining decoded fields and transaction cycles. The
format-only cleanup did not regenerate golden command streams or statistics.
The subsequent [GH timing correction](gh-streaming-timing.md) intentionally
changes GH issue cycles and recalculates GH stage delays; its changed timing
baselines are documented separately. Run
`ctest --test-dir build --output-on-failure` for the current regression suite.
Commands and cycles are compared exactly; energy fields retain the existing
floating-point tolerance.
