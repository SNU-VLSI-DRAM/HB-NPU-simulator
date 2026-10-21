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

## Instruction traces

Each line is `hex_word PIM earliest_submission_cycle`. The word is part of a
versioned tile-instruction packet, not a DRAM address. CPU queue backpressure
can delay submission; timestamps do not prescribe DRAM-command issue cycles.

`gen_LLM_trace.py` compiles OPT workloads into `CONFIG`, `STORE`,
`LOAD_INPUTS`, `TILE_GEMM`, and `END` packets. `gen_tile_trace.py` compiles a
single workload. The simulator executes these instructions without a mode
setting. See [packet layouts and execution rules](pim-instruction-format.md).

For the workload above:

```bash
python3 gen_tile_trace.py \
  -w workloads/OPT-2.7B/decode/createQKV \
  -s configs/HBNPU_8Gb.ini -t /tmp/opt-createqkv.trace
```

Use a fresh destination or explicitly pass `--overwrite`.
Model-wide generation with `gen_LLM_trace.py` writes to `output/traces/`
by default; `-p` selects another root.

## Existing paper traces

The five-record configuration/dimension/launch traces retained in `traces/`
are historical inputs. The current simulator rejects that format. Regenerate
instructions from the corresponding workloads; changing timestamps or header
fields does not convert a trace.

`run_demo.py` and `run_models.sh` generate fresh instruction traces in a temporary
directory and leave those supplied paper traces unchanged.
