# HB-NPU Simulator

HB-NPU Simulator models the HB-DRAM near-memory accelerator described in our
paper, using DRAMsim3 for DRAM timing and energy accounting. Python utilities
generate OPT prefill and decode workloads and encode their kernel
configurations into PIM transaction traces. The simulator schedules the
corresponding Local HB (LH) and Global HB (GH) bank operations dynamically.
Dense Qwen2/Mistral configurations can generate grouped-query attention (GQA) and projection
matrix workloads through this instruction path.

If you use HB-NPU for your research, please cite our [paper](https://ieeexplore.ieee.org/document/11132870):
```
@INPROCEEDINGS{11132870,
  author={Han, Sanghyeok and Yoon, Byungkuk and Park, Gyeonghwan and Song, Choungki and Kim, Dongkyun and Kim, Jae-Joon},
  booktitle={2025 62nd ACM/IEEE Design Automation Conference (DAC)}, 
  title={Near-Memory LLM Inference Processor based on 3D DRAM-to-logic Hybrid Bonding}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  keywords={Three-dimensional displays;Design automation;Large language models;AI accelerators;Computer architecture;Bandwidth;Data transfer;Boosting;Bonding},
  doi={10.1109/DAC63849.2025.11132870}}

```

## Supported workloads

| Workload | Entry point | Scope |
|---|---|---|
| OPT prefill/decode | `run_demo.py`, existing workload generators | Paper-oriented kernel timing workflows |
| Dense Qwen2 / Mistral GQA | `gen_gqa_workload.py` | Grouped QK/SV and Q/K/V/O, gate/up/down matrix kernels |
| Explicit tile instructions | `gen_tile_trace.py` | Compile existing workload descriptions into instruction traces |

Start with [building](#building), then use the
[Qwen/Mistral quick start](#dense-qwen2mistral-matrix-workloads) or
[OPT evaluation](#run-the-opt-evaluation). The canonical hardware configuration
is `configs/HBNPU_8Gb.ini`; do not substitute the historical `_pim.ini` fixture
when interpreting canonical performance results.

## Building and running the simulator

The simulator is based on [DRAMsim3](https://github.com/umd-memsys/DRAMsim3).
Use a C++11 compiler, CMake, and Python 3.7+. The GQA and tile-instruction
generators use the Python standard library only. The legacy OPT evaluation
and spreadsheet utilities also use:

```bash
python3 -m pip install openpyxl numpy
```

### Building

The commands below use CMake 3.13+ (`-S`/`-B` syntax). Build outside the source
directory and enable command tracing for the regression tests.

```bash
cmake -S . -B build -DCMD_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4

```

The build process creates `dramsim3main` and executables in the `build` directory.
By default, it also creates `libdramsim3.so` shared library in the project root directory.

### Testing

Configure a Debug build with command tracing, build all default targets, and
run the fast unit and regression checks:

```bash
cmake -S . -B build -DCMD_TRACE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j4
ctest --test-dir build --output-on-failure -L quick
```

Run the complete matrix, including the longer OPT-2.7B comparison, with:

```bash
ctest --test-dir build --output-on-failure
```

Regression comparisons are exact except for floating-point energy fields,
which use a relative tolerance of `1e-6` plus an absolute tolerance of `1e-9`.
Ordinary test runs never update goldens. Golden changes must be explicit via
`tests/regression/run_regression.py --update-golden` and reviewed separately.

The full suite includes C++ execution and packet tests, timing checks, workload
parsing, five exact legacy quick fixtures, a 46-kernel OPT window, CPU instruction
transport tests, and both exact and instruction-mode comparison tools.
A canonical-hardware audit against the original source covered 160 OPT kernel
pairs: 149 completed with matching data streams; 11 matched only their common
100,000-cycle prefixes. See the
[canonical audit](docs/refactoring/2026-10-canonical-configuration-revalidation.md).
The [September comparison](docs/refactoring/2026-09-original-master-llm-comparison.md)
records an earlier implementation and must not be conflated with this audit.
GQA tests cover packing/chunk identities, decode and prefill kernels, canonical
all-bank row/address coverage, completion, and command timing.
These checks validate simulator command behavior, not numerical model outputs.

To compare two command-tracing builds on the same kernel, including a bounded
run of a large workload:

```bash
python3 tests/regression/compare_binaries.py \
  --baseline /path/to/reference/build/dramsim3main \
  --candidate /path/to/candidate/build/dramsim3main \
  --config configs/HBNPU_8Gb.ini \
  --trace traces/OPT-66B_1024_128_128/prompt/L2 \
  --cycles 100000 --output /tmp/hbnpu-comparison-new
```

Use separate source trees: each build writes its shared library into its source
root. The output directory must not already exist. This tool compares physical
and logical command records exactly, including cycles, and prints the first
difference. It retains raw traces and a JSON result, applies the same energy
tolerance as the regressions, and distinguishes full completion from a matching
bounded run. It currently expects the eight-channel configuration used above.

### Execute tile instructions

Legacy execution remains the default. The canonical HB-NPU configuration is
`configs/HBNPU_8Gb.ini` (renamed from `HBM2_8Gb_x128.ini`, with unchanged settings).
It models 64-bit device/bus width, BL4, 32-byte requests, and 64 addressable
bursts per row after HBM column normalization. `channel_size = 256` is MiB.
The separate `HBM2_8Gb_x128_pim.ini` is a historical regression fixture with
different geometry and timing; its golden results do not validate this profile.
See the [canonical configuration audit](docs/refactoring/2026-10-canonical-configuration-revalidation.md)
for the original-source comparison, full-row attention checks, and corrected
roofline interpretation.
To use explicit tile instructions, copy `configs/HBNPU_8Gb.ini` and add:

```ini
[hbnpu]
execution_mode = tile_instructions
instruction_capacity = 16
```

Compile one existing workload, then run its instruction trace with that copy:

```bash
mkdir -p output/tile-example
python3 gen_tile_trace.py \
  -w workloads/OPT-2.7B/decode/createQKV \
  -s configs/HBNPU_8Gb.ini \
  -t output/tile-example/instructions.trace
./build/dramsim3main /path/to/instruction-mode.ini \
  -t output/tile-example/instructions.trace -c 100000 \
  -o output/tile-example
```

Each 64-bit word is one ordinary PIM transaction. The CPU retries when the
32-word ingress FIFO is full; it does not wait for each tile to finish.
The translator emits `CONFIG`, `STORE`, `LOAD_INPUTS`, `TILE_GEMM`, and `END`.
`LOAD_INPUTS` loads the resident operand (the executor's weight-loading stage);
`TILE_GEMM` streams the other operand from banks while computing. GEMM can be
dispatched while its matching load is pending, but cannot stream until that
load finishes. Loading alone never starts GEMM. `STORE` arms an output
descriptor before its compute group but cannot write before data is ready.
Existing fused `TILE_BLOCK` packets remain supported; use `--fused` with the
translator to generate them. Separate vector-processing instructions are not
supported, and no extra resident buffer or cross-tile reuse is modeled.

Instruction mode decomposes GEMM tiles into row-bounded operations inside
HB-NPU. Bank preparation and refresh may delay a row's start; after a common
start across channels, the read stream follows a fixed GH/LH schedule. A missed
beat is an error, not an implicit stall. `hbnpu_rows.jsonl` records completed
row intervals for checking against command traces. Set `row_lockstep = false`
in `[hbnpu]` to use the earlier per-vector execution path for comparison.
This models command-issue predictability within one HB-NPU, not physical PVT
variation or inter-device communication.

Check `hbnpu_instruction.json` in the output directory for `end_completed`,
buffer occupancy, CPU backpressure, and starvation counters. A cycle-capped or
truncated program is reported as incomplete, not successful. Instruction-mode
cycle counts include upload and dispatch delays; legacy exact regressions are
unchanged. See [the packet format and execution rules](docs/pim-instruction-format.md).

### Run the OPT evaluation

To run all OPT workloads in the configured evaluation set, use:
It can take about 30 minutes or longer depending on the running environment.
```bash
bash run_models.sh
```
To run a specific workload, enter below command.
```bash
# Running OPT-66B with input tokens 1024, output tokens 128, batch size 128
python3 run_demo.py -m OPT-66B -i 1024 -o 128 -b 128
```
You can check the detailed results reported to ```result.xlsx```.

For historical comparisons with the paper, the pre-update revision is
`502b52750778242c3a76845e40af32674e5baefd`. Use that revision with its matching
configurations and input traces. Current timing corrections can change cycle
counts relative to that version.

### Creating new trace files and workloads
The simulator now uses a single-array workload/trace format. Workload headers
contain `tile_M, post_delay, mcf, ucf, df`; the old cut fields and partition
selection bits have been removed. Checked-in workloads and traces are migrated.
**Old external traces are incompatible and must be regenerated** with the
updated workload generators and `gen_LLM_trace.py`. Removing the two header
fields alone does not convert an existing trace. See the
[workload and trace format](docs/pim-trace-format.md) for the new bit layout.

You can generate the LLM trace files by running ```gen_LLM_trace.py```. 
```bash
# generate traces for OPT-2.7B with input tokens 1024, output tokens 128, and batch size 128
python3 gen_LLM_trace.py -s configs/HBNPU_8Gb.ini -i 1024 -o 128 -b 128 -m OPT-2.7B
```

Also you can generate the new LLM workloads by running ```gen_workload_prompt.py``` and ```gen_workload_decode.py```.
```
# generating prompt workload for 128 tokens.
python3 gen_workload_prompt.py -s 128
# generating decode workload for tokens from 10-th to 1600-th generation and batch sizes from 32 to 128 (32, 64, 128, power of twos).
python3 gen_workload_decode.py -s 10 -e 1600 -sb 32 -eb 128
```
The `models` file supplies parameters to the existing workload generators.
Adding a configuration does not automatically implement a different model
architecture. Its fields are:
```
# [model_name] [parameter size (B)] [# of layers] [d_model] [# of heads] [d_head] [TP] [PP]
OPT-2.7B 2.7 32 2560 32 80 32 1
```
### Running an example trace file
You can run HB-NPU with a sample trace of a matrix multiplication kernel using the below command.
Trace files are in ```traces/``` folder.
```bash
mkdir -p output/example
./build/dramsim3main configs/HBNPU_8Gb.ini \
  -c 100000 \
  -t traces/OPT-2.7B_128_1024_32/prompt/createQKV \
  -o output/example/
```
`-c` limits simulated cycles. The simulator can finish earlier when the kernel
and its output transfers complete, reported by `Output Exhausted` and
`Turn off PIM`. Reaching the cycle limit alone does not establish completion.

The output directory contains:

| File | Contents |
|---|---|
| `dramsim3.json`, `dramsim3.txt` | Cycle counts, command statistics, and energy |
| `dramsim3ch_<channel>cmd.trace` | Physical effects: cycle, command, channel, rank, bank group, bank, row, column |
| `dramsim3ch_<channel>pim.trace` | Logical issue: cycle, channel, opcode, rank, target banks |

A broadcast or gang produces one logical record and multiple physical bank
records. Physical writes retain `pim_write` / `pim_write_p`; the logical trace
identifies the LH/GH operation. Row and column fields are hexadecimal.
The following physical trace excerpts illustrate the format; addresses and
cycles depend on the workload and configuration.
```bash
# Global HB Read
5                  pim_activate           1   0   0   0   0x199d      0x0
19                 gh_read                1   0   0   0   0x199d      0x0
21                 gh_read                1   0   0   0   0x199d      0x1
23                 gh_read                1   0   0   0   0x199d      0x2
25                 gh_read                1   0   0   0   0x199d      0x3
27                 gh_read                1   0   0   0   0x199d      0x4
29                 gh_read                1   0   0   0   0x199d      0x5
31                 gh_read                1   0   0   0   0x199d      0x6
...
139                gh_read                1   0   0   0   0x199d     0x3c
141                gh_read                1   0   0   0   0x199d     0x3d
143                gh_read                1   0   0   0   0x199d     0x3e
145                gh_read_p              1   0   0   0   0x199d     0x3f
# Local HB Read
164                pim_activate           1   0   0   0      0x0      0x0
164                pim_activate           1   0   0   1      0x0      0x0
164                pim_activate           1   0   0   2      0x0      0x0
164                pim_activate           1   0   0   3      0x0      0x0
194                pim_activate           1   0   1   0      0x0      0x0
194                pim_activate           1   0   1   1      0x0      0x0
194                pim_activate           1   0   1   2      0x0      0x0
194                pim_activate           1   0   1   3      0x0      0x0
224                pim_activate           1   0   2   0      0x0      0x0
224                pim_activate           1   0   2   1      0x0      0x0
224                pim_activate           1   0   2   2      0x0      0x0
224                pim_activate           1   0   2   3      0x0      0x0
254                pim_activate           1   0   3   0      0x0      0x0
254                pim_activate           1   0   3   1      0x0      0x0
254                pim_activate           1   0   3   2      0x0      0x0
254                pim_activate           1   0   3   3      0x0      0x0
268                lh_read                1   0   0   0      0x0      0x0
268                lh_read                1   0   0   1      0x0      0x0
268                lh_read                1   0   0   2      0x0      0x0
268                lh_read                1   0   0   3      0x0      0x0
268                lh_read                1   0   1   0      0x0      0x0
268                lh_read                1   0   1   1      0x0      0x0
268                lh_read                1   0   1   2      0x0      0x0
268                lh_read                1   0   1   3      0x0      0x0
268                lh_read                1   0   2   0      0x0      0x0
268                lh_read                1   0   2   1      0x0      0x0
268                lh_read                1   0   2   2      0x0      0x0
268                lh_read                1   0   2   3      0x0      0x0
268                lh_read                1   0   3   0      0x0      0x0
268                lh_read                1   0   3   1      0x0      0x0
268                lh_read                1   0   3   2      0x0      0x0
268                lh_read                1   0   3   3      0x0      0x0
```
The command statistics shows the number of pim commands, their row hits, and energy, etc.
```bash
# Local HB reads
num_lh_read_row_hits          =        10094   
num_lh_read_cmds              =        10272   
# Global HB reads
num_gh_read_row_hits          =        1435   
num_gh_read_cmds              =        1500   
```
## Dense Qwen2/Mistral matrix workloads

The generator supports the canonical 64-bit `HBNPU_8Gb.ini` and the historical
128-bit `_pim.ini` fixture. Use the canonical configuration for HB-NPU analysis.

`gen_gqa_workload.py` imports a local dense `qwen2` or `mistral` configuration
and emits grouped QK/SV plus separate Q/K/V/O and gate/up/down projection
programs. Existing OPT generators and instruction encodings are unchanged.

```bash
python3 gen_gqa_workload.py \
  --model-config tests/fixtures/models/qwen2.5-7b.json \
  --hardware configs/HBNPU_8Gb.ini \
  --phase decode --sequence-length 143 --operations qk sv \
  --output /tmp/qwen-gqa-143

mkdir -p /tmp/qwen-gqa-run
./build/dramsim3main /tmp/qwen-gqa-143/simulator.ini \
  -t /tmp/qwen-gqa-143/000000_qk.trace -c 100000 -o /tmp/qwen-gqa-run
```

Run these commands from the repository root after building. Generation creates
`simulator.ini`, `manifest.json`, and numbered instruction traces. The supplied
example produces one packed QK program and one packed SV program. Run each
trace separately in its own output directory and confirm `end_completed: true`
in `hbnpu_instruction.json`; reaching `-c` is not evidence of completion.

To use Mistral, replace `--model-config` with
`tests/fixtures/models/mistral-7b-v0.3.json` and choose a fresh output directory.
Use `--batch` for independent batch items, `--phase prefill` for per-token
causal prefixes, or omit `--operations` to include projection kernels as well.
The generated configuration enables instruction execution, row lockstep, and
GH read-arrival timing. It disables self-refresh, not periodic DRAM refresh.

The bundled JSONs are dimension-only fixtures verified against the publisher
configs for [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json)
and [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/blob/main/config.json),
not complete inference configurations. No downloads occur during generation.

- Queries sharing a KV head are resident: QK is `[g,d] @ [d,L]` and SV is
  `[g,L] @ [L,d]`, with K/V LH-streamed. SV assumes normalized scores supplied
  externally. There is no attention-specific opcode.
- The profile is eight channels and sixteen banks/channel. Attention packs
  independent KV groups/batch items first, in power-of-two packs of up to 16.
  For `G` groups, `ucf=G`, `mcf=16/G`: each group shards its output columns over
  `16/G` banks. Non-power-of-two remainders use smaller packs (six becomes 4+2).
  Qwen2.5-7B batch 1 packs four groups; batch 4 packs sixteen. Mistral-7B batch 1
  packs eight. Different layers, query positions, shapes, and causal prefixes
  are never combined. Projections keep their independent single-record lowering.
  Hardware geometry must have representable power-of-two rows/columns and a
  nonzero rank capacity; unsupported profiles are rejected before file creation.
  Logical K/output-column dimensions are padded to multiples of 128. Up to eight
  active query/token rows per group share the existing output protocol; fewer
  active rows do not reduce its write count. Larger groups split into chunks of
  eight, each charging another complete stream. Inactive lanes are not extra
  logical query outputs.
- `manifest.json` format version 2 records valid/padded shapes per group, head IDs, logical row slices,
  tensor/view identity, disjoint physical row regions and each instruction trace.
  Chunks share the same K/V view region rather than copying the cache per head.
  `members` is authoritative for packed programs: it records each batch/KV head,
  tensor identity, resident-vector range, and bank/column stripe. Legacy singular
  tensor/head fields are null for multi-group programs, not aliases of member 0.
  Different padded prefix views may use different physical packings; this is not
  a persistent online KV-cache implementation. Padding/layout conversion costs
  are not simulated.
- Output-store counts retain the existing fixed-footprint LH timing protocol.
  They are not a byte-accurate materialization of every logical query output.
  Roofline comparisons therefore distinguish ideal logical traffic from actual
  issued stream traffic; they do not establish full inference accuracy.
- Decode length includes the current token. Prefill (`--phase prefill`) emits
  attention separately for each causal prefix, preserving token/head identity.
- The CLI selects one layer (`--layer 0` by default) of TP rank 0 / PP stage 0.
  `--tp` must divide Q/KV heads and intermediate size; `--pp` must divide layers.
  O/down projection outputs are local contributions; collectives are excluded.
  The Python generator can enumerate all local layers. This is not a full-model
  memory-fit or end-to-end latency report.
- Each trace is an independent CONFIG-to-END program: run it separately and
  check `hbnpu_instruction.json` for `end_completed`. A cycle-capped incomplete
  run is not a completed kernel. Output directories must not already exist.
- Tensor values, softmax, RoPE, normalization, activation/gating, bias/residuals,
  embeddings/lm_head and KV updates are not executed. MoE and enabled sliding
  windows are rejected. Numerical inference correctness is not claimed.

Run the grouped workload tests with
`ctest --test-dir build -R hbnpu_gqa --output-on-failure` after rebuilding.
For canonical command-trace checks, roofline bounds, and the measured
head/batch-packing tradeoffs, see the
[GQA trace/roofline report](docs/refactoring/2026-10-gqa-hybrid-roofline.md).

## Code structure

HB-NPU-specific sources and headers live in `src/hbnpu/`. Shared DRAM and
host simulation files remain directly under `src/`.

| Path | Responsibility |
|---|---|
| `gen_workload_prompt.py`, `gen_workload_decode.py` | Generate prefill/decode kernel workloads |
| `gen_LLM_trace.py` | Encode workload parameters and base rows into PIM transactions |
| `hbnpu_model.py`, `hbnpu_gqa.py`, `gen_gqa_workload.py` | Validate local dense model configs, lower grouped LH matrices, and emit instruction programs/manifests |
| `src/cpu.cc` | Submit transactions with queue backpressure |
| `src/hbnpu/pim_transaction.{h,cc}` | Decode PIM transactions |
| `src/hbnpu/pim_config.h`, `src/hbnpu/pim_execution_state.h` | Decoded configuration, launch staging, and legacy progress diagnostics |
| `src/hbnpu/pim_kernel_context.h` | Launch-time kernel context and block descriptors |
| `src/hbnpu/legacy_tile_scheduler.{h,cc}` | Select compute and output blocks in legacy traversal order |
| `src/hbnpu/pim_tile_executor.{h,cc}` | Execute supplied blocks, track issued vectors, and preserve stage timing |
| `src/dram_system.cc` | Coordinate ingress, block selection, refresh, and channel dispatch |
| `src/hbnpu/pim_operation.h`, `src/hbnpu/pim_command_batch.h` | Represent logical PIM operations and batches |
| `src/controller.cc` | Arbitrate per-channel commands and issue eligible bank operations |
| `src/channel_state.cc`, `src/bankstate.cc`, `src/timing.cc` | Track bank state and enforce timing constraints |
| `src/refresh.cc` | Manage refresh scheduling |
| `configs/`, `workloads/`, `traces/` | DRAM configurations and generated inputs |
| `tests/`, `docs/` | Regression coverage and user documentation |

Block selection and execution are separate. The executor owns the active block
and its vector/stage progress; it cannot select the next matrix block on its own.
The legacy scheduler supplies blocks synchronously, without adding simulated
handoff cycles. Command readiness and bank timing remain in the channel
controllers. See the [legacy workload format](docs/pim-trace-format.md) and
[instruction packet format](docs/pim-instruction-format.md) for the two input paths.

## Documentation

- [GH/LH timing assumptions and validation](docs/gh-streaming-timing.md)
- [Instruction format, buffer behavior, and execution rules](docs/pim-instruction-format.md)
- [Canonical hardware and original-source audit](docs/refactoring/2026-10-canonical-configuration-revalidation.md)
- [GQA mapping, trace checks, and roofline analysis](docs/refactoring/2026-10-gqa-hybrid-roofline.md)
- [Earlier refactoring and format migration](docs/refactoring/2026-09-hbnpu-updates.md)
