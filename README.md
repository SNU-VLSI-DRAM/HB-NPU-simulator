
# HB-NPU Simulator
HB-NPU Simulator models the HB-DRAM near-memory accelerator described in our
paper, using DRAMsim3 for DRAM timing and energy accounting. Python utilities
generate OPT prefill and decode workloads and encode their kernel
configurations into PIM transaction traces. The simulator schedules the
corresponding Local HB (LH) and Global HB (GH) bank operations dynamically.

If you use HB-NPU for your research, please cite our [paper](https://ieeexplore.ieee.org/document/11132870),:
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


## Building and running the simulator

This simulator has been built based on DRAMsim3.
Follow the below instructions of DRAMsim3 to build the environment.
For details, goto link below:
https://github.com/umd-memsys/DRAMsim3/tree/master

Also we recommend to install following python packages to use our custom python programs to generate and run LLM workloads and trace files.
```
pip install configparser
pip install openpyxl
pip install numpy
```

### Building

We require CMake 3.0+ to build this simulator.
Doing out of source builds with CMake is recommended to avoid the build files cluttering the main directory.

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

The full suite contains six CTest groups, including 36 C++ test cases, timing
checks, workload parsing, five quick fixtures, and a 46-kernel OPT window.
A broader comparison across four OPT model sizes preserved per-bank data
streams in all 159 completed kernel pairs. The remaining OPT-66B kernel matched
over a 100,000-cycle common prefix; full completion of that case remains
unverified. See the [comparison report](docs/refactoring/2026-09-original-master-llm-comparison.md).
These checks validate simulator command behavior, not numerical model outputs.

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
python3 gen_LLM_trace.py -s configs/HBM2_8Gb_x128.ini -i 1024 -o 128 -b 128 -m OPT-2.7B
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
./build/dramsim3main configs/HBM2_8Gb_x128.ini \
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
## Code structure

| Path | Responsibility |
|---|---|
| `gen_workload_prompt.py`, `gen_workload_decode.py` | Generate prefill/decode kernel workloads |
| `gen_LLM_trace.py` | Encode workload parameters and base rows into PIM transactions |
| `src/cpu.cc` | Submit transactions with queue backpressure |
| `src/pim_transaction.{h,cc}` | Decode PIM transactions |
| `src/pim_config.h`, `src/pim_execution_state.h` | Kernel configuration and execution progress |
| `src/dram_system.cc` | Schedule matrix operations and dispatch channel work |
| `src/pim_operation.h`, `src/pim_command_batch.h` | Represent logical PIM operations and batches |
| `src/controller.cc` | Arbitrate per-channel commands and issue eligible bank operations |
| `src/channel_state.cc`, `src/bankstate.cc`, `src/timing.cc` | Track bank state and enforce timing constraints |
| `src/refresh.cc` | Manage refresh scheduling |
| `configs/`, `workloads/`, `traces/` | DRAM configurations and generated inputs |
| `tests/`, `docs/` | Regression coverage and user documentation |
