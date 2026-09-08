
# HB-NPU Simulator
HB-NPU simulator runs the HB-DRAM PNM accelerator, _HB-NPU_, based on DRAMsim3 open source DRAM simulator.
We implemented python programs to generate and run the LLM workloads and traces.

We modified DRAMsim3 simulator to support the custom trace format for HB-NPU and the custom DRAM commands of Local HB and Global HB operations.
Also we implemented Custom command scheduler that executes BLAS functions for the given LLM workloads by generating HB-NPU DRAM commands and enqueueing them to each channel controller of DRAM.

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
# cmake out of source build
mkdir build
cd build
cmake .. -DCMD_TRACE=1

# Build dramsim3 library and executables
make -j4

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

### Reproduce the results from the paper

To run all OPT workloads in our evaluation, enter below command. (We will add the support for other LLMs soon.)
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

### Creating new trace files and workloads
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
Or you can add a new model configuration by adding it to ```models``` file and re-run the above generators.
```
# [model_name] [parameter size (B)] [# of layers] [d_model] [# of heads] [d_head] [TP] [PP]
OPT-2.7B 2.7 32 2560 32 80 32 1
```
### Running an example trace file
You can run HB-NPU with a sample trace of a matrix multiplication kernel using the below command.
Trace files are in ```traces/``` folder.
```bash
./build/dramsim3main configs/HBM2_8Gb_x128.ini -c 5000000 -t [trace_file]
```
You can check the simulation results immediately on console.
```bash
3374 End of Computation 0 
3409 Output Exhausted. Array0 Turn off PIM mode. # Completed operations in 3409 cycles.
Turn off PIM
```
You can see the command trace and statistics in ```dramsim3ch_[0-7]cmd.trace``` and ```dramsim3.txt```.
Command trace shows the cycles and addresses of executed operations with their command types.
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
## Code Structure

```
├── configs                 # Configs of various protocols that describe timing constraints and power consumption.
├── ext                     # 
├── scripts                 # Tools and utilities
├── src                     # DRAMsim3 source files
├── tests                   # Tests of each model, includes a short example trace
├── CMakeLists.txt
├── Makefile
├── LICENSE
└── README.md

├── src  
    bankstate.cc: Added command supports for HB-NPU commands (LH_READ, GH_READ, etc.).
    channelstate.cc: Added update process for HB-NPU command timing and states
    configuration.cc: Added support for HB-NPU commands.
    controller.cc: Maintains the per-channel controller. We added the _in-order_ HB-NPU command queue management process.
    cpu.cc: handles PIM transactions and manages PIM transaction queue
    dram_system.cc:  Since DRAMsim3 implements per-channel controller originally, we added the upper hierarchy control scheme here. Our centralized custom controller generates DRAM commands for entire system simultaneously and schedules them to in-order command queues in per-channel controllers.
    refresh.cc: Added refresh checking process for pausing HB-NPU operations not to be interrupted by refresh.
    timing.cc: Added support for HB-NPU commands.
```

