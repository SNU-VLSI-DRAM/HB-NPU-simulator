
# HB-NPU Simulator




## Building and running the simulator

This simulator has been built based on DRAMsim3.
Follow the below instructions of DRAMsim3 to build the environment.
For details, goto link below:
https://github.com/umd-memsys/DRAMsim3/tree/master

### Building

We require CMake 3.0+ to build this simulator.
Doing out of source builds with CMake is recommended to avoid the build files cluttering the main directory.

```bash
# cmake out of source build
mkdir build
cd build
cmake ..

# Build dramsim3 library and executables
make -j4

```

The build process creates `dramsim3main` and executables in the `build` directory.
By default, it also creates `libdramsim3.so` shared library in the project root directory.

### Running an example workload
You can immediately run HB-NPU with a sample trace of 128x128x128 matrix multiplication using the below command.
```bash
./build/dramsim3main configs/HBM2_8Gb_x128.ini -c 5000000 -t sample.trc
```
You can check the simulation results immediately.
```bash
3374 End of Computation 0 
3409 Output Exhausted. Array0 Turn off PIM mode. # Completed operations in 3409 cycles.
Turn off PIM
```
You can see the command trace and statistics in ```dramsim3ch_[0-7]cmd.trace``` and ```dramsim3.txt```.
Command trace shows the cycles and addresses of executed operations with their command types.
```bash
# Loading Weights
5                  pim_activate           1   0   0   0    0x191      0x0
19                 pim_read               1   0   0   0    0x191      0x0
21                 pim_read               1   0   0   0    0x191      0x1
23                 pim_read               1   0   0   0    0x191      0x2
25                 pim_read               1   0   0   0    0x191      0x3
27                 pim_read               1   0   0   0    0x191      0x4
29                 pim_read               1   0   0   0    0x191      0x5
31                 pim_read               1   0   0   0    0x191      0x6
33                 pim_read               1   0   0   0    0x191      0x7
35                 pim_read               1   0   0   0    0x191      0x8
37                 pim_read               1   0   0   0    0x191      0x9
39                 pim_read               1   0   0   0    0x191      0xa
41                 pim_read               1   0   0   0    0x191      0xb
43                 pim_read               1   0   0   0    0x191      0xc
45                 pim_read               1   0   0   0    0x191      0xd
47                 pim_read               1   0   0   0    0x191      0xe
49                 pim_read_p             1   0   0   0    0x191      0xf
# Streaming Inputs
68                 pim_activate           1   0   0   0      0x0      0x0
68                 pim_activate           1   0   0   1      0x0      0x0
68                 pim_activate           1   0   0   2      0x0      0x0
68                 pim_activate           1   0   0   3      0x0      0x0
98                 pim_activate           1   0   1   0      0x0      0x0
98                 pim_activate           1   0   1   1      0x0      0x0
98                 pim_activate           1   0   1   2      0x0      0x0
98                 pim_activate           1   0   1   3      0x0      0x0
128                pim_activate           1   0   2   0      0x0      0x0
128                pim_activate           1   0   2   1      0x0      0x0
128                pim_activate           1   0   2   2      0x0      0x0
128                pim_activate           1   0   2   3      0x0      0x0
158                pim_activate           1   0   3   0      0x0      0x0
158                pim_activate           1   0   3   1      0x0      0x0
158                pim_activate           1   0   3   2      0x0      0x0
158                pim_activate           1   0   3   3      0x0      0x0
172                pim_read               1   0   0   0      0x0      0x0
172                pim_read               1   0   0   1      0x0      0x0
172                pim_read               1   0   0   2      0x0      0x0
172                pim_read               1   0   0   3      0x0      0x0
172                pim_read               1   0   1   0      0x0      0x0
172                pim_read               1   0   1   1      0x0      0x0
172                pim_read               1   0   1   2      0x0      0x0
172                pim_read               1   0   1   3      0x0      0x0
172                pim_read               1   0   2   0      0x0      0x0
172                pim_read               1   0   2   1      0x0      0x0
172                pim_read               1   0   2   2      0x0      0x0
172                pim_read               1   0   2   3      0x0      0x0
172                pim_read               1   0   3   0      0x0      0x0
172                pim_read               1   0   3   1      0x0      0x0
172                pim_read               1   0   3   2      0x0      0x0
172                pim_read               1   0   3   3      0x0      0x0
174                pim_read               1   0   0   0      0x0      0x1
174                pim_read               1   0   0   1      0x0      0x1
174                pim_read               1   0   0   2      0x0      0x1
174                pim_read               1   0   0   3      0x0      0x1
174                pim_read               1   0   1   0      0x0      0x1
174                pim_read               1   0   1   1      0x0      0x1
174                pim_read               1   0   1   2      0x0      0x1
174                pim_read               1   0   1   3      0x0      0x1
174                pim_read               1   0   2   0      0x0      0x1
174                pim_read               1   0   2   1      0x0      0x1
174                pim_read               1   0   2   2      0x0      0x1
174                pim_read               1   0   2   3      0x0      0x1
174                pim_read               1   0   3   0      0x0      0x1
174                pim_read               1   0   3   1      0x0      0x1
174                pim_read               1   0   3   2      0x0      0x1
174                pim_read               1   0   3   3      0x0      0x1
```
The command statistics shows the number of pim commands, their row hits, and energy, etc.
```bash
num_pim_read_row_hits          =        10094   # Number of read row buffer hits
...
num_pim_read_cmds              =        10272   # Number of READ/READP commands
...
pim_read_energy                =  1.08472e+06   # Read energy
```
### Running a custom workload
You can generate the LLM workload and traces by running our custom python programs, requiring following python packages to be installed:
```
pip install configparser
pip install openpyxl
```
Firstly, you should make workload file of matmul kernel. 

Example of 128x256x1 GEMV with (mcf,ucf) = (2,8) :
```bash
1, 1, 2048, 8, 2, 8, 1 
128, 256, 1
```
**First line format**: ```1 (fixed), 1 (fixed), Accumulation_buffer_depth, 8 (fixed), mcf, ucf, dataflow_option``` 

Fixed values are reserved for future works that are under the development.

dataflow option is 0 for Weight stationary GEMM kernel, 1 for Input stationary GEMV kernel.

With GEMM kernel workload, you can use the mcf value as the number of multi-columns and should fix ucf as 1.

For example, 128x256x512 GEMM kernel workload with 4 multicolumns is:
```bash
1, 1, 2048, 8, 4, 1, 0
128, 256, 512
```


**Second line format**: ```M, K, N``` for MxKxN GEMM kernel; ```M, K, 1``` for GEMV kernel of MxK matrix and Kx1 vector 

With the workload file you can generate transaction traces that will go into DRAMsim program by running below command.
It divides and reshapes the operand data and allocates them to the physical address of the bank.
Then you can now run HB-NPU with the generated trace file like the above.
```bash
python gen_pim_trace2.py -f False -w [workload_path]
./build/dramsim3main configs/HBM2_8Gb_x128.ini -c 5000000 -t sample.trc
```
Each trace line in a trace file is composed of memory address, transaction type, and cycle time to execute the transaction.
```bash
0x2f0060	PIM	0 # set variables such as mcf, ucf, dataflow option, etc.
0xc8800000a000	PIM	1 # set dimension M and physical (starting) address of weight
0x5020	PIM	2 # set dimension K and physical (starting) address of output
0x840	PIM	3 # set dimension N and physical (starting) address of input
0x3	PIM	4 # launch computation
```
Unlike the normal DRAM access transactions, PIM transactions are not directly translated to the DRAM commands because the management of transactions in DRAMsim3 does not fit into HB-NPU control system.
Instead, our transaction address contains the information about the workload to run, physical addresses of matrices, and dataflow configuration, etc and commands are dynamically generated by PIM command scheduler we have implemented.
We elaborated the simulator design in later section.


## Simulator Design

It assumes _one transaction accesses only one channel and one bank_, just as noraml DRAM access but HB-NPU accesses several banks in several channel simultaneously.

PIM command scheduler then stores these informations in registers and generates PIM commands dynamically based on the registers.
This control scheme is possible because matrix multiplication has regular access pattern unlike the normal DRAM access.

The transaction in first line 

Each channel controller has its own transaction queue and individually manages and generates the commands. (refer to the workflow diagram below.)
HB-NPU accesses several banks in several channel simultaneously and needs . 

BLSA command generating
DRAM command added -> different timing constraints, different power calculation -> report
refreshing

The other challenge is that DRAM controllers manage the command queue in dynamic way, so the commands can be executed out of order for higher command throughput, which is not problematic in normal access but dependency in PNM operations is not considered.
Each transaction 
this trace format hinders dynamic execution of memory transactions since the execution timing is deterministic.
Especially in Originally in DRAMsim3, 

<img src="https://github.com/shh8228/HBDRAMsim/blob/master/workflow.jpg" alt="Example Image" width="507" height="643" />

### Code Structure

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
    bankstate.cc: Records and manages DRAM bank timings and states which is modeled as a state machine.
    channelstate.cc: Records and manages channel timings and states.
    command_queue.cc: Maintains per-bank or per-rank FIFO queueing structures, determine which commands in the queues can be issued in this cycle.
    configuration.cc: Initiates, manages system and DRAM parameters, including protocol, DRAM timings, address mapping policy and power parameters.
    controller.cc: Maintains the per-channel controller, which manages a queue of pending memory transactions and issues corresponding DRAM commands, 
                   follows FR-FCFS policy.
    cpu.cc: Implements 3 types of simple CPU: 
            1. Random, can handle random CPU requests at full speed, the entire parallelism of DRAM protocol can be exploited without limits from address mapping and scheduling pocilies. 
            2. Stream, provides a streaming prototype that is able to provide enough buffer hits.
            3. Trace-based, consumes traces of workloads, feed the fetched transactions into the memory system.
    dram_system.cc:  Initiates JEDEC or ideal DRAM system, registers the supplied callback function to let the front end driver know that the request is finished. 
    hmc.cc: Implements HMC system and interface, HMC requests are translates to DRAM requests here and a crossbar interconnect between the high-speed links and the memory controllers is modeled.
    main.cc: Handles the main program loop that reads in simulation arguments, DRAM configurations and tick cycle forward.
    memory_system.cc: A wrapper of dram_system and hmc.
    refresh.cc: Raises refresh request based on per-rank refresh or per-bank refresh.
    timing.cc: Initiate timing constraints.
```

## Experiments

### Verilog Validation

First we generate a DRAM command trace.
There is a `CMD_TRACE` macro and by default it's disabled.
Use `cmake .. -DCMD_TRACE=1` to enable the command trace output build and then
whenever a simulation is performed the command trace file will be generated.

Next, `scripts/validation.py` helps generate a Verilog workbench for Micron's Verilog model
from the command trace file.
Currently DDR3, DDR4, and LPDDR configs are supported by this script.

Run

```bash
./script/validataion.py DDR4.ini cmd.trace
```

To generage Verilog workbench.
Our workbench format is compatible with ModelSim Verilog simulator,
other Verilog simulators may require a slightly different format.


## Related Work

[1] Li, S., Yang, Z., Reddy D., Srivastava, A. and Jacob, B., (2020) DRAMsim3: a Cycle-accurate, Thermal-Capable DRAM Simulator, IEEE Computer Architecture Letters.

[2] Jagasivamani, M., Walden, C., Singh, D., Kang, L., Li, S., Asnaashari, M., ... & Yeung, D. (2019). Analyzing the Monolithic Integration of a ReRAM-Based Main Memory Into a CPU's Die. IEEE Micro, 39(6), 64-72.

[3] Li, S., Reddy, D., & Jacob, B. (2018, October). A performance & power comparison of modern high-speed DRAM architectures. In Proceedings of the International Symposium on Memory Systems (pp. 341-353).

[4] Li, S., Verdejo, R. S., Radojković, P., & Jacob, B. (2019, September). Rethinking cycle accurate DRAM simulation. In Proceedings of the International Symposium on Memory Systems (pp. 184-191).

[5] Li, S., & Jacob, B. (2019, September). Statistical DRAM modeling. In Proceedings of the International Symposium on Memory Systems (pp. 521-530).

[6] Li, S. (2019). Scalable and Accurate Memory System Simulation (Doctoral dissertation).

