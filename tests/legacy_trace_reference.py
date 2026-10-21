"""Historical transaction encoder, used only as an independent test oracle."""
import ast
import math

def gen_pim_trace(workload, trace_file, row_addr):
    # Single-array format: one dataflow header and one (M, K, N) tuple.
    with open(workload, 'r') as fin:
        header = ast.literal_eval(fin.readline())
        if not isinstance(header, (tuple, list)) or len(header) != 5:
            raise ValueError("Expected five-field header: tile_M, post_delay, "
                             "mcf, ucf, df; regenerate legacy workloads")
        tile_M, post_delay, mcf, ucf, df = header
        m, k, n = ast.literal_eval(fin.readline())
        if fin.read().strip():
            raise ValueError("Expected exactly one (M, K, N) workload")

    # Bits 0..2 identify the transaction; remaining fields are contiguous.
    config_addr = (3 << 1)
    config_addr |= int(math.log(mcf, 2)) << 3
    config_addr |= int(math.log(ucf, 2)) << 6
    config_addr |= df << 9
    config_addr |= int(math.log(tile_M, 2)) << 10

    dims = (m, k, n)
    loadaddrs = []
    for j, dim in enumerate(dims):
        if df == 0 and j == 0:
            dim = int(dim/2)  # GEMM interleaving
        if df == 1:
            if j == 0:
                dim = max(1, int(dim/mcf))
            elif j == 1:
                dim = max(1, int(dim/ucf))
            else:
                dim *= ucf
        base_row = 0
        if df == 0 and j != 0:  # Inputs/outputs in GEMM
            base_row = int(k*n/1024)+1 if row_addr == -1 else int(row_addr)
        elif df == 1 and j != 2:  # Inputs/outputs in GEMV
            base_row = int(k*m/1024)+1 if row_addr == -1 else int(row_addr)
        loadaddrs.append((j << 1) | (dim << 3) | (base_row << 35))

    with open(trace_file, 'w') as fout:
        for cycle, addr in enumerate([config_addr] + loadaddrs + [0x1]):
            fout.write(hex(addr) + '\tPIM\t' + str(cycle) + '\n')
