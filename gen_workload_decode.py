import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-s", default=2, help="start sequence number")
parser.add_argument("-e", default=1600, help="end sequence number")
parser.add_argument("-sb", default=8, help="start batch size (e.g. 8 -> 8, 16, 32, ...)")
parser.add_argument("-eb", default=128, help="end batch size (e.g. 128 -> ...32, 64, 128)")


def gen_actual_workload(folder, name, M_tile, al, m, k, n, mcf, ucf):
    fout = open(folder + '/' + name, 'w')
    fout.write(", ".join(['1', '1', str(M_tile), str(al), str(mcf), str(ucf), '1']) + '\n')
    fout.write(", ".join([str(m), str(k), str(n)]) + '\n')
    fout.close()

def gen_actual_workload_ws(folder, name, M_tile, al, m, k, n, mc):
    if n < mc:
        return
    fout = open(folder + '/' + name, 'w')
    fout.write(", ".join(['1', '1', str(M_tile), str(al), str(mc), str(1), '0']) + '\n')
    fout.write(", ".join([str(m), str(k), str(n)]) + '\n')
    fout.close()

def get_optimum_tiling(n, k, mcfs):
    row_switching = []
    for mcf in mcfs:
        ucf = int(16/mcf)
        x_step = ucf*128
        row_switching.append(int(math.ceil(float(k)/x_step)) * int(math.ceil(float(n)/mcf/64)))
    return mcfs[row_switching.index(min(row_switching))]



def gen_workload(n_heads, d_head, d_model, start, end, model, p, batch_sizes):
    M_tile = 2048
    al = 8

    folder = "workloads/" + model + "/decode"
    if not os.path.exists(folder):
        os.makedirs(folder)

    folder_WS = folder + "/WS"
    if not os.path.exists(folder_WS):
        os.makedirs(folder_WS)

    folder_QKV = folder + "/QKV"
    if not os.path.exists(folder_QKV):
        os.makedirs(folder_QKV)

    # Create Q, K, V : 1 x d_model x d_head*n_heads/p
    # IS-All-bank GEMM dataflow. The workload of batch size 8 or less is generated. Larger ones can be calculated by simply multiplying the result of batch size 8.
    mcf = get_optimum_tiling(d_head*int(math.ceil(float(n_heads)/p)), d_model, [2, 4, 8, 16]) # mcf 1 is not compatible with bank interleaving of WS GEMM dataflow
    gen_actual_workload(folder, "createQKV", M_tile, al, d_head*int(math.ceil(float(n_heads)/p)), d_model, 8, mcf, int(16/mcf))

    # W_o : 1 x d_head*n_heads/p x d_model
    mcf = get_optimum_tiling(d_model, d_head*int(math.ceil(float(n_heads)/p)), [2, 4, 8, 16])
    gen_actual_workload(folder, "Wo", M_tile, al, d_model, d_head*int(math.ceil(float(n_heads)/p)), 8, mcf, int(16/mcf))

    # Linear 1 : 1 x d_model x 4*d_model/p
    mcf = get_optimum_tiling(4*d_model/p, d_model, [2, 4, 8, 16])
    gen_actual_workload(folder, "L1", M_tile, al, int(4*d_model/p), d_model, 8, mcf, int(16/mcf)) # TODO not headwise

    # Linear 2 : 1 x 4*d_model/p x d_model
    mcf = get_optimum_tiling(d_model, 4*d_model/p, [2, 4, 8, 16])
    gen_actual_workload(folder, "L2", M_tile, al, d_model, int(4*d_model/p), 8, mcf, int(16/mcf)) # TODO not headwise

    # WS GEMM dataflow
    for bs in batch_sizes:
        gen_actual_workload_ws(folder_WS, format(bs, '04') + "_createQKV", M_tile, al, bs, d_model, d_head*int(math.ceil(float(n_heads)/p)), 2)
        gen_actual_workload_ws(folder_WS, format(bs, '04') + "_Wo", M_tile, al, bs, d_head*int(math.ceil(float(n_heads)/p)), d_model, 2)
        gen_actual_workload_ws(folder_WS, format(bs, '04') + "_L1", M_tile, al, bs, d_model, int(4*d_model/p), 2)
        gen_actual_workload_ws(folder_WS, format(bs, '04') + "_L2", M_tile, al, bs, int(4*d_model/p), d_model, 2)

    for i in range(start, end+1):
        # Q x K_T : 1 x d_head x seq_num
        if i <= 16:
            m = max(2, i)
            # mcf = 1
        else:
            m = i
            # mcf_ = mcf

        assert d_head <= 128
        # We assume all 16 subarrays are occupied by different batches or heads when multiplying 16 to d_head,
        # but there are some cases 16 subarrays are not fully operated when batch size is small and TP is too high.
        # However the difference in cycle time is negligible since these subarrays are running in parallel.
        # Instead we apply larger multiplier in under-utilized cases.
        gen_actual_workload(folder_QKV, '_'.join(["QK", format(i, '04')]), M_tile, al, m, d_head*16, 1, 1, 16) # TODO *16 for subarrays; 16/mcf -> 1?
        # S x V : 1 x seq_num x d_head
        gen_actual_workload(folder_QKV, '_'.join(["SV", format(i, '04')]), M_tile, al, d_head*16, i, 1, 16, 1) # TODO same

    return


if __name__ == "__main__":
    args = parser.parse_args()
    fin = open('models', 'r')
    lines = fin.readlines()
    for line in lines:
        line = line.strip()
        model, param_size, n_layers, d_model, n_heads, d_head, par, PP = line.split(' ')
        print("Generating model " + model)
        print("TP: " + par + ", PP: " + PP)
        batch_sizes = []
        current = int(args.sb)
        while current <= int(args.eb):
            batch_sizes.append(int(current/int(PP)))
            current *= 2
        gen_workload(int(n_heads), int(d_head), int(d_model), int(args.s), int(args.e), model, int(par), batch_sizes)
