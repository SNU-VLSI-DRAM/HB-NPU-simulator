import os
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("-s", default=1024, help="sequence length")

def gen_actual_workload_ws(folder, name, M_tile, al, m, k, n, mc):
    if n < mc:
        return
    fout = open(folder + '/' + name, 'w')
    fout.write(", ".join(['1', '1', str(M_tile), str(al), str(mc), str(1), '0']) + '\n')
    fout.write(", ".join([str(m), str(k), str(n)]) + '\n')
    fout.close()


def gen_workload(n_heads, d_head, d_model, seq_len, model, p):
    M_tile = 2048
    al = 8

    folder = "workloads/" + model + "/prompt/" + format(seq_len, '04')
    if not os.path.exists(folder):
        os.makedirs(folder)


    # Create Q, K, V : seq_len x d_model x d_head*n_heads/p
    gen_actual_workload_ws(folder, "createQKV", M_tile, al, seq_len, d_model, d_head*int(math.ceil(float(n_heads)/p)), 2)

    # W_o : seq_len x d_head*n_heads/p x d_model
    gen_actual_workload_ws(folder, "Wo", M_tile, al, seq_len, d_head*int(math.ceil(float(n_heads)/p)), d_model, 2)

    # Linear 1 : seq_len x d_model x 4*d_model/p
    gen_actual_workload_ws(folder, "L1", M_tile, al, seq_len, d_model, int(4*d_model/p), 2)

    # Linear 2 : seq_len x 4*d_model/p x d_model
    gen_actual_workload_ws(folder, "L2", M_tile, al, seq_len, int(4*d_model/p), d_model, 2)

    # Q x K_T : seq_len x d_head x seq_len
    gen_actual_workload_ws(folder, "QK", M_tile, al, seq_len, d_head, seq_len, 2)

    # S x V : seq_len x seq_len x d_head
    gen_actual_workload_ws(folder, "SV", M_tile, al, seq_len, seq_len, d_head, 2)

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
        gen_workload(int(n_heads), int(d_head), int(d_model), int(args.s), model, int(par))
