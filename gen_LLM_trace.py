import os
import argparse
import math
import configparser
import ast

parser = argparse.ArgumentParser()
parser.add_argument("-w", default=".", help="file mode: workload file path")
parser.add_argument("-t", default="./sample.trc", help="file mode: trace file path")
parser.add_argument("-f", default="True", help="Model mode? Set False for file mode (give only -w, -t options)")
parser.add_argument("-m", default="OPT-2.7B", help="Model mode: model configuration to run")
parser.add_argument("-i", default=1024, help="Model mode: input tokens")
parser.add_argument("-o", default=128, help="Model mode: output tokens")
parser.add_argument("-b", default=64, help="Model mode: batch size")
parser.add_argument("-s", default="configs/HBM2_8Gb_x128.ini", help="DRAM specification file to read")


def extract_variables(file_path):
    desired_variables = {
        "dram_structure": ["protocol", "bankgroups", "banks_per_group", "rows", "columns", "device_width", "BL", "num_dies"],
        "timing": ["tCK", "CL", "CWL", "tRCDRD", "tRCDWR", "tRP", "tRAS", "tRFC", "tREFI"],
        "power": ["VDD", "IDD0", "IDD2P", "IDD2N", "IDD3P", "IDD3N", "IDD4W", "IDD4R", "IDD5AB", "IDD6x"],
        "system": ["channel_size", "channels", "bus_width", "address_mapping", "queue_structure", "row_buf_policy", "cmd_queue_size", "trans_queue_size", "unified_queue"]
    }

    config = configparser.ConfigParser()
    config.read(file_path)

    extracted_data = {}

    for section, variables in desired_variables.items():
        if section in config:
            extracted_data[section] = {}
            for var in variables:
                if var in config[section]:
                        extracted_data[section][var] = config[section][var]

    return extracted_data



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


if __name__ == "__main__":
    args = parser.parse_args()
    if eval(args.f):
        model_name, in_tokens, out_tokens, batch_size, spec_path = args.m, int(args.i), int(args.o), int(args.b),  args.s

        fin_ = open('models', 'r')
        lines = fin_.readlines()
        lines_parsed = [line.strip().split(' ') for line in lines if line.strip().split(' ')[0] == model_name]
        if not lines_parsed:
            raise Exception("Model Not Found!")
        else:
            _, params, n_layers, d_model, n_heads, d_head, TP, PP = lines_parsed[0]

        KV_cache = 2*(in_tokens+out_tokens)*int(d_model)*2*int(n_layers)*batch_size/(10**9)
        required_capacity = 2*float(params) + KV_cache
        total_capacity = int(TP) * int(PP) * 2
        if required_capacity  > total_capacity:
            raise Exception("Model does not fit in DRAM! " + str(math.ceil(required_capacity)) + "GB > " + str(total_capacity) + "GB")


        # weights and KV cache is evenly distributed between banks
        spec = extract_variables(args.s)
        banks = int(spec["system"]["channels"])*int(spec["dram_structure"]["bankgroups"])*int(spec["dram_structure"]["banks_per_group"])
        rows = int(spec["dram_structure"]["rows"])
        weight_rows_per_bank = int(required_capacity / int(TP) / int(PP) / banks / 2 * 10**6) #TODO assuming 2 KB row size


        trace_path = "traces/" + "_".join([model_name, str(in_tokens), str(out_tokens), str(batch_size)])
        print(model_name + " trace is generated in " + trace_path)

        prompt_path = "workloads/" + model_name + "/prompt/" + format(in_tokens, '04')
        if not os.path.exists(prompt_path):
            raise Exception("Prompt workloads do not exist!")

        flist = os.listdir(prompt_path)
        if not os.path.exists(trace_path + "/prompt"):
            os.makedirs(trace_path + "/prompt")
        for f in flist:
            gen_pim_trace(prompt_path + '/' + f, trace_path + "/prompt/" +  f, weight_rows_per_bank)

        decode_path = "workloads/" + model_name + "/decode/"
        QKV_path = decode_path + "QKV/"
        WS_path = decode_path + "WS/"

        flist = os.listdir(decode_path)
        if not os.path.exists(trace_path + "/decode"):
            os.makedirs(trace_path + "/decode")
        for f in flist:
            if not os.path.isdir(decode_path + f):
                gen_pim_trace(decode_path + f, trace_path + "/decode/" +  f, weight_rows_per_bank)

        if not os.path.exists(trace_path + "/decode/QKV/"):
            os.makedirs(trace_path + "/decode/QKV")
        for seq in range(in_tokens, in_tokens + out_tokens):
            QK = "QK_" + format(seq, '04')
            SV = "SV_" + format(seq, '04')
            if not os.path.exists(QKV_path + QK):
                raise Exception("Workload " + QK + " does not exist!")
            if not os.path.exists(QKV_path + SV):
                raise Exception("Workload " + SV + " does not exist!")
            gen_pim_trace(QKV_path + QK, trace_path + "/decode/QKV/" + QK, weight_rows_per_bank)
            gen_pim_trace(QKV_path + SV, trace_path + "/decode/QKV/" + SV, weight_rows_per_bank)

        flist = os.listdir(WS_path)
        if format(int(batch_size/int(PP)), '04') not in [f.split('_')[0] for f in flist]:
            raise Exception("No available workload for given batch size!")
        if not os.path.exists(trace_path + "/decode/WS"):
            os.makedirs(trace_path + "/decode/WS")
        for f in [f for f in flist if int(batch_size/int(PP)) == int(f.split('_')[0])]:
            gen_pim_trace(WS_path + f, trace_path + "/decode/WS/" + f.split('_')[1], weight_rows_per_bank)


    else:
        gen_pim_trace(args.w, args.t, -1)
