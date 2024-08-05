import os
import argparse
import math
import configparser

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
    fin = open(workload, 'r')
    fout = open(trace_file, 'w')
    line = fin.readline()
    cutV, cutH, tile_M, post_delay, mcf, ucf, df = eval(line)
    # print(cutV, cutH, tile_M, post_delay)

    exp = 5

    cutaddr = 3 * (2**exp)
    exp += 2
    cutaddr += math.log(cutV, 2) * (2**exp)
    exp += 3
    cutaddr += math.log(cutH, 2) * (2**exp)
    exp += 1
    cutaddr += math.log(mcf, 2) * (2**exp)
    exp += 3
    cutaddr += math.log(ucf, 2) * (2**exp)
    exp += 3

    cutaddr += df * (2**exp)
    exp += 1

    cutaddr += math.log(tile_M, 2) * (2**exp)

    loadaddrs = []

    cutSize = cutV*cutH
    for i in range(cutV*cutH):
        line = fin.readline()
        dims = eval(line)
        # print(dims)
        for (j, dim) in enumerate(dims):
            # print(j, dim)
            exp = 1
            loadaddr = i * (2**exp)
            exp += 4
            loadaddr += j * (2**exp)
            exp += 2
            if (df == 0 and j==0):
                dim = int(dim/2) # GEMM interleaving
            if (df == 1): # N[i] == 1 (mcf*ucf == 16)
                if (j == 0):
                    loadaddr += max(1, int(dim/mcf)) * (2**exp)
                elif (j == 1):
                    loadaddr += max(1, int(dim/ucf)) * (2**exp)
                else:
                    loadaddr += dim*ucf * (2**exp)
            else:
                loadaddr += dim * (2**exp)
            exp += 32
            if (df == 0 and j!=0): # Allocating addresses of Inputs and outputs in GEMM kernel
                loadaddr += ((int(dims[1]*dims[2]/1024)+1) if row_addr == -1 else  int(row_addr)) * (2**exp)
            elif (df == 1 and j!=2): # Allocating addresses of Inputs and outputs in GEMV kernel
                loadaddr += ((int(dims[1]*dims[0]/1024)+1) if row_addr == -1 else  int(row_addr)) * (2**exp)
            else: # Allocating address of Weights to 0
                loadaddr += 0
            loadaddrs.append(loadaddr)
    exp = 0
    compaddr = 1
    exp += 1
    compaddr += (2**(cutV*cutH)-1) * (2**exp)

    fin.close()

    cycle = 0
    fout.write(hex(int(cutaddr)) + '\tPIM\t' + str(cycle) + '\n')
    cycle += 1
    for addr in loadaddrs:
        fout.write(hex(int(addr)) + '\tPIM\t' + str(cycle) + '\n')
        cycle += 1
    fout.write(hex(int(compaddr)) + '\tPIM\t' + str(cycle) + '\n')

    fout.close()
    return


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
