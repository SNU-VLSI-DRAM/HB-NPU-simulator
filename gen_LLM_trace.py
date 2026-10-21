import os
import argparse
import math
import configparser
from hbnpu_instruction import load_workload, compile_kernel, write_trace

parser = argparse.ArgumentParser()
parser.add_argument("-p", default="output/traces", help="Model mode: generated instruction trace root")
parser.add_argument("--overwrite", action="store_true", help="explicitly replace generated trace files")
parser.add_argument("-w", default=".", help="file mode: workload file path")
parser.add_argument("-t", default="./sample.trc", help="file mode: trace file path")
parser.add_argument("-f", default="true", type=str.lower, choices=("true", "false"),
                    help="Model mode? Set False for file mode (give -w, -t options)")
parser.add_argument("-m", default="OPT-2.7B", help="Model mode: model configuration to run")
parser.add_argument("-i", default=1024, help="Model mode: input tokens")
parser.add_argument("-o", default=128, help="Model mode: output tokens")
parser.add_argument("-b", default=64, help="Model mode: batch size")
parser.add_argument("-s", default="configs/HBNPU_8Gb.ini", help="DRAM specification file to read")


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



def gen_pim_trace(workload, trace_file, row_addr, spec="configs/HBNPU_8Gb.ini", *, overwrite=False):
    context = load_workload(workload, spec, row_addr)
    # Validate the complete program before opening an existing destination.
    packets = list(compile_kernel(context))
    write_trace(packets, trace_file, overwrite=overwrite)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.f.lower() == "true":
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


        trace_path = os.path.join(args.p, "_".join([model_name, str(in_tokens), str(out_tokens), str(batch_size)]))
        print(model_name + " trace is generated in " + trace_path)

        prompt_path = "workloads/" + model_name + "/prompt/" + format(in_tokens, '04')
        if not os.path.exists(prompt_path):
            raise Exception("Prompt workloads do not exist!")

        flist = os.listdir(prompt_path)
        if not os.path.exists(trace_path + "/prompt"):
            os.makedirs(trace_path + "/prompt")
        for f in flist:
            gen_pim_trace(prompt_path + '/' + f, trace_path + "/prompt/" +  f, weight_rows_per_bank, args.s, overwrite=args.overwrite)

        decode_path = "workloads/" + model_name + "/decode/"
        QKV_path = decode_path + "QKV/"
        WS_path = decode_path + "WS/"

        flist = os.listdir(decode_path)
        if not os.path.exists(trace_path + "/decode"):
            os.makedirs(trace_path + "/decode")
        for f in flist:
            if not os.path.isdir(decode_path + f):
                gen_pim_trace(decode_path + f, trace_path + "/decode/" +  f, weight_rows_per_bank, args.s, overwrite=args.overwrite)

        if not os.path.exists(trace_path + "/decode/QKV/"):
            os.makedirs(trace_path + "/decode/QKV")
        for seq in range(in_tokens, in_tokens + out_tokens):
            QK = "QK_" + format(seq, '04')
            SV = "SV_" + format(seq, '04')
            if not os.path.exists(QKV_path + QK):
                raise Exception("Workload " + QK + " does not exist!")
            if not os.path.exists(QKV_path + SV):
                raise Exception("Workload " + SV + " does not exist!")
            gen_pim_trace(QKV_path + QK, trace_path + "/decode/QKV/" + QK, weight_rows_per_bank, args.s, overwrite=args.overwrite)
            gen_pim_trace(QKV_path + SV, trace_path + "/decode/QKV/" + SV, weight_rows_per_bank, args.s, overwrite=args.overwrite)

        flist = os.listdir(WS_path)
        if format(int(batch_size/int(PP)), '04') not in [f.split('_')[0] for f in flist]:
            raise Exception("No available workload for given batch size!")
        if not os.path.exists(trace_path + "/decode/WS"):
            os.makedirs(trace_path + "/decode/WS")
        for f in [f for f in flist if int(batch_size/int(PP)) == int(f.split('_')[0])]:
            gen_pim_trace(WS_path + f, trace_path + "/decode/WS/" + f.split('_')[1], weight_rows_per_bank, args.s, overwrite=args.overwrite)


    else:
        gen_pim_trace(args.w, args.t, -1, args.s, overwrite=args.overwrite)
