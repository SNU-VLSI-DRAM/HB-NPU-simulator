"""Compile one existing workload to an HB-NPU instruction trace."""
import argparse
from hbnpu_instruction import load_workload, compile_kernel, write_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-w", required=True, help="source workload")
    parser.add_argument("-s", required=True, help="DRAM INI configuration")
    parser.add_argument("-t", required=True, help="destination instruction trace")
    parser.add_argument("--row-addr", type=int, default=-1)
    parser.add_argument("--start-cycle", type=int, default=0)
    parser.add_argument("--word-period", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        context = load_workload(args.w, args.s, args.row_addr)
        write_trace(
            compile_kernel(context),
            args.t,
            args.start_cycle,
            args.word_period,
            overwrite=args.overwrite,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
