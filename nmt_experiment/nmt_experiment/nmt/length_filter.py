import argparse

from argparse import Namespace


def length_filter(args: Namespace):
    min_max_filter_count = 0
    ratio_filter_count = 0
    with open(args.src_path, "r", encoding="utf-8") as src_in, \
        open(args.tgt_path, "r", encoding="utf-8") as tgt_in, \
            open(args.output_src_path, "w", encoding="utf-8") as src_out, \
                open(args.output_tgt_path, "w", encoding="utf-8") as tgt_out:
                    for src_line, tgt_line in zip(src_in, tgt_in):
                        src_line = src_line.strip()
                        tgt_line = tgt_line.strip()

                        src_length = len(src_line.split())
                        tgt_length = len(tgt_line.split())

                        if not (args.min_src_length <= src_length <= args.max_src_length):
                            min_max_filter_count += 1
                            continue

                        if not (args.min_tgt_length <= tgt_length <= args.max_tgt_length):
                            min_max_filter_count += 1
                            continue

                        if src_length / tgt_length > args.max_length_ratio or tgt_length / src_length > args.max_length_ratio:
                            ratio_filter_count += 1
                            continue
                        
                        src_out.write(src_line + "\n")
                        tgt_out.write(tgt_line + "\n")

    print(f"min_max_filter_count: {min_max_filter_count}")
    print(f"ratio_filter_count: {ratio_filter_count}")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", required=True, type=str)
    parser.add_argument("--tgt-path", required=True, type=str)

    parser.add_argument("--min-src-length", default=1, type=int)
    parser.add_argument("--max-src-length", default=1000, type=int)
    
    parser.add_argument("--min-tgt-length", default=1, type=int)
    parser.add_argument("--max-tgt-length", default=1000, type=int)
    
    parser.add_argument("--max-length-ratio", default=3.0, type=float)

    parser.add_argument("--output-src-path", required=True, type=str)
    parser.add_argument("--output-tgt-path", required=True, type=str)

    args = parser.parse_args()
    length_filter(args)


if __name__ == "__main__":
    cli_main()
