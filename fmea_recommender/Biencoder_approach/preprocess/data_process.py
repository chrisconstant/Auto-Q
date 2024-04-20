import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="data process")
    parser.add_argument(
        "--in_seq_data", type=str, help=""
    )
    parser.add_argument(
        "--in_meta_data", type=str, help=""
    )    
    parser.add_argument(
        "--out_directory", type=str, help=""
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print("Output directory path:", args.out_directory)

    try:
        if not os.path.exists(args.out_directory):
            os.makedirs(args.out_directory)
    except FileNotFoundError as e:
        print("Error:", e)

