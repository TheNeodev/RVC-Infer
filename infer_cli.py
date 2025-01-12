import argparse
import os
import sys
from dotenv import load_dotenv
from scipy.io import wavfile

from config import Config
from modules import VC


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--f0up_key", type=int, default=0)
    parser.add_argument("--input_path", type=str, help="input path", required=True)
    parser.add_argument("--index_path", type=str, help="index path")
    parser.add_argument("--f0method", type=str, default="harvest", help="harvest or pm")
    parser.add_argument("--opt_path", type=str, help="output path", required=True)
    parser.add_argument("--model_name", type=str, help="model name (stored in assets/weight_root)", required=True)
    parser.add_argument("--index_rate", type=float, default=0.66, help="index rate")
    parser.add_argument("--device", type=str, help="device (e.g., cuda or cpu)")
    parser.add_argument("--is_half", type=bool, help="use half precision (True or False)", default=False)
    parser.add_argument("--filter_radius", type=int, default=3, help="filter radius")
    parser.add_argument("--resample_sr", type=int, default=0, help="resample sampling rate")
    parser.add_argument("--rms_mix_rate", type=float, default=1, help="RMS mix rate")
    parser.add_argument("--protect", type=float, default=0.33, help="protect value")

    args = parser.parse_args()
    return args


def main():
    load_dotenv()
    args = arg_parse()

    #config = Config()
    #config.device = args.device if args.device else config.device
    #config.is_half = args.is_half if args.is_half else config.is_half

    vc = VC()

    # Add a check for the model name
    if not args.model_name:
        print("Error: Model name must be provided.")
        sys.exit(1)

    vc.get_vc(args.model_name)

    # Process the audio file
    try:
        sid, wav_opt = vc.vc_single(
            0,
            args.input_path,
            args.f0up_key,
            None,
            args.f0method,
            args.index_path,
            None,
            args.index_rate,
            args.filter_radius,
            args.resample_sr,
            args.rms_mix_rate,
            args.protect,
        )

        if sid is None:
            print("Warning: sid is None. Skipping sid-related operations.")
        else:
            print(f"Processed sid: {sid}")

        # Save the processed audio
        wavfile.write(args.opt_path, wav_opt[0], wav_opt[1])
        print(f"Output saved to: {args.opt_path}")

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
