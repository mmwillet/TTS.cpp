#!/usr/bin/env python3

import argparse
from tts_encoders import DiaEncoder, DEFAULT_DIA_REPO_ID
from os.path import isdir, dirname


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="the path to save the converted gguf tts model too.")
    parser.add_argument("--repo-id", type=str, required=False, default=DEFAULT_DIA_REPO_ID, help="A custom Huggingface repository to pull the model from.")
    parser.add_argument("--never-make-dirs", default=False, action="store_true", help="When set the script will never add new directories.")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_arguments()
    if not isdir(dirname(args.save_path)) and args.never_make_dirs:
        raise ValueError(f"model path, {args.save_path} is not a valid path.")
    DiaEncoder(args.save_path, repo_id=args.repo_id).write()
