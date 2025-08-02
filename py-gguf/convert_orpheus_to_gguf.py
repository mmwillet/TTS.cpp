#!/usr/bin/env python3

import argparse
from tts_encoders.orpheus_gguf_encoder import OrpheusEncoder, DEFAULT_ORPHEUS_REPO_ID, DEFAULT_SNAC_REPO_ID
from os.path import isdir, dirname


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="the path to save the converted gguf tts model too.")
    parser.add_argument("--repo-id", type=str, required=False, default=DEFAULT_ORPHEUS_REPO_ID, help="The Huggingface repository to pull the model from.")
    parser.add_argument("--snac-repo-id", type=str, required=False, default=DEFAULT_SNAC_REPO_ID, help="The Huggingface repository to pull the snac audio decoder model from.")
    parser.add_argument("--never-make-dirs", default=False, action="store_true", help="When set the script will never add new directories.")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_arguments()
    if not isdir(dirname(args.save_path)) and args.never_make_dirs:
        raise ValueError(f"model path, {args.save_path} is not a valid path.")
    OrpheusEncoder(args.save_path, repo_id=args.repo_id).write()
