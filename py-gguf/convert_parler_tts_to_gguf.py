#!/usr/bin/env python3

import argparse
from tts_encoders import ParlerTTSEncoder, DEFAULT_PARLER_REPO_LARGE_ID, DEFAULT_PARLER_REPO_MINI_ID
from os.path import isdir, dirname


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="the path to save the converted gguf tts model too.")
    parser.add_argument("--voice-prompt", type=str, default="female voice", help="A description of the voice that the model should generate with.")
    parser.add_argument("--large-model", default=False, action='store_true', help="Whether to encode the large version of Parler TTS.")
    parser.add_argument("--repo-id-override", type=str, required=False, help="A custom Huggingface repository to pull the model from.")
    parser.add_argument("--never-make-dirs", default=False, action="store_true", help="When set the script will never add new directories.")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_arguments()
    if not isdir(dirname(args.save_path)) and args.never_make_dirs:
        raise ValueError(f"model path, {args.save_path} is not a valid path.")
    repo_id = DEFAULT_PARLER_REPO_LARGE_ID if args.large_model else DEFAULT_PARLER_REPO_MINI_ID
    if args.repo_id_override:
        repo_id = args.repo_id_override
    ParlerTTSEncoder(args.save_path, repo_id=repo_id, text_encoding_prompt=args.voice_prompt).write()
