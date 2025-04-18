#!/usr/bin/env python3

import argparse
from tts_encoders import KokoroEncoder, VOICES, DEFAULT_KOKORO_REPO
from os.path import isdir, dirname


def comma_separated_list(arg: str):
    return [v.strip() for v in arg.split(",")]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True, help="the path to save the converted gguf tts model too.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_KOKORO_REPO, help="The repo to load the model and configuration from.")
    parser.add_argument("--voices", type=comma_separated_list, default=VOICES, help="A common separated list of voice names.")
    # for now we should default to espeak phonemization, as there are still some problems with TTS phonemization encoding in GGUF
    parser.add_argument("--tts-phonemizer", default=False, action='store_true', help="Whether to encode the model to use the tts native phonemizer.")
    parser.add_argument("--never-make-dirs", default=False, action="store_true", help="When set the script will never add new directories.")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = parse_arguments()
    if not isdir(dirname(args.save_path)) and args.never_make_dirs:
        raise ValueError(f"model path, {args.save_path} is not a valid path.")
    KokoroEncoder(args.save_path, repo_id=args.repo_id, voices=args.voices, use_espeak=not args.tts_phonemizer).write()
