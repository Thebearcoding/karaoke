"""Create a fresh output directory; experiment artifacts are never overwritten."""

import argparse
from pathlib import Path


def output_directory(name):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,
        default=Path(__file__).resolve().parent/'experiments'/name)
    path=parser.parse_args().output
    path.mkdir(parents=True,exist_ok=False)
    return path
