"""Dispatch to the dense training, preparation, and sampling commands."""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="Dense video trajectory distillation")
    commands = {
        "prepare": "prepare",
        "train": "train",
        "sample": "sample",
        "report": "report",
        "prepare-h3": "h3.prepare",
        "import-h3-cache": "h3.import_cache",
        "train-h3": "h3.train",
        "sample-h3": "h3.sample",
    }
    parser.add_argument("command", choices=tuple(commands))
    args = sys.argv[1:]
    if not args:
        parser.print_help()
        return
    command = parser.parse_args(args[:1]).command
    sys.argv = [f"video-blade {command}", *args[1:]]
    importlib.import_module(f"video_blade.{commands[command]}").main()


if __name__ == "__main__":
    main()
