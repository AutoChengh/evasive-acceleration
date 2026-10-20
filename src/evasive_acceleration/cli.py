"""Command dispatcher for EA evaluation and portable runtime measurement."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__


def main(argv: Sequence[str] | None = None) -> int:
    """Forward command arguments to the existing workflow entry points."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="ea",
        description="Compute Evasive Acceleration (EA), in m/s^2.",
        epilog="Use 'ea single --help', 'ea batch --help', or 'ea benchmark --help'.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=("single", "batch", "benchmark"), nargs="?")

    # Delegate before parsing so each workflow owns its --help and options.
    if arguments and arguments[0] == "single":
        from single_frame import main as single_main

        return single_main(arguments[1:]) or 0
    if arguments and arguments[0] == "batch":
        try:
            from batch_compute import main as batch_main
        except ModuleNotFoundError as exc:
            if exc.name != "pandas":
                raise
            parser.exit(
                2,
                "The CSV workflow requires pandas. Install it with:\n"
                '  python -m pip install -e ".[batch]"\n',
            )
        return batch_main(arguments[1:]) or 0
    if arguments and arguments[0] == "benchmark":
        from .benchmark import main as benchmark_main

        return benchmark_main(arguments[1:])

    parser.parse_args(arguments)
    parser.print_help()
    return 0
