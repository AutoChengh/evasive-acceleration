"""Command dispatcher for EA evaluation and portable runtime measurement."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .runtime import _preflight_runtime


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sole EA implementation with the existing workflow options."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--backend" in arguments or any(value.startswith("--backend=") for value in arguments):
        print("ea: error: --backend was removed; current EA has one implementation. See docs/history.md for archived code.", file=sys.stderr)
        return 2
    try:
        return _dispatch(arguments)
    except RuntimeError as exc:
        print(f"ea: error: {exc}", file=sys.stderr)
        return 2


def _dispatch(arguments: list[str]) -> int:
    """Forward command arguments to the existing workflow entry points."""
    parser = argparse.ArgumentParser(
        prog="ea",
        description="Compute Evasive Acceleration (EA), in m/s^2.",
        epilog="Use 'ea single --help', 'ea batch --help', or 'ea benchmark --help'.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=("single", "batch", "benchmark"), nargs="?")

    # Do not let a missing native runtime be swallowed by
    # the CSV workflow's per-row error handling and become a file of NaNs.
    # Help/version remain available without optional runtime dependencies.
    if (arguments and arguments[0] in ("single", "batch", "benchmark")
            and not any(value in ("-h", "--help") for value in arguments[1:])):
        _preflight_runtime()

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
