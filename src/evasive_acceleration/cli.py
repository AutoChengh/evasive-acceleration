"""Command dispatcher for EA evaluation and portable runtime measurement."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from . import __version__
from .backends import BACKEND_NAMES, get_backend, use_backend, _preflight_backend


def main(argv: Sequence[str] | None = None) -> int:
    """Select an optional backend and forward the unchanged workflow options."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    selector = argparse.ArgumentParser(prog="ea", add_help=False)
    selector.add_argument("--backend", choices=BACKEND_NAMES)
    selection, workflow_arguments = selector.parse_known_args(arguments)
    try:
        with use_backend(selection.backend or get_backend()):
            return _dispatch(workflow_arguments)
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
    parser.add_argument(
        "--backend", choices=BACKEND_NAMES,
        help="Explicit solver selection; default: paper. v53 requires its native build.",
    )
    parser.add_argument("command", choices=("single", "batch", "benchmark"), nargs="?")

    # In particular, do not let a missing v53 installation be swallowed by
    # the CSV workflow's per-row error handling and become a file of NaNs.
    # Help/version remain available without optional runtime dependencies.
    if (arguments and arguments[0] in ("single", "batch", "benchmark")
            and not any(value in ("-h", "--help") for value in arguments[1:])):
        _preflight_backend()

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
