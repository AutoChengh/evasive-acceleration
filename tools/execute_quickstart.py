"""Execute every quickstart cell with this interpreter, without masking errors.

Install ``.[notebook]`` first. The tracked notebook stays output-free; executed
cells, plots, and run summaries go under the ignored outputs/ directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "ea_quickstart.ipynb"


def validate_notebook(notebook) -> None:
    """Do not report success if a cell failed or no figures were produced."""
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", []) if output.output_type == "error"]
    if errors:
        raise RuntimeError(f"Notebook contains {len(errors)} error output(s)")
    unexecuted = [cell for cell in notebook.cells if cell.cell_type == "code"
                  and cell.source.strip() and cell.execution_count is None]
    if unexecuted:
        raise RuntimeError(f"Notebook contains {len(unexecuted)} unexecuted code cell(s)")
    images = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    if len(images) < 3:
        raise RuntimeError("Expected at least three rendered PNG figures from the notebook")


def execute_quickstart(output_path: Path, *, timeout: int = 180) -> dict:
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    if timeout <= 0:
        raise ValueError("timeout must be positive seconds per cell")
    target = Path(output_path).resolve()
    if target == NOTEBOOK.resolve():
        raise ValueError("The executed copy must not overwrite the tracked notebook")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    # Avoid using an unrelated globally registered Python kernel. This changes
    # only an in-memory spec, never the user's kernelspec on disk.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=timeout, allow_errors=False,
        store_widget_state=False,
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    try:
        client.execute()
        validate_notebook(notebook)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep diagnostics if execution fails, but propagate the exception.
        nbformat.write(notebook, target)
    return {"notebook": str(target), "executed_code_cells": sum(
        cell.cell_type == "code" for cell in notebook.cells)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "outputs/notebooks/ea_quickstart.executed.ipynb")
    parser.add_argument("--timeout", type=int, default=180)
    arguments = parser.parse_args(argv)
    result = execute_quickstart(arguments.output, timeout=arguments.timeout)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
