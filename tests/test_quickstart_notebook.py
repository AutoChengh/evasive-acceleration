"""Fast structure checks; full cell execution is a separate CI job."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_portable_clean_and_reuses_the_public_workflows():
    notebook = json.loads((ROOT / "notebooks/ea_quickstart.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    sources = []
    for cell in notebook["cells"]:
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        sources.append(source)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            assert "D:\\" not in source and "C:\\Users" not in source
    source = "\n".join(sources)
    for name in ("run_comparison", "compute_from_dataframe", "plot_risk_history",
                 "resolve_figure_font", "interact_manual", "compute_ea"):
        assert name in source
    assert "synthetic" in source.lower()
    assert "collision probability" in source.lower()
