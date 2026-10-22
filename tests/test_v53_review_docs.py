"""Keep the v53 review checkout instructions and runtime guidance usable."""

from pathlib import Path
import re
import shlex
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
REVIEW_BRANCH = "review/v53-integration-20261022"
REPOSITORY_URL = "https://github.com/AutoChengh/evasive-acceleration-v2.git"


def test_readme_clones_v53_review_branch_before_installation():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installation = readme.split("## Install and run\n", 1)[1].split("\n## ", 1)[0]
    command = next(line for line in installation.splitlines() if line.startswith("git clone "))
    arguments = shlex.split(command)

    assert arguments[:2] == ["git", "clone"]
    assert arguments[arguments.index("--branch") + 1] == REVIEW_BRANCH
    assert arguments[-1] == REPOSITORY_URL
    assert installation.index(command) < installation.index("python -m pip install")
    assert "python -m evasive_acceleration.native_build" in installation


def test_runtime_guide_distinguishes_paper_and_v53_warmups():
    guide = " ".join((ROOT / "docs/runtime.md").read_text(encoding="utf-8").split())

    assert "default `paper` backend" in guide
    assert "three excluded warm-up calls" in guide
    assert "With `--backend v53`" in guide
    assert "`max(--warmup, number of selected inputs)`" in guide
    assert "every selected input is evaluated at least once before measurement" in guide
    assert "Both backends default to one measured pass per frame." in guide
    assert "requested and actual warm-up counts separately" in guide


def _local_targets(markdown):
    yield from re.findall(r"\[[^\]]+\]\(([^\s)]+)\)", markdown)
    yield from re.findall(r"(?:href|src)=[\"']([^\"']+)[\"']", markdown)


def _heading_anchors(markdown):
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+)$", markdown, flags=re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        anchors.add(slug)
    return anchors


@pytest.mark.parametrize("relative_path", ["README.md", "docs/runtime.md"])
def test_review_readme_and_runtime_local_links_resolve(relative_path):
    document = ROOT / relative_path
    markdown = document.read_text(encoding="utf-8")
    targets = list(_local_targets(markdown))
    assert targets

    for target in targets:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        destination = document.parent / unquote(parsed.path) if parsed.path else document
        assert destination.is_file(), f"Broken local link in {relative_path}: {target}"
        if parsed.fragment and destination.suffix == ".md":
            anchors = _heading_anchors(destination.read_text(encoding="utf-8"))
            assert unquote(parsed.fragment) in anchors, (
                f"Missing local heading in {relative_path}: {target}"
            )
