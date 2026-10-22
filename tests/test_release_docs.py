"""Keep the formal release instructions, runtime guidance and local links usable."""

from pathlib import Path
import re
import shlex
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_URL = "https://github.com/AutoChengh/evasive-acceleration.git"


def test_readme_installs_the_formal_repository_before_running_ea():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    installation = readme.split("## Install and run\n", 1)[1].split("\n## ", 1)[0]
    command = next(line for line in installation.splitlines() if line.startswith("git clone "))
    arguments = shlex.split(command)

    assert arguments[:2] == ["git", "clone"]
    assert "--branch" not in arguments
    assert arguments[-1] == REPOSITORY_URL
    assert installation.index(command) < installation.index("python -m pip install")
    assert "python -m evasive_acceleration.native_build" in installation


def test_runtime_guide_explains_full_input_warmup_and_one_measured_pass():
    guide = " ".join((ROOT / "docs/runtime.md").read_text(encoding="utf-8").split())
    assert "--backend" not in guide
    assert "max(--warmup, number of selected inputs)" in guide
    assert "Before measurement" in guide and "every selected input at least once" in guide
    assert "one measured pass" in guide
    assert "requested and actual" in guide and "warm-up" in guide



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
def test_release_readme_and_runtime_local_links_resolve(relative_path):
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
