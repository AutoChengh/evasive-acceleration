"""Keep the published implementation and its validation evidence named EA."""

import json
from pathlib import Path
import re

from evasive_acceleration.runtime import CORE_ID


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRECTORIES = (
    "src", "docs", "examples", "notebooks", "tools", "tests", ".github",
    "visualization",
)
PUBLIC_ROOT_FILES = (
    "README.md", "pyproject.toml", "requirements.txt", "MANIFEST.in",
    ".gitattributes", ".gitignore",
)
TEXT_SUFFIXES = {
    ".py", ".cpp", ".h", ".hpp", ".md", ".rst", ".txt", ".json",
    ".ipynb", ".toml", ".yaml", ".yml", ".in", ".sh", ".ps1",
    ".html", ".css", ".js", ".cff", ".csv", ".tsv",
}
IGNORED_PARTS = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".ipynb_checkpoints", ".git", "node_modules", "build", "dist", "outputs",
}
# A development identifier is not a semantic package release such as 0.3.0.
DEVELOPMENT_IDENTIFIER = re.compile(r"(?<![a-z0-9])v[0-9]{2}(?![0-9])", re.IGNORECASE)


def _published_paths():
    for directory in PUBLIC_DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            relative = path.relative_to(ROOT)
            if not any(part in IGNORED_PARTS or part.endswith(".egg-info")
                       for part in relative.parts):
                yield path
    for name in PUBLIC_ROOT_FILES:
        yield ROOT / name


def test_published_text_and_paths_use_ea_not_internal_development_identifiers():
    failures = []
    for path in _published_paths():
        relative = path.relative_to(ROOT).as_posix()
        if DEVELOPMENT_IDENTIFIER.search(relative):
            failures.append(f"Internal development identifier in path: {relative}")
        if path.is_file() and (path.suffix.lower() in TEXT_SUFFIXES or
                               path.name in PUBLIC_ROOT_FILES):
            for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if DEVELOPMENT_IDENTIFIER.search(line):
                    failures.append(f"Internal development identifier in text: {relative}:{number}")
    assert not failures, "\n".join(failures)


def test_ea_is_the_only_packaged_source_manifest():
    package = ROOT / "src/evasive_acceleration/_experimental"
    assert sorted(path.name for path in package.glob("*.json")) == ["ea_sources.json"]
    manifest = json.loads((package / "ea_sources.json").read_text(encoding="utf-8"))
    assert "version" not in manifest
    assert manifest["method"] == "Evasive Acceleration (EA)"
    assert manifest["core_id"] == CORE_ID
    assert len(manifest["sources"]) == 15
    for name, digest in manifest["sources"].items():
        assert (package / name).is_file(), name
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name


def test_frozen_validation_evidence_does_not_describe_another_current_backend():
    evidence = json.loads(
        (ROOT / "docs/data/ea_acceptance_20261022.json").read_text(encoding="utf-8")
    )
    assert evidence["schema_version"] == 2
    assert evidence["method"] == "Evasive Acceleration (EA)"
    assert evidence["core_id"] == CORE_ID
    assert evidence["record_type"] == "frozen_validation_evidence"
    forbidden_keys = {"default_public_backend", "public_backend", "solver"}

    def check_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in forbidden_keys, key
                assert "opt_in" not in key.lower(), key
                assert "optin" not in key.lower(), key
                check_keys(child)
        elif isinstance(value, list):
            for child in value:
                check_keys(child)

    check_keys(evidence)
