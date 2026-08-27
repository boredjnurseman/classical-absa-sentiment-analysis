from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).parents[1]


def test_project_metadata_declares_supported_runtime() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "classical-absa"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.12,<3.13"


def test_tracked_tree_contains_no_forbidden_artifacts() -> None:
    forbidden = {".DS_Store", ".pytest_cache", "__pycache__", ".ipynb_checkpoints"}
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not [path for path in tracked if forbidden & set(Path(path).parts)]


def test_readme_documents_reproduction_and_evaluation_boundary() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "python -m review_absa.experiments" in text
    assert "gold aspects" in text.lower()
    assert "predicted aspects" in text.lower()
    assert "artifacts/product_summaries.csv" in text
