from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "Classical_ABSA_Pipeline.ipynb"


def test_notebook_has_expected_story_and_no_implementation() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join("".join(cell.source) for cell in notebook.cells)
    headings = [
        "Problem and result at a glance",
        "System architecture",
        "Data and evaluation boundary",
        "Aspect extraction",
        "Opinion induction and negation",
        "Aspect-opinion linking",
        "End-to-end inference",
        "Product summaries and error analysis",
        "Conclusions",
    ]
    assert all(heading in source for heading in headings)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "def " not in code and "class " not in code
    assert "pip install" not in code and "google.colab" not in code


def test_saved_notebook_contains_no_error_outputs() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    errors = [
        output
        for cell in notebook.cells
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    assert not errors


def test_notebook_code_cells_remain_thin() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    long_cells = [
        cell.source
        for cell in notebook.cells
        if cell.cell_type == "code" and len(cell.source.splitlines()) > 10
    ]
    assert not long_cells


def test_notebook_omits_volatile_execution_timestamps() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert not [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "execution" in cell.metadata
    ]
