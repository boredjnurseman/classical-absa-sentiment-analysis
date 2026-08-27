from pathlib import Path

import matplotlib
import pandas as pd
import pytest

from matplotlib import pyplot as plt

from review_absa.experiments import (
    ArtifactValidationError,
    publish_artifacts,
    select_linker,
)


def test_linker_selection_uses_dev_macro_f1_then_coverage() -> None:
    rows = pd.DataFrame(
        [
            {"method": "pmi", "macro_f1": 0.70, "coverage": 0.28},
            {"method": "hybrid", "macro_f1": 0.65, "coverage": 0.64},
        ]
    )
    assert select_linker(rows) == "pmi"


def test_linker_selection_uses_coverage_as_tie_breaker() -> None:
    rows = pd.DataFrame(
        [
            {"method": "dep", "macro_f1": 0.65, "coverage": 0.55},
            {"method": "pmi->dep", "macro_f1": 0.65, "coverage": 0.70},
        ]
    )
    assert select_linker(rows) == "pmi->dep"


def test_failed_publication_preserves_previous_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    manifest = output / "run_manifest.json"
    manifest.write_text('{"status":"previous"}', encoding="utf-8")
    with pytest.raises(ArtifactValidationError):
        publish_artifacts({"crf_metrics.csv": pd.DataFrame()}, output)
    assert "previous" in manifest.read_text(encoding="utf-8")


def test_experiment_runner_uses_headless_matplotlib_backend() -> None:
    assert matplotlib.get_backend().lower() == "agg"


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_failed_publication_restores_the_complete_previous_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_on_call: int,
) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    previous = {
        "corpus_summary.csv": "previous-corpus",
        "ate_baselines.csv": "previous-ate",
        "crf_metrics.csv": "previous-crf",
        "negation_ablation.csv": "previous-negation",
        "linking_metrics.csv": "previous-linking",
        "end_to_end_metrics.csv": "previous-e2e",
        "product_summaries.csv": "previous-summary",
        "run_manifest.json": '{"status":"previous"}\n',
    }
    for name, contents in previous.items():
        (output / name).write_text(contents, encoding="utf-8")
    (output / "figures").mkdir()
    (output / "figures" / "ate_comparison.png").write_bytes(b"previous-ate-figure")
    (output / "figures" / "linking_tradeoff.png").write_bytes(b"previous-linking-figure")
    (output / "figures" / "product_summaries.png").write_bytes(b"previous-summary-figure")

    tables = {
        name: pd.DataFrame(columns=columns)
        for name, columns in {
            "corpus_summary.csv": [
                "dataset", "n_reviews", "n_files", "n_products", "n_labels",
                "positive_labels", "negative_labels",
            ],
            "ate_baselines.csv": [
                "split", "pipeline", "match", "precision", "recall", "f1",
                "true_positive", "false_positive", "false_negative", "predicted", "gold",
            ],
            "crf_metrics.csv": [
                "split", "match", "precision", "recall", "f1", "true_positive",
                "false_positive", "false_negative", "predicted", "gold",
            ],
            "negation_ablation.csv": [
                "split", "method", "negation", "coverage", "accuracy", "macro_f1",
                "n_total", "n_covered",
            ],
            "linking_metrics.csv": [
                "split", "method", "coverage", "accuracy", "macro_f1", "n_total", "n_covered",
            ],
            "end_to_end_metrics.csv": [
                "split", "linker", "match", "precision", "recall", "f1", "true_positive",
                "false_positive", "false_negative", "predicted", "gold", "linked_aspect_coverage",
            ],
            "product_summaries.csv": [
                "product", "aspect", "positive", "negative", "total", "count_mode",
            ],
        }.items()
    }
    figures = {
        "ate_comparison.png": plt.figure(),
        "linking_tradeoff.png": plt.figure(),
        "product_summaries.png": plt.figure(),
    }

    original_replace = Path.replace
    calls = 0

    def fail_on_new_result(source: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls == fail_on_call:
            raise OSError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_on_new_result)
    with pytest.raises(OSError, match="injected publication failure"):
        publish_artifacts(tables, output, figures=figures)

    for name, contents in previous.items():
        assert (output / name).read_text(encoding="utf-8") == contents
    assert (output / "figures" / "ate_comparison.png").read_bytes() == b"previous-ate-figure"
    assert (output / "figures" / "linking_tradeoff.png").read_bytes() == b"previous-linking-figure"
    assert (output / "figures" / "product_summaries.png").read_bytes() == b"previous-summary-figure"
    for figure in figures.values():
        plt.close(figure)
