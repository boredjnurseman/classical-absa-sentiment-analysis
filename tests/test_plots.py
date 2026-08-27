import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from review_absa.plots import (  # noqa: E402
    plot_ate_comparison,
    plot_linking_tradeoff,
    plot_product_summary,
)


def test_linking_tradeoff_labels_each_method() -> None:
    frame = pd.DataFrame(
        [
            {"method": "pmi", "coverage": 0.3, "accuracy": 0.8},
            {"method": "pmi->dep", "coverage": 0.6, "accuracy": 0.7},
        ]
    )
    fig, ax = plot_linking_tradeoff(frame)
    assert {text.get_text() for text in ax.texts} == set(frame["method"])
    assert ax.get_xlabel() and ax.get_ylabel()
    plt.close(fig)


def test_all_plot_families_return_figure_and_axes() -> None:
    ate = pd.DataFrame(
        [{"pipeline": "crf", "precision": 0.8, "recall": 0.7, "f1": 0.75}]
    )
    summary = pd.DataFrame(
        [
            {
                "product": "Camera",
                "aspect": "battery",
                "positive": 3,
                "negative": 1,
                "total": 4,
                "count_mode": "occurrences",
            }
        ]
    )
    for plot, frame in (
        (plot_ate_comparison, ate),
        (plot_product_summary, summary),
    ):
        fig, ax = plot(frame)
        assert fig is ax.figure
        assert ax.get_title()
        plt.close(fig)
