"""Static, caller-controlled visualisations for aggregate ABSA results."""

from __future__ import annotations

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def _axes(ax: Axes | None, *, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Reuse a caller's axes or create a figure with the requested size."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def plot_ate_comparison(
    frame: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot precision, recall, and F1 for each aspect extractor.

    Args:
        frame: DataFrame containing ``pipeline``, ``precision``, ``recall``,
            and ``f1`` columns.
        ax: Optional axes to draw on; a new figure is created when omitted.

    Returns:
        The figure and axes containing the grouped bar chart.
    """

    figure, axes = _axes(ax, figsize=(9, 5))
    positions = np.arange(len(frame))
    width = 0.24
    for offset, metric in zip((-1, 0, 1), ("precision", "recall", "f1")):
        axes.bar(positions + offset * width, frame[metric], width, label=metric.title())
    axes.set_xticks(positions, frame["pipeline"], rotation=25, ha="right")
    axes.set_ylim(0, 1)
    axes.set_ylabel("Score")
    axes.set_xlabel("Aspect extraction pipeline")
    axes.set_title("Aspect extraction performance")
    axes.legend()
    figure.tight_layout()
    return figure, axes


def plot_linking_tradeoff(
    frame: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot linker accuracy against gold-aspect coverage.

    Args:
        frame: DataFrame containing ``method``, ``coverage``, and ``accuracy``.
        ax: Optional axes to draw on; a new figure is created when omitted.

    Returns:
        The figure and axes containing the labelled trade-off scatter plot.
    """

    figure, axes = _axes(ax, figsize=(7, 5))
    axes.scatter(frame["coverage"], frame["accuracy"], s=65)
    for row in frame.itertuples(index=False):
        axes.annotate(str(row.method), (row.coverage, row.accuracy), xytext=(5, 5), textcoords="offset points")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel("Gold-aspect coverage")
    axes.set_ylabel("Polarity accuracy on linked aspects")
    axes.set_title("Aspect--opinion linking trade-off")
    figure.tight_layout()
    return figure, axes


def plot_product_summary(
    frame: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot positive and negative link counts for product/aspect rows.

    Args:
        frame: DataFrame containing ``product``, ``aspect``, ``positive``, and
            ``negative`` columns.
        ax: Optional axes to draw on; a new figure is created when omitted.

    Returns:
        The figure and axes containing the horizontal sentiment bars.
    """

    height = max(4.0, 0.38 * max(len(frame), 1) + 1.5)
    figure, axes = _axes(ax, figsize=(9, height))
    positions = np.arange(len(frame))
    labels = frame["product"].astype(str) + " · " + frame["aspect"].astype(str)
    axes.barh(positions, frame["positive"], label="Positive")
    axes.barh(positions, -frame["negative"], label="Negative")
    axes.set_yticks(positions, labels)
    axes.axvline(0, color="#333333", linewidth=0.8)
    axes.set_xlabel("Predicted opinion count")
    axes.set_ylabel("Product aspect")
    axes.set_title("Product-level aspect sentiment summary")
    axes.legend()
    axes.invert_yaxis()
    figure.tight_layout()
    return figure, axes
