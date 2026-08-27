"""Deterministic product-level summaries from predicted ABSA records."""

from __future__ import annotations

from typing import Iterable, Literal

import pandas as pd

from .linking import AspectOpinionLink


CountMode = Literal["unique_reviews", "occurrences"]
_COLUMNS = ["product", "aspect", "positive", "negative", "total", "count_mode"]


def _normalise(value: str) -> str:
    return " ".join(value.lower().split())


def make_product_summary(
    links: Iterable[AspectOpinionLink],
    *,
    count_mode: CountMode = "unique_reviews",
    top_k: int | None = None,
) -> pd.DataFrame:
    """Count predicted positive and negative evidence by product and aspect."""

    if count_mode not in {"unique_reviews", "occurrences"}:
        raise ValueError("count_mode must be 'unique_reviews' or 'occurrences'")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1 or None")

    records = [
        {
            "product_key": _normalise(link.product_name),
            "product": " ".join(link.product_name.split()),
            "aspect": _normalise(link.aspect_text),
            "review_uid": link.review_uid,
            "sentiment": "positive" if link.polarity > 0 else "negative",
        }
        for link in links
        if link.polarity != 0 and link.aspect_text.strip()
    ]
    if not records:
        return pd.DataFrame(columns=_COLUMNS)

    frame = pd.DataFrame.from_records(records).sort_values(
        ["product_key", "aspect", "review_uid", "sentiment", "product"]
    )
    if count_mode == "unique_reviews":
        frame = frame.drop_duplicates(
            ["product_key", "aspect", "review_uid", "sentiment"]
        )
    display_names = frame.groupby("product_key", sort=True)["product"].first()
    counts = (
        frame.groupby(["product_key", "aspect", "sentiment"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for sentiment in ("positive", "negative"):
        if sentiment not in counts:
            counts[sentiment] = 0
    counts["product"] = counts["product_key"].map(display_names)
    counts["total"] = counts["positive"] + counts["negative"]
    counts["count_mode"] = count_mode
    counts = counts.sort_values(
        ["product_key", "total", "aspect"],
        ascending=[True, False, True],
    )
    if top_k is not None:
        counts = counts.groupby("product_key", sort=False, as_index=False).head(top_k)
    return counts[_COLUMNS].reset_index(drop=True)
