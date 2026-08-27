import pytest

from review_absa.linking import AspectOpinionLink
from review_absa.summaries import make_product_summary


def _link(review_uid: str, aspect: str, polarity: int) -> AspectOpinionLink:
    return AspectOpinionLink(
        review_uid=review_uid,
        product_name="Camera",
        sentence_index=0,
        aspect_text=aspect,
        aspect_start_token=0,
        aspect_end_token=1,
        opinion_text="good",
        opinion_lemma="good",
        polarity=polarity,
    )


def _repeated_links() -> tuple[AspectOpinionLink, ...]:
    return (
        _link("r1", "Battery", 1),
        _link("r1", " battery ", 1),
        _link("r2", "battery", -1),
        _link("r3", "screen", 1),
    )


def test_unique_review_mode_deduplicates_repeated_mentions() -> None:
    frame = make_product_summary(
        _repeated_links(), count_mode="unique_reviews", top_k=None
    )
    battery = frame.loc[frame["aspect"] == "battery"].iloc[0]
    assert battery["positive"] == 1
    assert battery["negative"] == 1


def test_occurrence_mode_counts_every_link() -> None:
    frame = make_product_summary(
        _repeated_links(), count_mode="occurrences", top_k=1
    )
    assert frame[["aspect", "positive", "negative", "total"]].to_dict(
        "records"
    ) == [{"aspect": "battery", "positive": 2, "negative": 1, "total": 3}]


def test_summary_rejects_unknown_count_mode() -> None:
    with pytest.raises(ValueError, match="count_mode"):
        make_product_summary(_repeated_links(), count_mode="sentences")
