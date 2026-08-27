import pytest
from pathlib import Path
from spacy.language import Language

from review_absa.data import load_corpus
from review_absa.evaluation import (
    evaluate_aspects,
    evaluate_aspect_polarity,
    evaluate_linking_component,
    gold_aspect_records,
    phrase_key,
)
from review_absa.linking import AspectOpinionLink, PMIModel
from review_absa.opinions import Opinion
from review_absa.schema import GoldAspect


def test_exact_matching_counts_literal_span_matches() -> None:
    predicted = {("r1", 0): ("battery life", "screen")}
    gold = {("r1", 0): (GoldAspect("battery life", 1), GoldAspect("lens", -1))}
    result = evaluate_aspects(predicted, gold, match="exact")
    assert result.true_positive == 1
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.f1 == pytest.approx(0.5)


def test_head_matching_accepts_different_modifiers() -> None:
    predicted = {("r1", 0): ("picture quality",)}
    gold = {("r1", 0): (GoldAspect("image quality", 1),)}
    result = evaluate_aspects(predicted, gold, match="head")
    assert result.f1 == 1.0


def test_phrase_key_rejects_unknown_matching_rule() -> None:
    with pytest.raises(ValueError, match="Unsupported span matching rule"):
        phrase_key("battery", "substring")


def test_gold_component_evaluation_is_explicit(
    sample_data_root: Path,
    nlp: Language,
) -> None:
    corpus = load_corpus(sample_data_root, nlp)
    aspects = gold_aspect_records(corpus)
    assert len(aspects) == 10

    review = corpus.reviews[0]
    excellent = next(token for token in review.doc if token.text == "excellent")
    opinion = Opinion(
        review.uid,
        review.product.name,
        0,
        excellent.text,
        "excellent",
        excellent.i,
        1,
        excellent,
    )
    result = evaluate_linking_component(
        corpus,
        (opinion,),
        PMIModel.from_scores({("life", "excellent"): 1.0}),
        "pmi",
    )
    assert result.covered == 1
    assert result.total == 10
    assert result.accuracy == 1.0


def _link(*, aspect: str, polarity: int) -> AspectOpinionLink:
    return AspectOpinionLink(
        review_uid="r1",
        product_name="Camera",
        sentence_index=0,
        aspect_text=aspect,
        aspect_start_token=0,
        aspect_end_token=len(aspect.split()),
        opinion_text="poor",
        opinion_lemma="poor",
        polarity=polarity,
    )


def test_joint_metric_penalises_correct_span_with_wrong_polarity() -> None:
    links = (_link(aspect="battery life", polarity=-1),)
    gold = {("r1", 0): (GoldAspect("battery life", 1),)}
    result = evaluate_aspect_polarity(links, gold, match="exact")
    assert result.true_positive == 0
    assert result.false_positive == 1
    assert result.false_negative == 1
    assert result.alignment_report.reason_counts == {"wrong_polarity": 1}
    assert result.alignment_report.errors[0].predicted_polarity == -1


def test_joint_metric_reports_unlinked_prediction_coverage() -> None:
    gold = {("r1", 0): (GoldAspect("battery", 1),)}
    result = evaluate_aspect_polarity(
        (), gold, match="exact", predicted_aspect_count=2
    )
    assert result.linked_aspect_coverage == 0.0
    assert result.alignment_report.reason_counts == {"uncovered_aspect": 1}
