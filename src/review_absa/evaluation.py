"""Sentence-level evaluation shared by pipeline components."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from .aspects import AlignmentError, AlignmentReport, Aspect, _gold_token_spans
from .linking import AspectOpinionLink, PMIModel, LinkMethod, link_aspects
from .opinions import Opinion
from .schema import Corpus, GoldAspect, SentenceKey


MatchRule = Literal["exact", "head"]


@dataclass(frozen=True)
class SpanMetrics:
    """Aggregate span-level precision, recall and F1 counts."""

    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def predicted(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def gold(self) -> int:
        return self.true_positive + self.false_negative


@dataclass(frozen=True)
class LinkingMetrics:
    """Gold-aspect component metrics for polarity linking."""

    coverage: float
    accuracy: float
    macro_f1: float
    total: int
    covered: int
    alignment_report: AlignmentReport | None = None


@dataclass(frozen=True)
class JointMetrics:
    """Joint aspect-span and polarity performance for the deployed pipeline."""

    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    linked_aspect_coverage: float
    alignment_report: AlignmentReport | None = None


def phrase_key(text: str, match: MatchRule) -> str:
    """Normalise a phrase for exact or syntactic-head matching."""

    normalised = " ".join(text.lower().split())
    if match == "exact":
        return normalised
    if match == "head":
        return normalised.split()[-1] if normalised else ""
    raise ValueError(f"Unsupported span matching rule: {match}")


def evaluate_aspects(
    predicted: Mapping[SentenceKey, Iterable[str]],
    gold: Mapping[SentenceKey, Iterable[GoldAspect]],
    *,
    match: MatchRule = "exact",
) -> SpanMetrics:
    """Compare unique predicted and gold aspect spans within each sentence."""

    true_positive = false_positive = false_negative = 0
    for key in set(predicted) | set(gold):
        predicted_set = {
            phrase_key(text, match)
            for text in predicted.get(key, ())
            if text.strip()
        }
        gold_set = {
            phrase_key(item.text, match)
            for item in gold.get(key, ())
            if item.text.strip()
        }
        true_positive += len(predicted_set & gold_set)
        false_positive += len(predicted_set - gold_set)
        false_negative += len(gold_set - predicted_set)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SpanMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def gold_aspect_records(corpus: Corpus) -> tuple[Aspect, ...]:
    """Materialise aligned gold spans for isolated component evaluation."""

    aspects: list[Aspect] = []
    for review in corpus:
        for sentence_index, sentence in enumerate(review.sentences):
            for annotation in review.annotations.get(sentence_index, ()):
                if annotation.polarity == 0:
                    continue
                spans = _gold_token_spans(sentence, [annotation.text])
                if not spans:
                    continue
                start, end = spans[0]
                tokens = tuple(review.doc[start:end])
                aspects.append(
                    Aspect(
                        review_uid=review.uid,
                        product_name=review.product.name,
                        sentence_index=sentence_index,
                        text=" ".join(annotation.text.lower().split()),
                        start_token=start,
                        end_token=end,
                        tokens=tokens,
                    )
                )
    return tuple(
        sorted(
            aspects,
            key=lambda item: (
                item.review_uid,
                item.sentence_index,
                item.start_token,
                item.text,
            ),
        )
    )


def evaluate_linking_component(
    corpus: Corpus,
    opinions: Iterable[Opinion],
    pmi: PMIModel,
    method: LinkMethod,
    *,
    max_dep_distance: int = 4,
) -> LinkingMetrics:
    """Evaluate linking and polarity with gold aspect spans supplied explicitly."""

    aspects = gold_aspect_records(corpus)
    links = link_aspects(
        aspects,
        opinions,
        pmi=pmi,
        method=method,
        max_dep_distance=max_dep_distance,
    )
    gold_polarity = {
        (review.uid, sentence_index, phrase_key(annotation.text, "exact")): (
            1 if annotation.polarity > 0 else -1
        )
        for review in corpus
        for sentence_index, annotations in review.annotations.items()
        for annotation in annotations
        if annotation.polarity != 0
    }
    links_by_key = {
        (
            link.review_uid,
            link.sentence_index,
            phrase_key(link.aspect_text, "exact"),
        ): link
        for link in links
    }
    errors: list[AlignmentError] = []
    for key, actual in gold_polarity.items():
        link = links_by_key.get(key)
        if link is None:
            errors.append(
                AlignmentError(
                    reason="uncovered_aspect",
                    review_uid=key[0],
                    sentence_index=key[1],
                    aspect_text=key[2],
                    gold_polarity=actual,
                )
            )
        elif link.polarity != actual:
            errors.append(
                AlignmentError(
                    reason="wrong_polarity",
                    review_uid=link.review_uid,
                    sentence_index=link.sentence_index,
                    aspect_text=link.aspect_text,
                    gold_polarity=actual,
                    predicted_polarity=link.polarity,
                )
            )
    for key, link in links_by_key.items():
        if key not in gold_polarity:
            errors.append(
                AlignmentError(
                    reason="unmatched_span",
                    review_uid=link.review_uid,
                    sentence_index=link.sentence_index,
                    aspect_text=link.aspect_text,
                    predicted_polarity=link.polarity,
                )
            )
    paired = [
        (
            gold_polarity[
                (
                    link.review_uid,
                    link.sentence_index,
                    phrase_key(link.aspect_text, "exact"),
                )
            ],
            link.polarity,
        )
        for link in links
        if (
            link.review_uid,
            link.sentence_index,
            phrase_key(link.aspect_text, "exact"),
        )
        in gold_polarity
    ]
    covered = len(paired)
    total = len(gold_polarity)
    accuracy = (
        sum(actual == predicted for actual, predicted in paired) / covered
        if covered
        else 0.0
    )
    class_f1 = []
    for label in (-1, 1):
        true_positive = sum(
            actual == label and predicted == label for actual, predicted in paired
        )
        false_positive = sum(
            actual != label and predicted == label for actual, predicted in paired
        )
        false_negative = sum(
            actual == label and predicted != label for actual, predicted in paired
        )
        denominator = 2 * true_positive + false_positive + false_negative
        class_f1.append(2 * true_positive / denominator if denominator else 0.0)
    return LinkingMetrics(
        coverage=covered / total if total else 0.0,
        accuracy=accuracy,
        macro_f1=sum(class_f1) / len(class_f1),
        total=total,
        covered=covered,
        alignment_report=AlignmentReport(
            total_gold_aspects=total,
            aligned_aspects=covered,
            errors=tuple(
                sorted(
                    errors,
                    key=lambda error: (
                        error.review_uid,
                        error.sentence_index,
                        error.aspect_text,
                        error.reason,
                    ),
                )
            ),
        ),
    )


def evaluate_aspect_polarity(
    links: Iterable[AspectOpinionLink],
    gold: Mapping[SentenceKey, Iterable[GoldAspect]],
    *,
    match: MatchRule = "exact",
    predicted_aspect_count: int | None = None,
) -> JointMetrics:
    """Score joint ``(aspect span, polarity)`` predictions by sentence."""

    link_records = tuple(link for link in links if link.polarity != 0)
    predicted: defaultdict[SentenceKey, set[tuple[str, int]]] = defaultdict(set)
    for link in link_records:
        predicted[(link.review_uid, link.sentence_index)].add(
            (phrase_key(link.aspect_text, match), link.polarity)
        )

    true_positive = false_positive = false_negative = 0
    errors: list[AlignmentError] = []
    for key in set(predicted) | set(gold):
        predicted_set = predicted.get(key, set())
        gold_set = {
            (
                phrase_key(annotation.text, match),
                1 if annotation.polarity > 0 else -1,
            )
            for annotation in gold.get(key, ())
            if annotation.text.strip() and annotation.polarity != 0
        }
        true_positive += len(predicted_set & gold_set)
        false_positive += len(predicted_set - gold_set)
        false_negative += len(gold_set - predicted_set)

        gold_spans = {span for span, _polarity in gold_set}
        predicted_spans = {span for span, _polarity in predicted_set}
        for span, polarity in predicted_set:
            if span not in gold_spans:
                errors.append(
                    AlignmentError(
                        reason="unmatched_span",
                        review_uid=key[0],
                        sentence_index=key[1],
                        aspect_text=span,
                        predicted_polarity=polarity,
                    )
                )
            elif (span, polarity) not in gold_set:
                gold_polarity = next(
                    expected for expected_span, expected in gold_set if expected_span == span
                )
                errors.append(
                    AlignmentError(
                        reason="wrong_polarity",
                        review_uid=key[0],
                        sentence_index=key[1],
                        aspect_text=span,
                        gold_polarity=gold_polarity,
                        predicted_polarity=polarity,
                    )
                )
        for span, polarity in gold_set:
            if span not in predicted_spans:
                errors.append(
                    AlignmentError(
                        reason="uncovered_aspect",
                        review_uid=key[0],
                        sentence_index=key[1],
                        aspect_text=span,
                        gold_polarity=polarity,
                    )
                )

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = (
        len(link_records)
        if predicted_aspect_count is None
        else predicted_aspect_count
    )
    if denominator < 0:
        raise ValueError("predicted_aspect_count cannot be negative")
    return JointMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        linked_aspect_coverage=(len(link_records) / denominator if denominator else 0.0),
        alignment_report=AlignmentReport(
            total_gold_aspects=sum(
                1
                for annotations in gold.values()
                for annotation in annotations
                if annotation.polarity != 0
            ),
            aligned_aspects=sum(
                1
                for key in gold
                for annotation in gold[key]
                if annotation.polarity != 0
                and phrase_key(annotation.text, match)
                in {span for span, _polarity in predicted.get(key, set())}
            ),
            errors=tuple(
                sorted(
                    errors,
                    key=lambda error: (
                        error.review_uid,
                        error.sentence_index,
                        error.aspect_text,
                        error.reason,
                    ),
                )
            ),
        ),
    )
