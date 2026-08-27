"""Classical aspect candidates and train-fitted filtering baselines."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Sequence, Self

import sklearn_crfsuite
from spacy.tokens import Span
from spacy.tokens import Token

from .schema import Corpus, GoldAspect, SentenceKey


@dataclass(frozen=True)
class Aspect:
    """One predicted aspect span."""

    review_uid: str
    product_name: str
    sentence_index: int
    text: str
    start_token: int
    end_token: int
    tokens: tuple[Token, ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class SequenceDataset:
    """Sentence-level CRF inputs with the context needed to decode spans."""

    features: tuple[tuple[dict[str, object], ...], ...]
    labels: tuple[tuple[str, ...], ...]
    keys: tuple[SentenceKey, ...]
    product_names: tuple[str, ...]
    tokens: tuple[tuple[Token, ...], ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class AlignmentError:
    """Internal, location-aware explanation for one alignment failure."""

    reason: str
    review_uid: str
    sentence_index: int
    aspect_text: str
    gold_polarity: int | None = None
    predicted_polarity: int | None = None


@dataclass(frozen=True)
class AlignmentReport:
    """Coverage of sentence annotations by exact contiguous token spans."""

    total_gold_aspects: int
    aligned_aspects: int
    errors: tuple[AlignmentError, ...] = ()

    @property
    def reason_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter(error.reason for error in self.errors)
        return dict(sorted(counts.items()))

    @property
    def match_rate(self) -> float:
        return (
            self.aligned_aspects / self.total_gold_aspects
            if self.total_gold_aspects
            else 0.0
        )


def _reasonable(text: str) -> bool:
    value = text.strip()
    return len(value) >= 2 and bool(re.search(r"[A-Za-z]", value)) and not any(
        character.isdigit() for character in value
    )


def _sort(aspects: Iterable[Aspect]) -> tuple[Aspect, ...]:
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


def extract_aspect_candidates(corpus: Corpus) -> tuple[Aspect, ...]:
    """Extract noun chunks and uncovered standalone nouns."""

    candidates: list[Aspect] = []
    for review in corpus:
        for sentence_index, sentence in enumerate(review.sentences):
            chunks = tuple(sentence.noun_chunks)
            for chunk in chunks:
                text = " ".join(chunk.text.lower().split())
                if _reasonable(text):
                    candidates.append(
                        Aspect(
                            review_uid=review.uid,
                            product_name=review.product.name,
                            sentence_index=sentence_index,
                            text=text,
                            start_token=chunk.start,
                            end_token=chunk.end,
                            tokens=tuple(chunk),
                        )
                    )
            for token in sentence:
                if token.pos_ not in {"NOUN", "PROPN"}:
                    continue
                if any(chunk.start <= token.i < chunk.end for chunk in chunks):
                    continue
                text = token.text.lower().strip()
                if _reasonable(text):
                    candidates.append(
                        Aspect(
                            review_uid=review.uid,
                            product_name=review.product.name,
                            sentence_index=sentence_index,
                            text=text,
                            start_token=token.i,
                            end_token=token.i + 1,
                            tokens=(token,),
                        )
                    )
    return _sort(candidates)


def linguistic_filter(aspects: Iterable[Aspect]) -> tuple[Aspect, ...]:
    """Keep noun phrases with at least one non-stop lexical token."""

    kept = []
    for aspect in aspects:
        lexical = [token for token in aspect.tokens if not token.is_space and not token.is_punct]
        if not lexical or all(token.is_stop for token in lexical):
            continue
        if lexical[-1].pos_ in {"PRON", "DET", "AUX", "PART"}:
            continue
        kept.append(aspect)
    return _sort(kept)


def _top_vocabulary(aspects: tuple[Aspect, ...], top_k: int) -> dict[str, frozenset[str]]:
    by_product_review: defaultdict[str, defaultdict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for aspect in aspects:
        by_product_review[aspect.product_name][aspect.review_uid][aspect.text] += 1

    result: dict[str, frozenset[str]] = {}
    for product, reviews in by_product_review.items():
        document_frequency: Counter[str] = Counter()
        term_frequency: Counter[str] = Counter()
        for counts in reviews.values():
            document_frequency.update(counts.keys())
            term_frequency.update(counts)
        n_documents = len(reviews)
        scores = {
            term: term_frequency[term]
            * (math.log((1 + n_documents) / (1 + document_frequency[term])) + 1)
            for term in term_frequency
        }
        ranked = sorted(scores, key=lambda term: (-scores[term], term))[:top_k]
        result[product] = frozenset(ranked)
    return result


def _filter_vocabulary(
    aspects: Iterable[Aspect],
    vocabulary: dict[str, frozenset[str]],
) -> tuple[Aspect, ...]:
    return _sort(
        aspect
        for aspect in aspects
        if aspect.text in vocabulary.get(aspect.product_name, frozenset())
    )


@dataclass(frozen=True)
class ATEBaselineSuite:
    """Five candidate baselines with statistical state fitted on training data."""

    raw_vocabulary: dict[str, frozenset[str]]
    linguistic_vocabulary: dict[str, frozenset[str]]

    @classmethod
    def fit(cls, corpus: Corpus, *, top_k: int = 200) -> Self:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        raw = extract_aspect_candidates(corpus)
        linguistic = linguistic_filter(raw)
        return cls(
            raw_vocabulary=_top_vocabulary(raw, top_k),
            linguistic_vocabulary=_top_vocabulary(linguistic, top_k),
        )

    def transform(self, corpus: Corpus) -> dict[str, tuple[Aspect, ...]]:
        raw = extract_aspect_candidates(corpus)
        linguistic = linguistic_filter(raw)
        statistical = _filter_vocabulary(raw, self.raw_vocabulary)
        linguistic_then_statistical = _filter_vocabulary(
            linguistic,
            self.linguistic_vocabulary,
        )
        statistical_then_linguistic = linguistic_filter(statistical)
        return {
            "raw": raw,
            "linguistic": linguistic,
            "statistical": statistical,
            "linguistic_then_statistical": linguistic_then_statistical,
            "statistical_then_linguistic": statistical_then_linguistic,
        }


def aspects_by_sentence(
    aspects: Iterable[Aspect],
) -> dict[SentenceKey, tuple[str, ...]]:
    """Group predicted aspect text by review and sentence."""

    grouped: defaultdict[SentenceKey, list[str]] = defaultdict(list)
    for aspect in aspects:
        grouped[(aspect.review_uid, aspect.sentence_index)].append(aspect.text)
    return {key: tuple(values) for key, values in grouped.items()}


def _normalised_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z0-9]+(?:['/-][A-Za-z0-9]+)*", text.lower()))


def _gold_token_spans(
    sentence: Span,
    aspect_texts: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Align gold phrases to non-overlapping sentence token spans."""

    annotations = tuple(GoldAspect(text, 0) for text in aspect_texts)
    return tuple(
        sorted(
            span
            for _annotation, span in _aligned_gold_annotations(sentence, annotations)
            if span is not None
        )
    )


def _aligned_gold_annotations(
    sentence: Span,
    annotations: Sequence[GoldAspect],
) -> tuple[tuple[GoldAspect, tuple[int, int] | None], ...]:
    """Return each annotation paired with its unambiguous token span."""

    sentence_words = tuple(token.text.lower() for token in sentence)
    occupied: set[int] = set()
    aligned: list[tuple[GoldAspect, tuple[int, int] | None]] = [
        (annotation, None) for annotation in annotations
    ]
    ranked = sorted(
        enumerate(annotations),
        key=lambda item: (-len(_normalised_tokens(item[1].text)), item[0]),
    )
    for original_index, annotation in ranked:
        words = _normalised_tokens(annotation.text)
        if not words:
            continue
        for relative_start in range(len(sentence_words) - len(words) + 1):
            relative_end = relative_start + len(words)
            if sentence_words[relative_start:relative_end] != words:
                continue
            token_indexes = range(relative_start, relative_end)
            if any(index in occupied for index in token_indexes):
                continue
            occupied.update(token_indexes)
            aligned[original_index] = (
                annotation,
                (sentence.start + relative_start, sentence.start + relative_end),
            )
            break
    return tuple(aligned)


def _token_features(sentence: Span, relative_index: int) -> dict[str, object]:
    token = sentence[relative_index]

    def lexical(prefix: str, value: Token) -> dict[str, object]:
        lower = value.text.lower()
        return {
            f"{prefix}lower": lower,
            f"{prefix}lemma": value.lemma_.lower() or lower,
            f"{prefix}pos": value.pos_,
            f"{prefix}tag": value.tag_,
            f"{prefix}dep": value.dep_,
            f"{prefix}shape": value.shape_,
            f"{prefix}is_stop": value.is_stop,
            f"{prefix}is_alpha": value.is_alpha,
            f"{prefix}prefix2": lower[:2],
            f"{prefix}prefix3": lower[:3],
            f"{prefix}suffix2": lower[-2:],
            f"{prefix}suffix3": lower[-3:],
        }

    features = {"bias": 1.0, **lexical("", token)}
    if relative_index == 0:
        features["BOS"] = True
    else:
        features.update(lexical("-1:", sentence[relative_index - 1]))
    if relative_index == len(sentence) - 1:
        features["EOS"] = True
    else:
        features.update(lexical("+1:", sentence[relative_index + 1]))
    return features


def build_bio_sequences(
    corpus: Corpus,
    *,
    include_labels: bool = True,
) -> SequenceDataset:
    """Convert reviews to sentence-level features and BIO labels."""

    feature_sequences: list[tuple[dict[str, object], ...]] = []
    label_sequences: list[tuple[str, ...]] = []
    keys: list[SentenceKey] = []
    products: list[str] = []
    token_sequences: list[tuple[Token, ...]] = []

    for review in corpus:
        for sentence_index, sentence in enumerate(review.sentences):
            labels = ["O"] * len(sentence)
            if include_labels:
                gold = review.annotations.get(sentence_index, ())
                spans = _gold_token_spans(sentence, [item.text for item in gold])
                for start, end in spans:
                    labels[start - sentence.start] = "B"
                    for token_index in range(start + 1, end):
                        labels[token_index - sentence.start] = "I"
            feature_sequences.append(
                tuple(_token_features(sentence, index) for index in range(len(sentence)))
            )
            label_sequences.append(tuple(labels))
            keys.append((review.uid, sentence_index))
            products.append(review.product.name)
            token_sequences.append(tuple(sentence))

    return SequenceDataset(
        features=tuple(feature_sequences),
        labels=tuple(label_sequences),
        keys=tuple(keys),
        product_names=tuple(products),
        tokens=tuple(token_sequences),
    )


def bio_alignment_diagnostic(corpus: Corpus) -> AlignmentReport:
    """Report how many gold phrases can supply unambiguous BIO supervision."""

    total = 0
    aligned = 0
    errors: list[AlignmentError] = []
    for review in corpus:
        for sentence_index, sentence in enumerate(review.sentences):
            annotations = review.annotations.get(sentence_index, ())
            aligned_annotations = _aligned_gold_annotations(sentence, annotations)
            total += len(aligned_annotations)
            for annotation, span in aligned_annotations:
                if span is not None:
                    aligned += 1
                else:
                    errors.append(
                        AlignmentError(
                            reason="unmatched_span",
                            review_uid=review.uid,
                            sentence_index=sentence_index,
                            aspect_text=annotation.text,
                            gold_polarity=annotation.polarity,
                        )
                    )
    return AlignmentReport(
        total_gold_aspects=total,
        aligned_aspects=aligned,
        errors=tuple(errors),
    )


def decode_bio_predictions(
    dataset: SequenceDataset,
    predictions: Sequence[Sequence[str]],
) -> tuple[Aspect, ...]:
    """Decode BIO label sequences into aspect spans."""

    aspects: list[Aspect] = []
    for key, product, tokens, labels in zip(
        dataset.keys,
        dataset.product_names,
        dataset.tokens,
        predictions,
        strict=True,
    ):
        start: int | None = None
        for index, label in enumerate((*labels, "O")):
            if label == "B" or (label == "I" and start is None):
                if start is not None:
                    span_tokens = tokens[start:index]
                    aspects.append(
                        _decoded_aspect(key, product, span_tokens)
                    )
                start = index
            elif label != "I" and start is not None:
                span_tokens = tokens[start:index]
                aspects.append(_decoded_aspect(key, product, span_tokens))
                start = None
    return _sort(aspects)


def _decoded_aspect(
    key: SentenceKey,
    product_name: str,
    tokens: Sequence[Token],
) -> Aspect:
    first, last = tokens[0], tokens[-1]
    return Aspect(
        review_uid=key[0],
        product_name=product_name,
        sentence_index=key[1],
        text=" ".join(token.text.lower() for token in tokens),
        start_token=first.i,
        end_token=last.i + 1,
        tokens=tuple(tokens),
    )


class CRFAspectExtractor:
    """Linear-chain CRF trained to identify aspect spans with BIO labels."""

    def __init__(
        self,
        *,
        c1: float = 0.1,
        c2: float = 0.1,
        max_iterations: int = 100,
    ) -> None:
        self.model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=c1,
            c2=c2,
            max_iterations=max_iterations,
            all_possible_transitions=True,
        )
        self._is_fitted = False

    def fit(self, corpus: Corpus) -> Self:
        dataset = build_bio_sequences(corpus)
        self.model.fit(dataset.features, dataset.labels)
        self._is_fitted = True
        return self

    def predict(self, corpus: Corpus) -> tuple[Aspect, ...]:
        if not self._is_fitted:
            raise RuntimeError("CRFAspectExtractor must be fitted before prediction")
        dataset = build_bio_sequences(corpus, include_labels=False)
        predictions = self.model.predict(dataset.features)
        return decode_bio_predictions(dataset, predictions)
