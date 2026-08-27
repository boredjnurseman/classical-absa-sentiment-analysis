"""Pure runtime aspect--opinion linking with train-fitted PMI state."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Literal, Mapping, Self

from spacy.tokens import Token

from .aspects import Aspect
from .opinions import Opinion, OpinionLexicon
from .schema import Corpus, SentenceKey


LinkMethod = Literal["dep", "pmi", "dep->pmi", "pmi->dep"]
_METHODS = frozenset({"dep", "pmi", "dep->pmi", "pmi->dep"})


@dataclass(frozen=True)
class AspectOpinionLink:
    """One predicted aspect, opinion and polarity relationship."""

    review_uid: str
    product_name: str
    sentence_index: int
    aspect_text: str
    aspect_start_token: int
    aspect_end_token: int
    opinion_text: str
    opinion_lemma: str
    polarity: int


class PMIModel:
    """Sentence co-occurrence scores learned exclusively from training data."""

    def __init__(self, *, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.scores: dict[tuple[str, str], float] = {}
        self.aspect_counts: dict[str, int] = {}
        self.opinion_counts: dict[str, int] = {}
        self.pair_counts: dict[tuple[str, str], int] = {}
        self.sentence_count = 0
        self._is_fitted = False

    @classmethod
    def from_scores(
        cls,
        scores: Mapping[tuple[str, str], float],
        *,
        threshold: float = 0.0,
    ) -> Self:
        model = cls(threshold=threshold)
        model.scores = {
            (_normalise(aspect), _normalise(opinion)): float(score)
            for (aspect, opinion), score in scores.items()
        }
        model._is_fitted = True
        return model

    def fit(self, corpus: Corpus, lexicon: OpinionLexicon) -> Self:
        opinions = _group_opinions(lexicon.transform(corpus))
        aspect_counts: Counter[str] = Counter()
        opinion_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        sentence_count = 0

        for review in corpus:
            for sentence_index, _sentence in enumerate(review.sentences):
                sentence_count += 1
                aspect_heads = {
                    _aspect_key(annotation.text)
                    for annotation in review.annotations.get(sentence_index, ())
                    if annotation.text.strip()
                }
                opinion_lemmas = {
                    opinion.lemma
                    for opinion in opinions.get((review.uid, sentence_index), ())
                }
                aspect_counts.update(aspect_heads)
                opinion_counts.update(opinion_lemmas)
                pair_counts.update(
                    (aspect, opinion)
                    for aspect in aspect_heads
                    for opinion in opinion_lemmas
                )

        self.sentence_count = sentence_count
        self.aspect_counts = dict(aspect_counts)
        self.opinion_counts = dict(opinion_counts)
        self.pair_counts = dict(pair_counts)
        self.scores = {
            pair: math.log(
                (count * sentence_count)
                / (aspect_counts[pair[0]] * opinion_counts[pair[1]])
            )
            for pair, count in pair_counts.items()
            if sentence_count
        }
        self._is_fitted = True
        return self

    def score(self, aspect_text: str, opinion_lemma: str) -> float | None:
        if not self._is_fitted:
            raise RuntimeError("PMIModel must be fitted before scoring")
        return self.scores.get((_aspect_key(aspect_text), _normalise(opinion_lemma)))


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _aspect_key(text: str) -> str:
    words = _normalise(text).split()
    return words[-1] if words else ""


def _group_opinions(
    opinions: Iterable[Opinion],
) -> dict[SentenceKey, tuple[Opinion, ...]]:
    grouped: defaultdict[SentenceKey, list[Opinion]] = defaultdict(list)
    for opinion in opinions:
        if opinion.polarity == 0:
            continue
        grouped[(opinion.review_uid, opinion.sentence_index)].append(opinion)
    return {
        key: tuple(sorted(values, key=lambda item: item.token_index))
        for key, values in grouped.items()
    }


def _dependency_distance(left: Token, right: Token) -> int | None:
    def chain(token: Token) -> dict[int, int]:
        distances: dict[int, int] = {token.i: 0}
        current = token
        while current.head.i not in distances:
            distances[current.head.i] = distances[current.i] + 1
            if current.head.i == current.i:
                break
            current = current.head
        return distances

    left_chain = chain(left)
    right_chain = chain(right)
    common = set(left_chain) & set(right_chain)
    if not common:
        return None
    return min(left_chain[index] + right_chain[index] for index in common)


def _aspect_head(aspect: Aspect) -> Token:
    token_indexes = {token.i for token in aspect.tokens}
    external_heads = [
        token for token in aspect.tokens if token.head.i not in token_indexes
    ]
    return external_heads[-1] if external_heads else aspect.tokens[-1]


def _pmi_choice(
    aspect: Aspect,
    opinions: tuple[Opinion, ...],
    pmi: PMIModel,
) -> Opinion | None:
    scored = []
    for opinion in opinions:
        score = pmi.score(aspect.text, opinion.lemma)
        if score is None or score < pmi.threshold:
            continue
        scored.append((score, -abs(opinion.token_index - aspect.start_token), opinion))
    return max(scored, key=lambda item: (item[0], item[1], -item[2].token_index))[2] if scored else None


def _dependency_choice(
    aspect: Aspect,
    opinions: tuple[Opinion, ...],
    max_distance: int,
) -> Opinion | None:
    head = _aspect_head(aspect)
    candidates = []
    for opinion in opinions:
        distance = _dependency_distance(head, opinion.token)
        if distance is not None and distance <= max_distance:
            candidates.append((distance, opinion.token_index, opinion))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None


def link_aspects(
    aspects: Iterable[Aspect],
    opinions: Iterable[Opinion],
    *,
    pmi: PMIModel,
    method: LinkMethod,
    max_dep_distance: int = 4,
) -> tuple[AspectOpinionLink, ...]:
    """Link supplied predictions without access to annotations or corpus state."""

    if method not in _METHODS:
        raise ValueError(f"Unsupported linking method: {method}")
    if max_dep_distance < 0:
        raise ValueError("max_dep_distance cannot be negative")
    grouped_opinions = _group_opinions(opinions)
    links: list[AspectOpinionLink] = []

    for aspect in aspects:
        sentence_opinions = grouped_opinions.get(
            (aspect.review_uid, aspect.sentence_index), ()
        )
        dependency = lambda: _dependency_choice(  # noqa: E731
            aspect, sentence_opinions, max_dep_distance
        )
        statistical = lambda: _pmi_choice(aspect, sentence_opinions, pmi)  # noqa: E731
        if method == "dep":
            opinion = dependency()
        elif method == "pmi":
            opinion = statistical()
        elif method == "dep->pmi":
            opinion = dependency() or statistical()
        else:
            opinion = statistical() or dependency()
        if opinion is None:
            continue
        links.append(
            AspectOpinionLink(
                review_uid=aspect.review_uid,
                product_name=aspect.product_name,
                sentence_index=aspect.sentence_index,
                aspect_text=aspect.text,
                aspect_start_token=aspect.start_token,
                aspect_end_token=aspect.end_token,
                opinion_text=opinion.text,
                opinion_lemma=opinion.lemma,
                polarity=opinion.polarity,
            )
        )
    return tuple(
        sorted(
            links,
            key=lambda item: (
                item.review_uid,
                item.sentence_index,
                item.aspect_start_token,
                item.opinion_lemma,
            ),
        )
    )
