"""Train-fitted opinion terms and lightweight polarity assignment."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Mapping, Self

from spacy.tokens import Token

from .aspects import _gold_token_spans
from .schema import Corpus


_NEGATORS = frozenset(
    {"not", "n't", "no", "never", "none", "hardly", "barely", "rarely"}
)


@dataclass(frozen=True)
class Opinion:
    """One opinion-bearing token with its context-adjusted polarity."""

    review_uid: str
    product_name: str
    sentence_index: int
    text: str
    lemma: str
    token_index: int
    polarity: int
    token: Token = field(compare=False, repr=False)


class OpinionLexicon:
    """Weakly supervised polarity lexicon fitted from training annotations."""

    def __init__(
        self,
        *,
        window: int = 6,
        min_count: int = 5,
        alpha: float = 1.0,
        negation_window: int = 3,
    ) -> None:
        if window < 0 or negation_window < 0:
            raise ValueError("token windows cannot be negative")
        if min_count < 1:
            raise ValueError("min_count must be at least 1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.window = window
        self.min_count = min_count
        self.alpha = alpha
        self.negation_window = negation_window
        self.polarities: dict[str, int] = {}
        self.counts: dict[str, tuple[int, int]] = {}
        self._is_fitted = False

    @classmethod
    def from_polarities(
        cls,
        polarities: Mapping[str, int],
        *,
        negation_window: int = 3,
    ) -> Self:
        model = cls(min_count=1, negation_window=negation_window)
        model.polarities = {
            lemma.lower(): int(polarity) for lemma, polarity in polarities.items()
        }
        model.counts = {lemma: (0, 0) for lemma in model.polarities}
        model._is_fitted = True
        return model

    def fit(self, corpus: Corpus) -> Self:
        positive: Counter[str] = Counter()
        negative: Counter[str] = Counter()
        for review in corpus:
            for sentence_index, sentence in enumerate(review.sentences):
                annotations = review.annotations.get(sentence_index, ())
                for annotation in annotations:
                    if annotation.polarity == 0:
                        continue
                    aligned = _gold_token_spans(sentence, [annotation.text])
                    if not aligned:
                        continue
                    aspect_start, aspect_end = aligned[0]
                    for token in sentence:
                        if token.pos_ not in {"ADJ", "VERB"}:
                            continue
                        if aspect_start <= token.i < aspect_end:
                            continue
                        distance = min(
                            abs(token.i - aspect_start),
                            abs(token.i - (aspect_end - 1)),
                        )
                        if distance > self.window:
                            continue
                        lemma = _lemma(token)
                        if annotation.polarity > 0:
                            positive[lemma] += 1
                        else:
                            negative[lemma] += 1

        vocabulary = sorted(set(positive) | set(negative))
        self.counts = {
            lemma: (positive[lemma], negative[lemma])
            for lemma in vocabulary
            if positive[lemma] + negative[lemma] >= self.min_count
        }
        self.polarities = {}
        for lemma, (positive_count, negative_count) in self.counts.items():
            score = math.log(
                (positive_count + self.alpha) / (negative_count + self.alpha)
            )
            self.polarities[lemma] = 1 if score > 0 else -1 if score < 0 else 0
        self._is_fitted = True
        return self

    def transform(self, corpus: Corpus) -> tuple[Opinion, ...]:
        if not self._is_fitted:
            raise RuntimeError("OpinionLexicon must be fitted before transformation")
        opinions: list[Opinion] = []
        for review in corpus:
            for sentence_index, sentence in enumerate(review.sentences):
                for relative_index, token in enumerate(sentence):
                    if token.pos_ not in {"ADJ", "VERB"}:
                        continue
                    lemma = _lemma(token)
                    if lemma not in self.polarities:
                        continue
                    polarity = self.polarities[lemma]
                    left = sentence[
                        max(0, relative_index - self.negation_window) : relative_index
                    ]
                    if polarity and any(_lemma(item) in _NEGATORS for item in left):
                        polarity *= -1
                    opinions.append(
                        Opinion(
                            review_uid=review.uid,
                            product_name=review.product.name,
                            sentence_index=sentence_index,
                            text=token.text,
                            lemma=lemma,
                            token_index=token.i,
                            polarity=polarity,
                            token=token,
                        )
                    )
        return tuple(
            sorted(
                opinions,
                key=lambda item: (
                    item.review_uid,
                    item.sentence_index,
                    item.token_index,
                ),
            )
        )


def _lemma(token: Token) -> str:
    return (token.lemma_ or token.text).lower().strip()
