"""Domain objects shared across the opinion-mining pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from spacy.tokens import Doc, Span


SentenceKey: TypeAlias = tuple[str, int]


@dataclass(frozen=True)
class Product:
    """A product or domain within one source dataset."""

    dataset: str
    name: str


@dataclass(frozen=True)
class GoldAspect:
    """A sentence-level aspect annotation and its signed polarity."""

    text: str
    polarity: int


@dataclass
class Review:
    """One parsed review with stable provenance and linguistic annotations."""

    dataset: str
    uid: str
    review_index: int
    product: Product
    source_file: str
    annotated_text: str
    clean_text: str
    doc: Doc
    annotations: dict[int, tuple[GoldAspect, ...]]

    @property
    def sentences(self) -> tuple[Span, ...]:
        """Return the spaCy sentences in document order."""

        return tuple(self.doc.sents)


@dataclass(frozen=True)
class Corpus:
    """A collection of products and review records."""

    products: dict[tuple[str, str], Product]
    reviews: tuple[Review, ...]

    def __iter__(self) -> Iterator[Review]:
        return iter(self.reviews)

    def __len__(self) -> int:
        return len(self.reviews)


@dataclass(frozen=True)
class CorpusSplits:
    """Non-overlapping training, development and test partitions."""

    train: Corpus
    dev: Corpus
    test: Corpus
