"""Domain objects shared across the opinion-mining pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from spacy.tokens import Doc, Span


SentenceKey: TypeAlias = tuple[str, int]


@dataclass(frozen=True)
class Product:
    """Identify a product or domain within one source dataset.

    Attributes:
        dataset: Canonical directory name for the source corpus.
        name: Product name derived from the source filename.
    """

    dataset: str
    name: str


@dataclass(frozen=True)
class GoldAspect:
    """Represent one sentence-level annotated aspect.

    Attributes:
        text: Surface form supplied by the Hu--Liu annotation.
        polarity: Original signed score, normally in the range ``[-3, 3]``.
            A value of zero denotes a neutral annotation and is excluded from
            deployable polarity links.
    """

    text: str
    polarity: int


@dataclass
class Review:
    """Store one parsed review and its aligned linguistic representation.

    ``annotated_text`` preserves the source representation for supervision;
    ``clean_text`` has the annotation markup replaced with spaces so that all
    remaining character offsets still refer to the same positions.  The
    spaCy ``doc`` is produced from ``clean_text`` and ``annotations`` maps
    sentence indexes to the corresponding gold aspect records.

    Attributes:
        dataset: Canonical corpus directory containing the review.
        uid: Stable review identifier used for split isolation.
        review_index: One-based position in the source file.
        product: Product or domain associated with the source file.
        source_file: Original filename, retained for provenance.
        annotated_text: Raw review text including inline labels.
        clean_text: Same-length model input with labels and markup masked.
        doc: spaCy document parsed from ``clean_text``.
        annotations: Gold aspects grouped by spaCy sentence index.
    """

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
        """Return sentence spans in the order produced by spaCy.

        Returns:
            An immutable tuple of sentence spans backed by ``doc``.
        """

        return tuple(self.doc.sents)


@dataclass(frozen=True)
class Corpus:
    """Represent the parsed corpus used by every pipeline component.

    Attributes:
        products: Product identities keyed by ``(dataset, name)``.
        reviews: Parsed reviews in deterministic source order.
    """

    products: dict[tuple[str, str], Product]
    reviews: tuple[Review, ...]

    def __iter__(self) -> Iterator[Review]:
        return iter(self.reviews)

    def __len__(self) -> int:
        return len(self.reviews)


@dataclass(frozen=True)
class CorpusSplits:
    """Hold the non-overlapping train, development and test partitions.

    The split is performed at review level rather than sentence level.  This
    prevents sentences from a single review from leaking across the fitted,
    selected, and held-out stages of the experiment.

    Attributes:
        train: Reviews used to fit vocabularies and models.
        dev: Reviews used to select a linker and inspect development metrics.
        test: Reviews used once for the final held-out evaluation.
    """

    train: Corpus
    dev: Corpus
    test: Corpus
