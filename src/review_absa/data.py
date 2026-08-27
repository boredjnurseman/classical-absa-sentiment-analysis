"""Parse Hu-Liu review files and construct leakage-safe corpus splits."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from spacy.language import Language

from .schema import Corpus, CorpusSplits, GoldAspect, Product, Review, SentenceKey


_DATASET_LAYOUTS = {
    "Customer_review_data": "titles",
    "Reviews-9-products": "titles",
    "CustomerReviews-3_domains": "lines",
}
_NUMERIC_TAG = re.compile(r"\[([+-]?\d+)\]")
_MALFORMED_POLARITY_TAG = re.compile(r"\[[+-]?\d+\}")
_UNSCORED_POLARITY_TAG = re.compile(r"\[[+-]\]")
_MARKER_TAG = re.compile(
    r"\[(?:t|r|p|u|s|cc|cs|v|a)\]", flags=re.IGNORECASE
)
_TRAILING_MARKERS = re.compile(
    r"(?:[ \t]*\[(?:t|r|p|u|s|cc|cs|v|a)\])+$", flags=re.IGNORECASE
)
_LEADING_MARKERS = re.compile(
    r"^[ \t]*(?:\[(?:t|r|p|u|s|cc|cs|v|a)\])+", flags=re.IGNORECASE
)
_TITLE_BLOCK = re.compile(
    r"(?ms)^\s*\[t\]\s*(?P<title>[^\n]*?)\s*(?:\n|$)"
    r"(?P<body>.*?)(?=^\s*\[t\]|\Z)"
)


@dataclass(frozen=True)
class _AnnotationMatch:
    aspect: str
    polarity: int
    aspect_start: int
    mask_start: int
    mask_end: int


def _marker_end(text: str, start: int) -> int:
    """Return the end of any marker tags attached to a polarity label.

    Hu--Liu files sometimes append tags such as ``[u]`` or ``[cc]`` after a
    numeric polarity.  They are annotation metadata rather than review text,
    so the masking boundary must include the complete run of adjacent tags.

    Args:
        text: Annotated source text.
        start: Character offset immediately after a numeric polarity tag.

    Returns:
        The first character after the adjacent marker sequence.
    """

    end = start
    while end < len(text):
        whitespace_end = end
        while whitespace_end < len(text) and text[whitespace_end] in " \t":
            whitespace_end += 1
        marker = _MARKER_TAG.match(text, whitespace_end)
        if marker is None:
            break
        end = marker.end()
    return end


def _last_sentence_boundary(raw: str) -> int | None:
    """Find a prose boundary before a trailing annotation clause.

    A few source lines contain a sentence followed by an annotation prefix
    before the next ``##`` delimiter.  Restricting this heuristic to a
    sentence-ending mark followed by whitespace keeps punctuation inside
    aspects such as ``camera.v2`` intact.

    Args:
        raw: Text between the line/separator boundary and a polarity tag.

    Returns:
        The end offset of the last plausible sentence boundary, or ``None``
        when the clause contains no such boundary.
    """

    boundary: int | None = None
    for match in re.finditer(r"(?<!\d)[.!?](?=\s+)", raw):
        if len(raw[: match.start()].split()) >= 3:
            boundary = match.end()
    return boundary


def _candidate_start(text: str, score_start: int, previous_end: int) -> int:
    """Locate the start of the annotation clause before one polarity tag.

    The usual ``##`` separator is preferred, but the raw corpora also contain
    single-hash, comma-separated, and occasionally missing-separator layouts.
    The previous numeric tag prevents a later tag from absorbing an earlier
    clause when several annotations share a line.

    Args:
        text: Complete annotated source string.
        score_start: Character offset of the current ``[+n]`` tag.
        previous_end: End offset of the preceding polarity tag.

    Returns:
        Character offset at which the current candidate clause begins.
    """

    line_start = text.rfind("\n", 0, score_start) + 1
    start = max(line_start, previous_end)
    separator = max(
        text.rfind("##", start, score_start),
        text.rfind("#", start, score_start),
    )
    if separator >= start:
        start = separator + (2 if text.startswith("##", separator) else 1)

    raw = text[start:score_start]
    boundary = _last_sentence_boundary(raw)
    if boundary is not None:
        start += boundary
    return start


def _aspect_text(text: str, start: int, score_start: int) -> tuple[str, int]:
    """Extract one aspect phrase from a structural annotation clause.

    Marker tags and malformed preceding polarity fragments are removed before
    comma-separated clauses are reduced to the phrase associated with the
    current numeric tag.  The returned offset is measured in the original
    string, which is important for sentence anchoring and leakage-safe
    masking.

    Args:
        text: Complete annotated source string.
        start: Start of the candidate clause.
        score_start: Offset of the current numeric polarity tag.

    Returns:
        A pair containing the normalised aspect text and its source offset.
    """

    original = text[start:score_start]
    raw = _TRAILING_MARKERS.sub("", original)
    raw = _MALFORMED_POLARITY_TAG.sub("", raw)
    raw = _UNSCORED_POLARITY_TAG.sub("", raw)
    raw = _MARKER_TAG.sub("", raw)
    if "," in raw:
        raw = raw.rsplit(",", maxsplit=1)[-1]
    raw = _LEADING_MARKERS.sub("", raw)
    raw = raw.lstrip(" \t,;|:")
    value = " ".join(raw.strip().split())
    value_start = original.find(raw)
    return value, start + (value_start if value_start >= 0 else 0)


def _annotation_matches(text: str) -> tuple[_AnnotationMatch, ...]:
    """Parse every numeric polarity tag into an offset-aware match.

    Parsing is deliberately structural rather than suffix-based: a valid
    aspect may begin with a digit, contain punctuation, or be longer than the
    common examples in the original files.  Each numeric tag still defines a
    distinct supervision record, including records in comma-separated clauses.

    Args:
        text: Annotated review text in a Hu--Liu layout.

    Returns:
        Matches containing aspect text, polarity, source offset, and the range
        that must be blanked before linguistic processing.
    """

    matches: list[_AnnotationMatch] = []
    previous_end = 0
    for score in _NUMERIC_TAG.finditer(text):
        start = _candidate_start(text, score.start(), previous_end)
        aspect, aspect_start = _aspect_text(text, start, score.start())
        end = _marker_end(text, score.end())
        matches.append(
            _AnnotationMatch(
                aspect=aspect.lower(),
                polarity=int(score.group(1)),
                aspect_start=aspect_start,
                mask_start=start,
                mask_end=end,
            )
        )
        previous_end = score.end()
    return tuple(matches)


def clean_text_length_preserving(text: str | None) -> str:
    """Mask annotation supervision while preserving every character offset.

    The model must see the review sentence but not the labelled aspect prefix.
    Replacing supervision with spaces, rather than deleting it, keeps the
    annotation offsets and spaCy character coordinates comparable to the raw
    string.  Numeric labels, marker tags, hash separators, and malformed label
    fragments are all treated as markup.

    Args:
        text: Annotated review text, or ``None`` for an empty input.

    Returns:
        A same-length string suitable for linguistic processing.
    """

    if text is None:
        return ""
    source = text.replace("\u00a0", " ")
    cleaned = list(source)

    def blank(start: int, end: int) -> None:
        for index in range(start, end):
            cleaned[index] = " "

    for match in _annotation_matches(source):
        # The parser also handles source lines where the usual ``##``
        # delimiter is missing.  The score marker still identifies the
        # preceding clause as supervision, so hide the whole clause in both
        # layouts without changing any character positions.
        blank(match.mask_start, match.mask_end)

    for pattern in (
        _NUMERIC_TAG,
        _MALFORMED_POLARITY_TAG,
        _UNSCORED_POLARITY_TAG,
        _MARKER_TAG,
    ):
        for match in pattern.finditer(source):
            blank(match.start(), match.end())
    for index, character in enumerate(source):
        if character in "#[]":
            cleaned[index] = " "
    return "".join(cleaned)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r", "")


def _parse_title_file(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    for index, match in enumerate(_TITLE_BLOCK.finditer(_read_text(path)), start=1):
        title = match.group("title").strip()
        body = match.group("body").strip()
        text = f"{title}\n{body}".strip() if title else body
        if text:
            rows.append((index, text))
    return rows or _parse_line_file(path)


def _parse_line_file(path: Path) -> list[tuple[int, str]]:
    lines = (line.strip() for line in _read_text(path).splitlines())
    return [(index, line) for index, line in enumerate(filter(None, lines), start=1)]


def _aspect_pairs(text: str) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (match.aspect, match.polarity, match.aspect_start)
        for match in _annotation_matches(text)
        if match.aspect
    )


def _sentence_annotations(text: str, doc) -> dict[int, tuple[GoldAspect, ...]]:
    grouped: defaultdict[int, list[GoldAspect]] = defaultdict(list)
    sentences = tuple(doc.sents)
    for aspect, polarity, start in _aspect_pairs(text):
        line_end = text.find("\n", start)
        if line_end == -1:
            line_end = len(text)
        separator = text.find("##", start, line_end)
        anchor = separator + 2 if separator != -1 else start
        while anchor < line_end and text[anchor].isspace():
            anchor += 1
        for sentence_index, sentence in enumerate(sentences):
            if sentence.start_char <= anchor < sentence.end_char:
                grouped[sentence_index].append(GoldAspect(aspect, polarity))
                break
    return {index: tuple(values) for index, values in grouped.items()}


def load_corpus(data_dir: Path | str, nlp: Language) -> Corpus:
    """Load the three supported Hu--Liu layouts from local files.

    Each source review is parsed, masked, and passed through the supplied
    spaCy pipeline.  The parser keeps the annotated source and clean model
    input together so later components can use the same stable review UID and
    sentence boundaries.

    Args:
        data_dir: Directory containing the three canonical corpus folders.
        nlp: Configured spaCy language pipeline used to parse clean text.

    Returns:
        A :class:`Corpus` containing product identities and parsed reviews.

    Raises:
        FileNotFoundError: If one of the required corpus folders is absent.
        ValueError: If the supplied directories contain no parseable reviews.
    """

    root = Path(data_dir)
    missing = [name for name in _DATASET_LAYOUTS if not (root / name).is_dir()]
    if missing:
        names = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing Hu-Liu dataset folders: {names}. See data/README.md."
        )

    products: dict[tuple[str, str], Product] = {}
    reviews: list[Review] = []
    for dataset, layout in _DATASET_LAYOUTS.items():
        parser = _parse_title_file if layout == "titles" else _parse_line_file
        for source_path in sorted((root / dataset).glob("*.txt")):
            if source_path.name.lower() == "readme.txt":
                continue
            product_key = (dataset, source_path.stem)
            product = products.setdefault(
                product_key,
                Product(dataset=dataset, name=source_path.stem),
            )
            for review_index, annotated_text in parser(source_path):
                clean_text = clean_text_length_preserving(annotated_text)
                doc = nlp(clean_text)
                uid = f"{dataset}::{source_path.name}::{review_index}"
                reviews.append(
                    Review(
                        dataset=dataset,
                        uid=uid,
                        review_index=review_index,
                        product=product,
                        source_file=source_path.name,
                        annotated_text=annotated_text,
                        clean_text=clean_text,
                        doc=doc,
                        annotations=_sentence_annotations(annotated_text, doc),
                    )
                )
    if not reviews:
        raise ValueError("No reviews were parsed from the supplied data directory.")
    return Corpus(products=products, reviews=tuple(reviews))


def gold_by_sentence(
    corpus: Corpus,
    *,
    polarity_as_sign: bool = False,
) -> dict[SentenceKey, tuple[GoldAspect, ...]]:
    """Return annotations keyed by stable review UID and sentence index.

    Args:
        corpus: Parsed reviews whose annotations should be materialised.
        polarity_as_sign: If true, collapse the source scale to ``+1`` and
            ``-1`` and omit neutral ``0`` annotations.

    Returns:
        A mapping from ``(review_uid, sentence_index)`` to immutable aspect
        records.  Empty sentence entries are omitted.
    """

    result: dict[SentenceKey, tuple[GoldAspect, ...]] = {}
    for review in corpus:
        for sentence_index, annotations in review.annotations.items():
            values = annotations
            if polarity_as_sign:
                values = tuple(
                    GoldAspect(item.text, 1 if item.polarity > 0 else -1)
                    for item in annotations
                    if item.polarity != 0
                )
            if values:
                result[(review.uid, sentence_index)] = values
    return result


def _subset(corpus: Corpus, selected: set[str]) -> Corpus:
    reviews = tuple(review for review in corpus if review.uid in selected)
    keys = {(review.dataset, review.product.name) for review in reviews}
    products = {key: corpus.products[key] for key in keys}
    return Corpus(products=products, reviews=reviews)


def split_corpus(corpus: Corpus, *, seed: int = 42) -> CorpusSplits:
    """Split reviews deterministically into review-level 60/20/20 partitions.

    Sorting UIDs before seeded shuffling makes the split independent of file
    enumeration order.  Partitioning complete reviews, rather than individual
    sentences, prevents near-duplicate sentences from crossing evaluation
    boundaries.

    Args:
        corpus: Corpus to partition.
        seed: Seed for the deterministic NumPy generator.

    Returns:
        A :class:`CorpusSplits` containing train, development, and test corpora.
    """

    uids = np.array(sorted(review.uid for review in corpus), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(uids)
    n_test = round(0.20 * len(uids))
    n_dev = round(0.20 * len(uids))
    test_uids = set(uids[:n_test])
    dev_uids = set(uids[n_test : n_test + n_dev])
    train_uids = set(uids[n_test + n_dev :])
    return CorpusSplits(
        train=_subset(corpus, train_uids),
        dev=_subset(corpus, dev_uids),
        test=_subset(corpus, test_uids),
    )
