from pathlib import Path

import pytest
from spacy.language import Language

from review_absa.data import (
    _aspect_pairs,
    clean_text_length_preserving,
    gold_by_sentence,
    load_corpus,
    split_corpus,
)


def test_cleaning_preserves_annotation_offsets() -> None:
    text = "battery life[+2]##battery life is not bad"
    cleaned = clean_text_length_preserving(text)
    assert len(cleaned) == len(text)
    assert cleaned.index("battery life") == text.rindex("battery life")
    assert "[+2]" not in cleaned


def test_cleaning_hides_inline_annotation_prefix_from_model_input() -> None:
    text = "battery life[+2]##The battery life is excellent."
    cleaned = clean_text_length_preserving(text)
    assert len(cleaned) == len(text)
    assert cleaned.count("battery life") == 1
    assert cleaned.rstrip().endswith("The battery life is excellent.")


def test_load_corpus_parses_all_three_layouts(
    sample_data_root: Path,
    nlp: Language,
) -> None:
    corpus = load_corpus(sample_data_root, nlp)
    assert len(corpus) == 10
    assert {review.dataset for review in corpus} == {
        "Customer_review_data",
        "Reviews-9-products",
        "CustomerReviews-3_domains",
    }
    assert len({review.uid for review in corpus}) == 10


def test_gold_annotations_are_sentence_scoped(
    sample_data_root: Path,
    nlp: Language,
) -> None:
    corpus = load_corpus(sample_data_root, nlp)
    gold = gold_by_sentence(corpus, polarity_as_sign=True)
    assert sum(len(aspects) for aspects in gold.values()) == 10
    assert {item.polarity for aspects in gold.values() for item in aspects} == {-1, 1}


def test_split_is_deterministic_and_disjoint(
    sample_data_root: Path,
    nlp: Language,
) -> None:
    corpus = load_corpus(sample_data_root, nlp)
    first = split_corpus(corpus, seed=42)
    second = split_corpus(corpus, seed=42)
    assert [review.uid for review in first.train] == [review.uid for review in second.train]
    uid_sets = [
        {review.uid for review in part}
        for part in (first.train, first.dev, first.test)
    ]
    assert [len(part) for part in uid_sets] == [6, 2, 2]
    assert not (
        uid_sets[0] & uid_sets[1]
        or uid_sets[0] & uid_sets[2]
        or uid_sets[1] & uid_sets[2]
    )


def test_missing_dataset_folder_points_to_data_instructions(
    tmp_path: Path,
    nlp: Language,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"data/README\.md"):
        load_corpus(tmp_path, nlp)


def test_title_layout_falls_back_to_lines_when_markers_are_absent(
    sample_data_root: Path,
    nlp: Language,
) -> None:
    path = sample_data_root / "Reviews-9-products" / "Untitled.txt"
    path.write_text(
        "sound[+1] is clear.\nbattery[-1] is weak.\n",
        encoding="utf-8",
    )
    corpus = load_corpus(sample_data_root, nlp)
    reviews = [review for review in corpus if review.source_file == "Untitled.txt"]
    assert len(reviews) == 2
    assert {review.product.name for review in reviews} == {"Untitled"}


def test_structural_parser_keeps_all_realistic_annotation_prefixes(
    annotation_edge_data_root: Path,
    nlp: Language,
) -> None:
    corpus = load_corpus(annotation_edge_data_root, nlp)
    labels = [
        annotation
        for review in corpus
        for annotations in review.annotations.values()
        for annotation in annotations
    ]
    assert len(labels) == 11
    assert [annotation.text for annotation in labels] == [
        "1-800",
        "camera.v2",
        "operating-system version 2.0",
        "user's guide",
        "lcd",
        "camera quality",
        "size",
        "very long aspect phrase with seven words",
        "mcafee anti-virus 8",
        "setup",
        "wi-fi/6",
    ]


def test_cleaning_blanks_the_entire_structural_prefix_without_shifting_offsets() -> None:
    text = (
        "1-800[+2], camera.v2[-1][u]##The 1-800 camera.v2 is useful."
    )
    cleaned = clean_text_length_preserving(text)
    boundary = text.index("##") + 2
    assert len(cleaned) == len(text)
    assert cleaned[:boundary] == " " * boundary
    assert cleaned[boundary:] == text[boundary:]


def test_cleaning_blanks_valid_prefixes_even_when_hash_delimiter_is_missing() -> None:
    text = "touch pad[-2], design[-1], volume control[-1] Weak Points:"
    cleaned = clean_text_length_preserving(text)
    boundary = text.index("Weak Points")
    assert len(cleaned) == len(text)
    assert cleaned[:boundary] == " " * boundary
    assert cleaned[boundary:] == text[boundary:]


def test_structural_parser_discards_malformed_prior_tags() -> None:
    assert [
        (aspect, polarity) for aspect, polarity, _start in _aspect_pairs(
            "USB[+2}, charger[+1]##the charger works."
        )
    ] == [("charger", 1)]


def test_structural_parser_preserves_decimal_aspect_text() -> None:
    assert _aspect_pairs("foo bar version 2. camera[+1]##the camera works.")[0][0] == (
        "foo bar version 2. camera"
    )
