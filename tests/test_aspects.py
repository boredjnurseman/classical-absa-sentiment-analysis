from spacy.language import Language
from spacy.tokens import Doc

from review_absa.aspects import (
    ATEBaselineSuite,
    CRFAspectExtractor,
    bio_alignment_diagnostic,
    build_bio_sequences,
    decode_bio_predictions,
    extract_aspect_candidates,
)
from review_absa.schema import Corpus, GoldAspect, Product, Review


def _review(
    nlp: Language,
    *,
    uid: str,
    product: Product,
    words: list[str],
    heads: list[int],
    deps: list[str],
    pos: list[str],
) -> Review:
    doc = Doc(
        nlp.vocab,
        words=words,
        heads=heads,
        deps=deps,
        pos=pos,
        sent_starts=[True] + [False] * (len(words) - 1),
    )
    return Review(
        dataset=product.dataset,
        uid=uid,
        review_index=1,
        product=product,
        source_file="Camera.txt",
        annotated_text=" ".join(words),
        clean_text=" ".join(words),
        doc=doc,
        annotations={},
    )


def _corpus(nlp: Language, terms: list[tuple[str, str]]) -> Corpus:
    product = Product("reviews", "Camera")
    reviews = []
    for index, (modifier, noun) in enumerate(terms, start=1):
        reviews.append(
            _review(
                nlp,
                uid=f"r{index}",
                product=product,
                words=[modifier, noun, "works", "."],
                heads=[1, 2, 2, 2],
                deps=["compound", "nsubj", "ROOT", "punct"],
                pos=["NOUN", "NOUN", "VERB", "PUNCT"],
            )
        )
    return Corpus(products={(product.dataset, product.name): product}, reviews=tuple(reviews))


def _labelled_corpus(nlp: Language) -> Corpus:
    corpus = _corpus(
        nlp,
        [("battery", "life"), ("picture", "quality"), ("camera", "lens")],
    )
    for review in corpus:
        aspect = " ".join(token.text.lower() for token in review.doc[:2])
        review.annotations = {0: (GoldAspect(aspect, 1),)}
    return corpus


def test_candidate_extraction_recovers_noun_chunk(nlp: Language) -> None:
    corpus = _corpus(nlp, [("battery", "life")])
    candidates = extract_aspect_candidates(corpus)
    assert [item.text for item in candidates] == ["battery life"]


def test_baselines_publish_all_five_regimes(nlp: Language) -> None:
    train = _corpus(nlp, [("battery", "life"), ("battery", "life"), ("picture", "quality")])
    dev = _corpus(nlp, [("battery", "life"), ("picture", "quality")])
    result = ATEBaselineSuite.fit(train, top_k=1).transform(dev)
    assert tuple(result) == (
        "raw",
        "linguistic",
        "statistical",
        "linguistic_then_statistical",
        "statistical_then_linguistic",
    )
    assert [item.text for item in result["statistical"]] == ["battery life"]


def test_bio_decoder_recovers_multiword_span(nlp: Language) -> None:
    corpus = _labelled_corpus(nlp)
    dataset = build_bio_sequences(corpus)
    aspects = decode_bio_predictions(dataset, dataset.labels)
    assert aspects[0].text == "battery life"
    assert aspects[0].end_token - aspects[0].start_token == 2


def test_alignment_diagnostic_counts_contiguous_gold_spans(nlp: Language) -> None:
    report = bio_alignment_diagnostic(_labelled_corpus(nlp))
    assert report.total_gold_aspects == 3
    assert report.aligned_aspects == 3
    assert report.match_rate == 1.0
    assert report.reason_counts == {}
    assert report.errors == ()


def test_alignment_diagnostic_records_unmatched_spans(nlp: Language) -> None:
    corpus = _labelled_corpus(nlp)
    corpus.reviews[0].annotations = {0: (GoldAspect("missing camera control", 1),)}
    report = bio_alignment_diagnostic(corpus)
    assert report.reason_counts == {"unmatched_span": 1}
    assert len(report.errors) == 1
    assert report.errors[0].reason == "unmatched_span"
    assert report.errors[0].review_uid == "r1"
    assert report.errors[0].aspect_text == "missing camera control"


def test_crf_fits_and_predicts_synthetic_corpus(nlp: Language) -> None:
    corpus = _labelled_corpus(nlp)
    model = CRFAspectExtractor(c1=0.1, c2=0.1, max_iterations=25)
    assert model.fit(corpus) is model
    assert isinstance(model.predict(corpus), tuple)
