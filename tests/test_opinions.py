from spacy.language import Language
from spacy.tokens import Doc

from review_absa.opinions import OpinionLexicon
from review_absa.schema import Corpus, GoldAspect, Product, Review


def _review(
    nlp: Language,
    *,
    uid: str,
    words: list[str],
    pos: list[str],
    aspect: str,
    polarity: int,
) -> Review:
    product = Product("reviews", "Camera")
    doc = Doc(
        nlp.vocab,
        words=words,
        pos=pos,
        heads=[len(words) - 1] * len(words),
        deps=["dep"] * (len(words) - 1) + ["ROOT"],
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
        annotations={0: (GoldAspect(aspect, polarity),)},
    )


def test_lexicon_requires_minimum_training_count(nlp: Language) -> None:
    reviews = (
        _review(
            nlp,
            uid="r1",
            words=["excellent", "camera", "works"],
            pos=["ADJ", "NOUN", "VERB"],
            aspect="camera",
            polarity=1,
        ),
        _review(
            nlp,
            uid="r2",
            words=["excellent", "camera", "works"],
            pos=["ADJ", "NOUN", "VERB"],
            aspect="camera",
            polarity=1,
        ),
        _review(
            nlp,
            uid="r3",
            words=["rareword", "camera", "works"],
            pos=["ADJ", "NOUN", "VERB"],
            aspect="camera",
            polarity=1,
        ),
    )
    product = reviews[0].product
    corpus = Corpus(
        products={(product.dataset, product.name): product},
        reviews=reviews,
    )
    lexicon = OpinionLexicon(window=6, min_count=2).fit(corpus)
    assert lexicon.polarities["excellent"] == 1
    assert "rareword" not in lexicon.polarities


def test_negation_flips_known_polarity(nlp: Language) -> None:
    review = _review(
        nlp,
        uid="r1",
        words=["not", "good", "camera"],
        pos=["PART", "ADJ", "NOUN"],
        aspect="camera",
        polarity=-1,
    )
    corpus = Corpus(
        products={(review.product.dataset, review.product.name): review.product},
        reviews=(review,),
    )
    lexicon = OpinionLexicon.from_polarities({"good": 1}, negation_window=3)
    opinions = lexicon.transform(corpus)
    assert [(item.lemma, item.polarity) for item in opinions] == [("good", -1)]
