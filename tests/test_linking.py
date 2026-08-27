import inspect

from spacy.language import Language
from spacy.tokens import Doc

from review_absa.aspects import Aspect
from review_absa.linking import PMIModel, link_aspects
from review_absa.opinions import Opinion


def _records(nlp: Language) -> tuple[Aspect, tuple[Opinion, Opinion]]:
    doc = Doc(
        nlp.vocab,
        words=["battery", "is", "excellent", "but", "heavy"],
        pos=["NOUN", "AUX", "ADJ", "CCONJ", "ADJ"],
        heads=[2, 2, 2, 4, 2],
        deps=["nsubj", "cop", "ROOT", "cc", "conj"],
        sent_starts=[True, False, False, False, False],
    )
    aspect = Aspect("r1", "Camera", 0, "battery", 0, 1, (doc[0],))
    opinions = (
        Opinion("r1", "Camera", 0, "excellent", "excellent", 2, 1, doc[2]),
        Opinion("r1", "Camera", 0, "heavy", "heavy", 4, -1, doc[4]),
    )
    return aspect, opinions


def test_pmi_linker_chooses_highest_available_pair(nlp: Language) -> None:
    aspect, opinions = _records(nlp)
    pmi = PMIModel.from_scores(
        {("battery", "excellent"): 1.2, ("battery", "heavy"): 0.4}
    )
    links = link_aspects((aspect,), opinions, pmi=pmi, method="pmi")
    assert links[0].opinion_lemma == "excellent"


def test_dependency_linker_chooses_shortest_path(nlp: Language) -> None:
    aspect, opinions = _records(nlp)
    links = link_aspects(
        (aspect,), opinions, pmi=PMIModel.from_scores({}), method="dep"
    )
    assert links[0].opinion_lemma == "excellent"


def test_ordered_fallbacks_respect_primary_method(nlp: Language) -> None:
    aspect, opinions = _records(nlp)
    pmi = PMIModel.from_scores({("battery", "heavy"): 2.0})
    dep_first = link_aspects(
        (aspect,), opinions, pmi=pmi, method="dep->pmi"
    )
    pmi_first = link_aspects(
        (aspect,), opinions, pmi=pmi, method="pmi->dep"
    )
    assert dep_first[0].opinion_lemma == "excellent"
    assert pmi_first[0].opinion_lemma == "heavy"


def test_runtime_linker_cannot_read_gold_annotations() -> None:
    signature = inspect.signature(link_aspects)
    assert "corpus" not in signature.parameters
    assert "gold" not in signature.parameters


def test_zero_polarity_opinions_do_not_create_deployable_links(nlp: Language) -> None:
    aspect, _ = _records(nlp)
    token = aspect.tokens[0]
    neutral = Opinion("r1", "Camera", 0, "fine", "fine", 2, 0, token)
    links = link_aspects(
        (aspect,),
        (neutral,),
        pmi=PMIModel.from_scores({("battery", "fine"): 2.0}),
        method="pmi",
    )
    assert links == ()
