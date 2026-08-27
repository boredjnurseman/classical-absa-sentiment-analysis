from pathlib import Path

import pytest
import spacy
from spacy.language import Language


@pytest.fixture
def nlp() -> Language:
    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    return pipeline


@pytest.fixture
def sample_data_root(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    five_product = root / "Customer_review_data"
    nine_product = root / "Reviews-9-products"
    three_domain = root / "CustomerReviews-3_domains"
    for directory in (five_product, nine_product, three_domain):
        directory.mkdir(parents=True)

    (five_product / "Camera.txt").write_text(
        "[t] First camera\nbattery life[+2]##battery life is excellent.\n"
        "[t] Second camera\npicture quality[-1]##picture quality is poor.\n"
        "[t] Third camera\nlens[+1]##the lens is sharp.\n",
        encoding="utf-8",
    )
    (nine_product / "Phone.txt").write_text(
        "[t] First phone\nscreen[+1]##the screen looks bright.\n"
        "[t] Second phone\nspeaker[-2]##the speaker sounds weak.\n"
        "[t] Third phone\nsize[+1]##the size feels compact.\n",
        encoding="utf-8",
    )
    (three_domain / "Router.txt").write_text(
        "signal[+1]##the signal is reliable.\n"
        "setup[-1]##the setup is difficult.\n"
        "speed[+2]##the speed is excellent.\n"
        "range[-1]##the range is limited.\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def annotation_edge_data_root(tmp_path: Path) -> Path:
    """A small corpus containing the annotation irregularities seen in Hu--Liu."""

    root = tmp_path / "raw"
    five_product = root / "Customer_review_data"
    nine_product = root / "Reviews-9-products"
    three_domain = root / "CustomerReviews-3_domains"
    for directory in (five_product, nine_product, three_domain):
        directory.mkdir(parents=True)

    (five_product / "Camera.txt").write_text(
        "[t] Structured camera review\n"
        "1-800[+2], camera.v2[-1], operating-system version 2.0[+3], "
        "user's guide[+1][u]##the camera is useful.\n"
        "LCD[+3]camera quality[+3]##the display is clear.\n"
        "size[cs][-2]##the size is awkward.\n"
        "very long aspect phrase with seven words[-3]##the phrase is poor.\n",
        encoding="utf-8",
    )
    (nine_product / "Phone.txt").write_text(
        "[t] Punctuated phone review\n"
        "McAfee Anti-Virus 8[+2]]##the software works.\n",
        encoding="utf-8",
    )
    (three_domain / "Router.txt").write_text(
        "setup[+1], Wi-Fi/6[-1][v] ##the setup is clear.\n",
        encoding="utf-8",
    )
    return root
