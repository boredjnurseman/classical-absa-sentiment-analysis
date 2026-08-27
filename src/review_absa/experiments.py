"""Reproducible experiment orchestration and validated artifact publication."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Mapping
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import pandas as pd
import spacy

from .aspects import (
    ATEBaselineSuite,
    CRFAspectExtractor,
    aspects_by_sentence,
    bio_alignment_diagnostic,
)
from .data import gold_by_sentence, load_corpus, split_corpus
from .evaluation import (
    JointMetrics,
    SpanMetrics,
    evaluate_aspect_polarity,
    evaluate_aspects,
    evaluate_linking_component,
)
from .linking import LinkMethod, PMIModel, link_aspects
from .opinions import OpinionLexicon
from .plots import plot_ate_comparison, plot_linking_tradeoff, plot_product_summary
from .schema import Corpus
from .summaries import make_product_summary


_TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "corpus_summary.csv": (
        "dataset", "n_reviews", "n_files", "n_products", "n_labels",
        "positive_labels", "negative_labels",
    ),
    "ate_baselines.csv": (
        "split", "pipeline", "match", "precision", "recall", "f1",
        "true_positive", "false_positive", "false_negative", "predicted", "gold",
    ),
    "crf_metrics.csv": (
        "split", "match", "precision", "recall", "f1", "true_positive",
        "false_positive", "false_negative", "predicted", "gold",
    ),
    "negation_ablation.csv": (
        "split", "method", "negation", "coverage", "accuracy", "macro_f1",
        "n_total", "n_covered",
    ),
    "linking_metrics.csv": (
        "split", "method", "coverage", "accuracy", "macro_f1", "n_total",
        "n_covered",
    ),
    "end_to_end_metrics.csv": (
        "split", "linker", "match", "precision", "recall", "f1",
        "true_positive", "false_positive", "false_negative", "predicted", "gold",
        "linked_aspect_coverage",
    ),
    "product_summaries.csv": (
        "product", "aspect", "positive", "negative", "total", "count_mode",
    ),
}
_FIGURE_NAMES = (
    "ate_comparison.png",
    "linking_tradeoff.png",
    "product_summaries.png",
)
_LINK_METHODS: tuple[LinkMethod, ...] = ("dep", "pmi", "dep->pmi", "pmi->dep")
_FORBIDDEN_ARTIFACT_TEXT = ("/Users/", "/home/", "/content/drive", "::")


class ArtifactValidationError(ValueError):
    """Signal that a result set cannot satisfy the public artifact contract."""


@dataclass(frozen=True)
class RunConfig:
    """Capture all settings needed to reproduce one experiment.

    Paths identify local input/output locations and are deliberately omitted
    from the published manifest.  The remaining values define the seed,
    feature vocabulary, opinion window, dependency cutoff, and spaCy model.
    """

    data_dir: Path
    output_dir: Path
    seed: int = 42
    top_k: int = 200
    window: int = 6
    min_count: int = 5
    max_dep_distance: int = 4
    spacy_model: str = "en_core_web_sm"


@dataclass(frozen=True)
class RunResult:
    """Return the selected deployment method and published manifest.

    Attributes:
        selected_linker: Method chosen using development-only metrics.
        output_dir: Directory containing the complete result set.
        manifest: Aggregate run metadata and artifact checksums.
    """

    selected_linker: str
    output_dir: Path
    manifest: Mapping[str, object]


def select_linker(rows: pd.DataFrame) -> str:
    """Select a linker by macro-F1, then coverage, then method name.

    The stable lexical tie-break makes selection deterministic while keeping
    macro-F1 as the declared primary objective.

    Args:
        rows: Development linker metrics with ``method``, ``macro_f1``, and
            ``coverage`` columns.

    Returns:
        Name of the highest-ranked linker.

    Raises:
        ValueError: If no rows exist or required columns are missing.
    """

    required = {"method", "macro_f1", "coverage"}
    if rows.empty or not required.issubset(rows.columns):
        raise ValueError("linker rows require method, macro_f1 and coverage")
    ranked = rows.sort_values(
        ["macro_f1", "coverage", "method"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    return str(ranked.iloc[0]["method"])


def _validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    if set(tables) != set(_TABLE_SCHEMAS):
        missing = sorted(set(_TABLE_SCHEMAS) - set(tables))
        extra = sorted(set(tables) - set(_TABLE_SCHEMAS))
        raise ArtifactValidationError(
            f"Artifact table set mismatch; missing={missing}, extra={extra}"
        )
    for name, expected in _TABLE_SCHEMAS.items():
        actual = tuple(tables[name].columns)
        if actual != expected:
            raise ArtifactValidationError(
                f"{name} columns must be {expected}; received {actual}"
            )
        strings = tables[name].select_dtypes(include=["object", "string"])
        for value in strings.fillna("").astype(str).to_numpy().ravel():
            if any(marker in value for marker in _FORBIDDEN_ARTIFACT_TEXT):
                raise ArtifactValidationError(f"Unsafe text found in {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish_artifacts(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path | str,
    *,
    figures: Mapping[str, Figure] | None = None,
    manifest: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Validate and atomically publish one complete aggregate result set.

    Tables, figures, and the manifest are written beneath a staging directory
    first.  A directory rename then swaps the finished set into place; if
    either rename fails, the previous complete set is restored.  This prevents
    readers from observing a mixture of old and new artifacts.

    Args:
        tables: Mapping containing exactly the public CSV schemas.
        output_dir: Destination directory for the result set.
        figures: Exactly one figure for each declared public PNG.
        manifest: Additional aggregate metadata to include in the manifest.

    Returns:
        SHA-256 hashes for the published tables and figures.

    Raises:
        ArtifactValidationError: If schemas, figure names, or public text are
            unsafe or incomplete.
    """

    _validate_tables(tables)
    if figures is None or set(figures) != set(_FIGURE_NAMES):
        raise ArtifactValidationError(
            f"Figures must contain exactly {list(_FIGURE_NAMES)}"
        )

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{uuid4().hex}"
    staged_result = staging / "result"
    staged_figures = staged_result / "figures"
    staged_figures.mkdir(parents=True)
    try:
        for name, frame in tables.items():
            frame.to_csv(staged_result / name, index=False)
        for name, figure in figures.items():
            figure.savefig(staged_figures / name, dpi=160, bbox_inches="tight")

        published_files = [*sorted(_TABLE_SCHEMAS)] + [
            f"figures/{name}" for name in _FIGURE_NAMES
        ]
        hashes = {name: _sha256(staged_result / name) for name in published_files}
        complete_manifest = dict(manifest or {})
        complete_manifest.update({"status": "complete", "artifact_sha256": hashes})
        (staged_result / "run_manifest.json").write_text(
            json.dumps(complete_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        backup = output.parent / f".{output.name}.backup-{uuid4().hex}"
        previous_exists = output.exists()
        previous_moved = False
        try:
            if previous_exists:
                output.replace(backup)
                previous_moved = True
            staged_result.replace(output)
        except BaseException:
            # The directory swap is the transaction boundary. If either
            # rename fails, put the previous complete result set back before
            # allowing the publication error to escape.
            if output.exists() and (previous_moved or not previous_exists):
                failed_result = output.parent / f".{output.name}.failed-{uuid4().hex}"
                output.replace(failed_result)
                shutil.rmtree(failed_result)
            if previous_moved and backup.exists():
                backup.replace(output)
            raise
        else:
            if backup.exists():
                shutil.rmtree(backup)
        return hashes
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _span_row(split: str, match: str, metrics: SpanMetrics) -> dict[str, object]:
    return {
        "split": split,
        "match": match,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "predicted": metrics.predicted,
        "gold": metrics.gold,
    }


def _joint_row(
    linker: str,
    match: str,
    metrics: JointMetrics,
) -> dict[str, object]:
    return {
        "split": "test",
        "linker": linker,
        "match": match,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "predicted": metrics.true_positive + metrics.false_positive,
        "gold": metrics.true_positive + metrics.false_negative,
        "linked_aspect_coverage": metrics.linked_aspect_coverage,
    }


def _corpus_summary(corpus: Corpus) -> pd.DataFrame:
    rows = []
    for dataset in sorted({review.dataset for review in corpus}):
        reviews = [review for review in corpus if review.dataset == dataset]
        labels = [
            annotation.polarity
            for review in reviews
            for annotations in review.annotations.values()
            for annotation in annotations
            if annotation.polarity != 0
        ]
        rows.append(
            {
                "dataset": dataset,
                "n_reviews": len(reviews),
                "n_files": len({review.source_file for review in reviews}),
                "n_products": len({review.product.name for review in reviews}),
                "n_labels": len(labels),
                "positive_labels": sum(value > 0 for value in labels),
                "negative_labels": sum(value < 0 for value in labels),
            }
        )
    return pd.DataFrame(rows, columns=_TABLE_SCHEMAS["corpus_summary.csv"])


def _data_checksums(data_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(data_dir)): _sha256(path)
        for path in sorted(data_dir.rglob("*.txt"))
    }


def run_experiment(config: RunConfig) -> RunResult:
    """Run the leakage-safe train/select/test pipeline once.

    Models, vocabularies, opinion polarity, and PMI state are fitted on the
    training split.  Development metrics select the linker; the frozen choice
    is then evaluated on held-out test reviews and used to build predicted-
    aspect product summaries.

    Args:
        config: Reproducibility settings and local data/artifact paths.

    Returns:
        The selected linker and manifest for the atomically published result.
    """

    started = time.perf_counter()
    nlp = spacy.load(config.spacy_model)
    corpus = load_corpus(config.data_dir, nlp)
    splits = split_corpus(corpus, seed=config.seed)

    baseline = ATEBaselineSuite.fit(splits.train, top_k=config.top_k)
    baseline_rows = []
    dev_gold = gold_by_sentence(splits.dev)
    for pipeline, aspects in baseline.transform(splits.dev).items():
        predicted = aspects_by_sentence(aspects)
        for match in ("exact", "head"):
            row = _span_row(
                "dev", match, evaluate_aspects(predicted, dev_gold, match=match)
            )
            baseline_rows.append({"pipeline": pipeline, **row})
    baseline_frame = pd.DataFrame(
        baseline_rows, columns=_TABLE_SCHEMAS["ate_baselines.csv"]
    )

    crf = CRFAspectExtractor().fit(splits.train)
    dev_crf_aspects = crf.predict(splits.dev)
    dev_crf_predictions = aspects_by_sentence(dev_crf_aspects)
    crf_rows = [
        _span_row(
            "dev", match, evaluate_aspects(dev_crf_predictions, dev_gold, match=match)
        )
        for match in ("exact", "head")
    ]

    lexicon = OpinionLexicon(
        window=config.window,
        min_count=config.min_count,
        negation_window=3,
    ).fit(splits.train)
    no_negation = OpinionLexicon.from_polarities(
        lexicon.polarities, negation_window=0
    )
    pmi = PMIModel().fit(splits.train, lexicon)
    linking_rows = []
    negation_rows = []
    for negation, opinion_model in (("enabled", lexicon), ("disabled", no_negation)):
        dev_opinions = opinion_model.transform(splits.dev)
        for method in _LINK_METHODS:
            metrics = evaluate_linking_component(
                splits.dev,
                dev_opinions,
                pmi,
                method,
                max_dep_distance=config.max_dep_distance,
            )
            row = {
                "split": "dev",
                "method": method,
                "coverage": metrics.coverage,
                "accuracy": metrics.accuracy,
                "macro_f1": metrics.macro_f1,
                "n_total": metrics.total,
                "n_covered": metrics.covered,
            }
            negation_rows.append({**row, "negation": negation})
            if negation == "enabled":
                linking_rows.append(row)
    linking_frame = pd.DataFrame(
        linking_rows, columns=_TABLE_SCHEMAS["linking_metrics.csv"]
    )
    selected_linker = select_linker(linking_frame)

    test_gold = gold_by_sentence(splits.test, polarity_as_sign=True)
    test_aspects = crf.predict(splits.test)
    test_crf_predictions = aspects_by_sentence(test_aspects)
    crf_rows.extend(
        _span_row(
            "test",
            match,
            evaluate_aspects(test_crf_predictions, test_gold, match=match),
        )
        for match in ("exact", "head")
    )
    test_opinions = lexicon.transform(splits.test)
    test_links = link_aspects(
        test_aspects,
        test_opinions,
        pmi=pmi,
        method=selected_linker,
        max_dep_distance=config.max_dep_distance,
    )
    end_to_end_rows = [
        _joint_row(
            selected_linker,
            match,
            evaluate_aspect_polarity(
                test_links,
                test_gold,
                match=match,
                predicted_aspect_count=len(test_aspects),
            ),
        )
        for match in ("exact", "head")
    ]
    summaries = pd.concat(
        [
            make_product_summary(test_links, count_mode="unique_reviews", top_k=None),
            make_product_summary(test_links, count_mode="occurrences", top_k=None),
        ],
        ignore_index=True,
    )

    crf_frame = pd.DataFrame(crf_rows, columns=_TABLE_SCHEMAS["crf_metrics.csv"])
    negation_frame = pd.DataFrame(
        negation_rows, columns=_TABLE_SCHEMAS["negation_ablation.csv"]
    )
    end_to_end_frame = pd.DataFrame(
        end_to_end_rows, columns=_TABLE_SCHEMAS["end_to_end_metrics.csv"]
    )
    ate_plot_data = pd.concat(
        [
            baseline_frame.query("match == 'exact'")[
                ["pipeline", "precision", "recall", "f1"]
            ],
            crf_frame.query("split == 'dev' and match == 'exact'")
            .assign(pipeline="crf")[["pipeline", "precision", "recall", "f1"]],
        ],
        ignore_index=True,
    )
    ate_figure, _ = plot_ate_comparison(ate_plot_data)
    linking_figure, _ = plot_linking_tradeoff(linking_frame)
    plot_summary = summaries.query("count_mode == 'unique_reviews'").head(30)
    summary_figure, _ = plot_product_summary(plot_summary)
    figures = {
        "ate_comparison.png": ate_figure,
        "linking_tradeoff.png": linking_figure,
        "product_summaries.png": summary_figure,
    }
    tables = {
        "corpus_summary.csv": _corpus_summary(corpus),
        "ate_baselines.csv": baseline_frame,
        "crf_metrics.csv": crf_frame,
        "negation_ablation.csv": negation_frame,
        "linking_metrics.csv": linking_frame,
        "end_to_end_metrics.csv": end_to_end_frame,
        "product_summaries.csv": summaries,
    }
    alignment = bio_alignment_diagnostic(splits.train)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,
        "split_review_counts": {
            "train": len(splits.train),
            "dev": len(splits.dev),
            "test": len(splits.test),
        },
        "selected_linker": selected_linker,
        "settings": {
            key: value
            for key, value in asdict(config).items()
            if key not in {"data_dir", "output_dir"}
        },
        "training_alignment": {
            "total_gold_aspects": alignment.total_gold_aspects,
            "aligned_aspects": alignment.aligned_aspects,
            "match_rate": alignment.match_rate,
            "reason_counts": alignment.reason_counts,
        },
        "learned_state": {
            "opinion_lexicon_size": len(lexicon.polarities),
            "pmi_pair_count": len(pmi.scores),
        },
        "versions": {
            "python": platform.python_version(),
            "spacy": metadata.version("spacy"),
            "sklearn_crfsuite": metadata.version("sklearn-crfsuite"),
            "spacy_model": nlp.meta.get("version", "unknown"),
        },
        "data_sha256": _data_checksums(config.data_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    try:
        hashes = publish_artifacts(
            tables,
            config.output_dir,
            figures=figures,
            manifest=manifest,
        )
    finally:
        for figure in figures.values():
            plt.close(figure)
    final_manifest = {
        **manifest,
        "status": "complete",
        "artifact_sha256": hashes,
    }
    return RunResult(
        selected_linker=selected_linker,
        output_dir=config.output_dir,
        manifest=final_manifest,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the reproducible experiment command-line entry point.

    The parser accepts the local raw-data directory, artifact output
    directory, and random seed.  The function prints the publication location
    and selected linker; all tables and figures are written by
    :func:`run_experiment`.

    Returns:
        ``None``.  Data and validation errors propagate to the caller.
    """
    args = _parse_args()
    result = run_experiment(
        RunConfig(data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed)
    )
    print(f"Published artifacts to {result.output_dir}")
    print(f"Selected linker: {result.selected_linker}")


if __name__ == "__main__":
    main()
