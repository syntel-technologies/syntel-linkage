"""The facade: two files in, scored pairs and a training report out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from syntel_linkage.blocking import BlockingReport, blocked_fields, candidate_pairs
from syntel_linkage.estimate import train
from syntel_linkage.model import Comparison, Parameters, Prediction, Record, TrainingReport


@dataclass(frozen=True, slots=True)
class LinkageResult:
    predictions: tuple[Prediction, ...]
    parameters: Parameters
    training: TrainingReport
    blocking: BlockingReport

    def above(self, threshold: float) -> tuple[Prediction, ...]:
        return tuple(p for p in self.predictions if p.probability >= threshold)


def score(comparison: Comparison, parameters: Parameters, fields: Sequence[str]) -> Prediction:
    """The Fellegi-Sunter weight for one pair, and the posterior it implies.

    Every field contributes: an agreement adds log2(m/u), a disagreement adds log2((1-m)/(1-u)),
    which is negative. Scoring only the agreements is the common shortcut and it makes every pair
    look better than it is.
    """
    weight = sum(
        parameters.agreement_weight(name) if name in comparison.agreed else parameters.disagreement_weight(name)
        for name in fields
    )
    total = weight + parameters.prior_weight
    probability = 1.0 / (1.0 + 2.0 ** (-total))
    return Prediction(
        left_id=comparison.left_id,
        right_id=comparison.right_id,
        weight=round(weight, 6),
        probability=round(probability, 6),
        agreed=tuple(sorted(comparison.agreed)),
    )


def link(
    left: Sequence[Record],
    right: Sequence[Record],
    *,
    blocking_keys: Sequence[str],
    comparison_fields: Sequence[str],
    id_field: str = "unique_id",
    threshold: float = 0.0,
    sample_pairs: int = 10_000,
    max_pairs: int = 1_000_000,
    seed: int = 0,
) -> LinkageResult:
    """Block, measure u, learn m, score every candidate pair.

    `threshold` filters the returned predictions only. The parameters are trained on every
    candidate pair, because throwing away the low-scoring ones before training would remove exactly
    the evidence expectation maximisation needs about what a non-match looks like.
    """
    fields = list(dict.fromkeys([*comparison_fields, *blocking_keys]))
    comparisons, blocking = candidate_pairs(
        left,
        right,
        blocking_keys=blocking_keys,
        comparison_fields=fields,
        id_field=id_field,
        max_pairs=max_pairs,
    )
    parameters, report = train(
        left,
        right,
        comparisons,
        fields=fields,
        blocked=blocked_fields(blocking_keys, fields),
        sample_pairs=sample_pairs,
        seed=seed,
    )
    predictions = tuple(
        sorted(
            (score(comparison, parameters, fields) for comparison in comparisons),
            key=lambda p: (-p.probability, p.left_id, p.right_id),
        )
    )
    return LinkageResult(
        predictions=tuple(p for p in predictions if p.probability >= threshold),
        parameters=parameters,
        training=report,
        blocking=blocking,
    )


__all__ = ["LinkageResult", "link", "score"]
