"""Estimating m and u.

`u` is measured, `m` is learned, and the two need different methods because the data available for
each is different.

**u by random sampling.** Draw pairs at random from the two files. In any realistic pair of files
almost every random pair is a non-match, so how often a field agrees across those pairs *is* u,
to a very good approximation. No labels are needed.

**m by expectation maximisation.** Among the candidate pairs there is a mixture of matches and
non-matches, and nobody has labelled which is which. EM alternates between assigning each pair a
probability of being a match given the current parameters, and re-estimating the parameters given
those probabilities, until they stop moving.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

from syntel_linkage.model import Comparison, Parameters, Record, TrainingReport, clamp

#: EM stops when no parameter moves by more than this.
TOLERANCE = 1e-6
#: Iterations are cheap on a bounded sample and the tail of EM converges slowly, so the cap is
#: generous. Reaching it is reported rather than treated as convergence.
MAX_ITERATIONS = 200
#: Below this many candidate pairs there is nothing for EM to learn from, and iterating would just
#: return the values it started with wearing a trained label.
MINIMUM_PAIRS_FOR_EM = 20


def estimate_u(
    left: Sequence[Record],
    right: Sequence[Record],
    *,
    fields: Sequence[str],
    sample_pairs: int = 10_000,
    seed: int = 0,
) -> tuple[dict[str, float], int]:
    """How often each field agrees between two records drawn at random.

    Seeded on purpose: the same inputs must produce the same estimate, or a match probability is
    not reproducible and cannot be evidence.
    """
    if not left or not right:
        return ({name: clamp(0.0) for name in fields}, 0)
    rng = random.Random(seed)
    total = len(left) * len(right)
    draws = min(sample_pairs, total)
    agreements = dict.fromkeys(fields, 0)
    for _ in range(draws):
        a = left[rng.randrange(len(left))]
        b = right[rng.randrange(len(right))]
        for name in fields:
            value = str(a.get(name, "")).strip()
            if value and value == str(b.get(name, "")).strip():
                agreements[name] += 1
    return ({name: clamp(agreements[name] / draws) for name in fields}, draws)


def default_prior(left_size: int, right_size: int) -> float:
    """Roughly one counterpart per record, which is what a candidate join asserts.

    Using the all-pairs prior instead would make a genuine identifier match look like a coin toss,
    because the comparison space here is an admitted pair of columns, not two whole estates.
    """
    if left_size <= 0 or right_size <= 0:
        return clamp(0.0)
    return clamp(min(left_size, right_size) / (left_size * right_size))


def estimate_m(
    comparisons: Sequence[Comparison],
    *,
    fields: Sequence[str],
    u: Mapping[str, float],
    prior: float,
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[dict[str, float], float, int, bool]:
    """Expectation maximisation over the candidate pairs.

    Returns the trained m, the trained prior, how many iterations it took and whether it settled.
    u is held fixed: it was measured from random pairs, which is better evidence than anything this
    mixture can infer about non-matches.
    """
    # A field that agrees in every candidate pair has no variance to learn from, and including it
    # in the likelihood is worse than useless: its m/u ratio then pushes every pair the same way,
    # so a misspecified u for it drags the whole mixture. Blocking keys are always in this case.
    informative = [name for name in fields if not _always_agrees(comparisons, name)]
    m = dict.fromkeys(fields, 0.9)
    current_prior = prior
    iterations = 0
    converged = False
    if not informative:
        return m, prior, 0, True
    for iterations in range(1, max_iterations + 1):  # noqa: B007 - the final value is reported
        responsibilities: list[float] = []
        for comparison in comparisons:
            match_likelihood = current_prior
            non_match_likelihood = 1.0 - current_prior
            for name in informative:
                agreed = name in comparison.agreed
                match_likelihood *= clamp(m[name]) if agreed else clamp(1.0 - m[name])
                non_match_likelihood *= clamp(u[name]) if agreed else clamp(1.0 - u[name])
            total = match_likelihood + non_match_likelihood
            responsibilities.append(match_likelihood / total if total > 0 else 0.0)
        weight = sum(responsibilities)
        if weight <= 0:
            break
        updated = dict(m)
        for name in informative:
            agreeing = sum(
                responsibility
                for responsibility, comparison in zip(responsibilities, comparisons, strict=True)
                if name in comparison.agreed
            )
            updated[name] = clamp(agreeing / weight)
        updated_prior = clamp(weight / len(comparisons))
        moved = max(
            [abs(updated[name] - m[name]) for name in informative] + [abs(updated_prior - current_prior)],
            default=0.0,
        )
        m, current_prior = updated, updated_prior
        if moved < TOLERANCE:
            converged = True
            break
    return m, current_prior, iterations, converged


def _always_agrees(comparisons: Sequence[Comparison], name: str) -> bool:
    return bool(comparisons) and all(name in comparison.agreed for comparison in comparisons)


def train(
    left: Sequence[Record],
    right: Sequence[Record],
    comparisons: Sequence[Comparison],
    *,
    fields: Sequence[str],
    blocked: Sequence[str] = (),
    sample_pairs: int = 10_000,
    seed: int = 0,
) -> tuple[Parameters, TrainingReport]:
    """Measure u, then learn m, and say honestly how each was arrived at."""
    u, sampled = estimate_u(left, right, fields=fields, sample_pairs=sample_pairs, seed=seed)
    prior = default_prior(len(left), len(right))
    constant = tuple(sorted(name for name in fields if _always_agrees(comparisons, name)))
    untrainable = tuple(sorted({*blocked, *constant}))
    notes: list[str] = []
    if blocked:
        notes.append(
            "used for blocking, so they always agree within a block and carry no information: " + ", ".join(blocked)
        )
    for name in constant:
        if name not in set(blocked):
            notes.append(f"{name} agreed in every candidate pair here, so there is no variation to learn from")
    if len(comparisons) < MINIMUM_PAIRS_FOR_EM or len(constant) == len(fields):
        reason = (
            f"only {len(comparisons)} candidate pairs"
            if len(comparisons) < MINIMUM_PAIRS_FOR_EM
            else "every compared field agreed in every pair, including the blocking key"
        )
        return (
            Parameters(m=dict.fromkeys(fields, 0.9), u=u, prior=prior),
            TrainingReport(
                method_m=f"default ({reason})",
                method_u="random_sampling",
                prior=prior,
                untrainable=untrainable,
                pairs_compared=len(comparisons),
                pairs_sampled=sampled,
                notes=tuple(notes),
            ),
        )
    m, trained_prior, iterations, converged = estimate_m(comparisons, fields=fields, u=u, prior=prior)
    if not converged:
        notes.append(f"expectation maximisation stopped at {iterations} iterations without settling")
    informative = [name for name in fields if name not in set(untrainable)]
    if len(informative) == 1:
        # One binary comparison gives one observed agreement rate and two free parameters, so any
        # (m, prior) pair on the same ridge fits equally well. The numbers below are a point on
        # that ridge rather than the point, and a caller that treats them as measured is wrong.
        notes.append(
            f"only one field varies here ({informative[0]}), so m and the prior cannot be separated: "
            "the pair of values fits the data but neither is individually identified"
        )
    return (
        Parameters(m=m, u=u, prior=trained_prior),
        TrainingReport(
            method_m="expectation_maximisation",
            method_u="random_sampling",
            prior=trained_prior,
            iterations=iterations,
            converged=converged,
            untrainable=untrainable,
            pairs_compared=len(comparisons),
            pairs_sampled=sampled,
            notes=tuple(notes),
        ),
    )


__all__ = [
    "MAX_ITERATIONS",
    "MINIMUM_PAIRS_FOR_EM",
    "TOLERANCE",
    "default_prior",
    "estimate_m",
    "estimate_u",
    "train",
]
