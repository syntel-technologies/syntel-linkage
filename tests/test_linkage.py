"""The method, tested against data whose right answer is known by construction.

A linkage library that merely runs is worthless; the question is whether it recovers the truth when
the truth is knowable, and whether it says so when it cannot.
"""

from __future__ import annotations

import random

import pytest

from syntel_linkage import (
    Comparison,
    Parameters,
    clusters,
    default_prior,
    estimate_m,
    estimate_u,
    link,
    score,
    suspicious,
)


def _people(count: int, *, prefix: str, key_prefix: str = "") -> list[dict[str, str]]:
    return [
        {"unique_id": f"{prefix}{i}", "key": f"{key_prefix}{i:04d}", "name": f"person {i}", "town": f"town {i % 7}"}
        for i in range(count)
    ]


# --- the numbers -----------------------------------------------------------------------------


def test_u_is_measured_and_matches_the_rate_it_was_built_with() -> None:
    """A field taking 7 values agrees between random records about one time in seven."""
    left = _people(300, prefix="L")
    right = _people(300, prefix="R")
    u, sampled = estimate_u(left, right, fields=["town", "key"], sample_pairs=20_000, seed=1)
    assert sampled == 20_000
    assert u["town"] == pytest.approx(1 / 7, abs=0.02)
    assert u["key"] == pytest.approx(1 / 300, abs=0.01), "a near-unique field almost never agrees by chance"


def test_u_is_reproducible_because_a_probability_nobody_can_reproduce_is_not_evidence() -> None:
    left, right = _people(100, prefix="L"), _people(100, prefix="R")
    first, _ = estimate_u(left, right, fields=["town"], sample_pairs=5_000, seed=7)
    second, _ = estimate_u(left, right, fields=["town"], sample_pairs=5_000, seed=7)
    assert first == second


def test_expectation_maximisation_recovers_an_m_it_was_not_told() -> None:
    """Build pairs where name agrees in 80% of true matches and town in 60%, and see if EM finds them.

    Two informative fields, not one: with a single binary comparison there is one observed
    agreement rate and two free parameters, so m and the prior are not separately identifiable.
    """
    rng = random.Random(3)
    comparisons: list[Comparison] = []
    for i in range(600):  # true matches
        agreed = {"key"}
        if rng.random() < 0.8:
            agreed.add("name")
        if rng.random() < 0.6:
            agreed.add("town")
        comparisons.append(Comparison(left_id=f"L{i}", right_id=f"R{i}", agreed=frozenset(agreed)))
    for i in range(600):  # non-matches that collided on the blocking key
        agreed = {"key"}
        if rng.random() < 0.02:
            agreed.add("name")
        if rng.random() < 0.15:
            agreed.add("town")
        comparisons.append(Comparison(left_id=f"L{i}", right_id=f"X{i}", agreed=frozenset(agreed)))
    m, prior, iterations, converged = estimate_m(
        comparisons, fields=["key", "name", "town"], u={"key": 1.0, "name": 0.02, "town": 0.15}, prior=0.5
    )
    assert converged, f"stopped after {iterations} iterations"
    assert m["name"] == pytest.approx(0.8, abs=0.06), m
    assert m["town"] == pytest.approx(0.6, abs=0.08), m
    assert prior == pytest.approx(0.5, abs=0.06)


def test_one_informative_field_cannot_separate_m_from_the_prior() -> None:
    """A real limit of the method, reported rather than presented as a measurement."""
    left = [{"unique_id": f"L{i}", "key": str(i), "name": f"n{i}"} for i in range(100)]
    right = [{"unique_id": f"R{i}", "key": str(i), "name": f"n{i}" if i % 4 else "other"} for i in range(100)]
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name"])
    assert any("cannot be separated" in note for note in result.training.notes), result.training.notes


def test_a_field_that_always_agrees_is_left_alone_rather_than_trained_to_one() -> None:
    comparisons = [Comparison(left_id=f"L{i}", right_id=f"R{i}", agreed=frozenset({"key"})) for i in range(50)]
    m, _, _, _ = estimate_m(comparisons, fields=["key"], u={"key": 0.001}, prior=0.5)
    assert m["key"] == 0.9, "untouched: inside a block this field carries no information"


def test_a_disagreement_counts_against_a_pair() -> None:
    """Scoring only the agreements is the common shortcut, and it flatters every pair."""
    parameters = Parameters(m={"a": 0.9, "b": 0.9}, u={"a": 0.1, "b": 0.1}, prior=0.5)
    both = score(Comparison("L", "R", frozenset({"a", "b"})), parameters, ["a", "b"])
    one = score(Comparison("L", "R", frozenset({"a"})), parameters, ["a", "b"])
    assert one.weight < both.weight
    assert one.weight == pytest.approx(0.0, abs=1e-9), "one agreement cancelled by one disagreement"


def test_the_prior_moves_the_posterior_without_moving_the_evidence() -> None:
    comparison = Comparison("L", "R", frozenset({"a"}))
    fields = ["a"]
    rare = score(comparison, Parameters(m={"a": 0.9}, u={"a": 0.01}, prior=1e-5), fields)
    likely = score(comparison, Parameters(m={"a": 0.9}, u={"a": 0.01}, prior=0.5), fields)
    assert rare.weight == likely.weight, "the evidence is the same"
    assert rare.probability < likely.probability, "what it implies is not"


def test_the_default_prior_is_about_one_counterpart_per_record() -> None:
    assert default_prior(100, 100) == pytest.approx(0.01)
    assert default_prior(0, 5) < 1e-5


# --- the whole thing -------------------------------------------------------------------------


def test_true_matches_score_far_above_records_that_merely_collided() -> None:
    """The question a linkage library exists to answer: which of these pairs are the same thing."""
    left = _people(200, prefix="L")
    right = _people(200, prefix="R")
    # Fifty records from a third file collide on the key and are nothing to do with the left side.
    right += [
        {"unique_id": f"C{i}", "key": f"{i:04d}", "name": f"someone else {i}", "town": "elsewhere"} for i in range(50)
    ]
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name", "town"])
    scores = {(p.left_id, p.right_id): p.probability for p in result.predictions}
    true_pairs = [scores[(f"L{i}", f"R{i}")] for i in range(200)]
    collisions = [scores[(f"L{i}", f"C{i}")] for i in range(50)]
    assert min(true_pairs) > 0.9, min(true_pairs)
    assert max(collisions) < 0.1, max(collisions)
    assert result.training.method_m == "expectation_maximisation"
    assert result.training.converged
    assert result.blocking.left_coverage == 1.0


def test_accepted_matches_are_one_to_one_when_the_data_is() -> None:
    left, right = _people(200, prefix="L"), _people(200, prefix="R")
    right += [{"unique_id": f"C{i}", "key": f"{i:04d}", "name": "x", "town": "y"} for i in range(50)]
    accepted = link(left, right, blocking_keys=["key"], comparison_fields=["name", "town"]).above(0.5)
    assert len(accepted) == 200
    assert len({p.left_id for p in accepted}) == 200, "no left record matched twice"
    assert len({p.right_id for p in accepted}) == 200


def test_the_prefixed_key_case_the_service_needs() -> None:
    """`E001` against `001`: the caller normalises, and the library links what it is given."""
    left = [{"unique_id": f"L{i}", "key": f"E{i:03d}", "name": f"person {i}"} for i in range(1, 101)]
    right = [{"unique_id": f"R{i}", "key": f"{i:03d}", "name": f"person {i}"} for i in range(1, 101)]
    raw = link(left, right, blocking_keys=["key"], comparison_fields=["name"], threshold=0.5)
    assert raw.predictions == (), "unnormalised, there is nothing to block on and nothing is found"

    for row in left:
        row["key"] = row["key"].lstrip("E").lstrip("0") or "0"
    for row in right:
        row["key"] = row["key"].lstrip("0") or "0"
    normalised = link(left, right, blocking_keys=["key"], comparison_fields=["name"], threshold=0.5)
    assert len(normalised.predictions) == 100
    assert all(p.probability > 0.9 for p in normalised.predictions)


def test_a_record_missing_its_blocking_key_is_excluded_not_matched_to_everything() -> None:
    left = [*_people(10, prefix="L"), {"unique_id": "Lx", "key": "", "name": "nobody", "town": "town 1"}]
    right = _people(10, prefix="R")
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name"], threshold=0.0)
    assert "Lx" not in {p.left_id for p in result.predictions}
    assert result.blocking.left_coverage < 1.0, "and the coverage says so"


def test_blocking_reports_what_it_could_not_reach() -> None:
    """A true match that disagrees on the blocking key is never compared, so recall rests on this."""
    left = _people(50, prefix="L")
    right = _people(50, prefix="R", key_prefix="Z")
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name"])
    assert result.predictions == ()
    assert result.blocking.pairs == 0
    assert result.blocking.left_coverage == 0.0


def test_too_few_pairs_to_learn_from_is_reported_rather_than_estimated() -> None:
    left, right = _people(5, prefix="L"), _people(5, prefix="R")
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name"])
    assert result.training.method_m.startswith("default")
    assert "only 5 candidate pairs" in result.training.method_m


def test_blocking_on_the_only_compared_field_leaves_nothing_to_train() -> None:
    left, right = _people(100, prefix="L"), _people(100, prefix="R")
    result = link(left, right, blocking_keys=["key"], comparison_fields=["key"])
    assert result.training.untrainable == ("key",)
    assert result.training.method_m.startswith("default")
    assert any("carry no information" in note for note in result.training.notes)
    assert len(result.predictions) == 100, "the pairs are still scored"


def test_the_same_inputs_produce_the_same_scores() -> None:
    left, right = _people(80, prefix="L"), _people(80, prefix="R")
    first = link(left, right, blocking_keys=["key"], comparison_fields=["name", "town"])
    second = link(left, right, blocking_keys=["key"], comparison_fields=["name", "town"])
    assert [(p.left_id, p.right_id, p.probability) for p in first.predictions] == [
        (p.left_id, p.right_id, p.probability) for p in second.predictions
    ]


def test_the_pair_limit_is_honoured() -> None:
    left = [{"unique_id": f"L{i}", "key": "same", "name": f"n{i}"} for i in range(200)]
    right = [{"unique_id": f"R{i}", "key": "same", "name": f"n{i}"} for i in range(200)]
    result = link(left, right, blocking_keys=["key"], comparison_fields=["name"], max_pairs=500)
    assert result.blocking.pairs <= 500


# --- clustering ------------------------------------------------------------------------------


def test_connected_components() -> None:
    assert clusters([("a", "b"), ("b", "c"), ("d", "e")]) == [["a", "b", "c"], ["d", "e"]]
    assert clusters([]) == []


def test_a_chain_held_together_by_transitivity_is_flagged() -> None:
    """A matched B and B matched C, but A and C were never compared. Worth a person's eye."""
    assert suspicious([("a", "b"), ("b", "c")]) == [["a", "b", "c"]]
    assert suspicious([("a", "b"), ("b", "c"), ("a", "c")]) == [], "fully connected needs no review"
    assert suspicious([("a", "b")]) == [], "a pair is not a chain"
