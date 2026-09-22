"""Blocking: compare the pairs worth comparing.

Two files of 100,000 records make ten billion pairs. Blocking says only pairs agreeing on some key
are worth scoring, which is the difference between a method that runs and one that does not.

The cost is stated rather than hidden: a true match that disagrees on every blocking key is never
compared and therefore never found. `coverage` reports how much of each side a blocking pass
reached, so a caller can say what its recall rests on.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from syntel_linkage.model import Comparison, Record


@dataclass(frozen=True, slots=True)
class BlockingReport:
    pairs: int
    left_covered: int
    right_covered: int
    left_total: int
    right_total: int

    @property
    def left_coverage(self) -> float:
        return self.left_covered / self.left_total if self.left_total else 0.0

    @property
    def right_coverage(self) -> float:
        return self.right_covered / self.right_total if self.right_total else 0.0


def key_of(record: Record, keys: Sequence[str]) -> tuple[str, ...] | None:
    """The blocking key, or None when the record cannot be blocked on it.

    A record missing a blocking field is excluded rather than matched against everything: an empty
    key would otherwise become the largest block in the file.
    """
    values = tuple(str(record.get(key, "")).strip() for key in keys)
    return values if all(values) else None


def candidate_pairs(
    left: Sequence[Record],
    right: Sequence[Record],
    *,
    blocking_keys: Sequence[str],
    comparison_fields: Sequence[str],
    id_field: str = "unique_id",
    max_pairs: int = 1_000_000,
) -> tuple[tuple[Comparison, ...], BlockingReport]:
    """Every pair agreeing on the blocking key, with which comparison fields agreed."""
    index: dict[tuple[str, ...], list[Record]] = {}
    for record in right:
        key = key_of(record, blocking_keys)
        if key is not None:
            index.setdefault(key, []).append(record)
    comparisons: list[Comparison] = []
    left_covered: set[str] = set()
    right_covered: set[str] = set()
    for record in left:
        key = key_of(record, blocking_keys)
        if key is None:
            continue
        for other in index.get(key, ()):
            if len(comparisons) >= max_pairs:
                break
            agreed = frozenset(
                name
                for name in comparison_fields
                if str(record.get(name, "")).strip() and str(record.get(name, "")) == str(other.get(name, ""))
            )
            comparisons.append(Comparison(left_id=str(record[id_field]), right_id=str(other[id_field]), agreed=agreed))
            left_covered.add(str(record[id_field]))
            right_covered.add(str(other[id_field]))
    return tuple(comparisons), BlockingReport(
        pairs=len(comparisons),
        left_covered=len(left_covered),
        right_covered=len(right_covered),
        left_total=len(left),
        right_total=len(right),
    )


def blocked_fields(blocking_keys: Iterable[str], comparison_fields: Iterable[str]) -> tuple[str, ...]:
    """Comparison fields that are also blocking keys, and so always agree within a block."""
    keys = set(blocking_keys)
    return tuple(sorted(name for name in comparison_fields if name in keys))


__all__ = ["BlockingReport", "blocked_fields", "candidate_pairs", "key_of"]
