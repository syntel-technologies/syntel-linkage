"""The values the method works with (Fellegi and Sunter 1969).

Deliberately small: a record is a mapping the caller already has, a comparison is a field name, and
a parameter set is two probabilities per field. Nothing here knows about databases, dataframes or
where the records came from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: Probabilities are clamped away from 0 and 1. A field that agreed in every observed match would
#: otherwise give an infinite weight, and one observation is not certainty.
EPSILON = 1e-6


def clamp(value: float, *, low: float = EPSILON, high: float = 1.0 - EPSILON) -> float:
    return min(max(value, low), high)


Record = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Parameters:
    """m and u per field, and the prior probability that a random pair is a match.

    `m[f]` is how often field `f` agrees when two records *are* the same thing; `u[f]` is how often
    it agrees when they are not. The ratio between them is what a field's agreement is worth.
    """

    m: Mapping[str, float]
    u: Mapping[str, float]
    prior: float

    def agreement_weight(self, field_name: str) -> float:
        """log2(m/u): what it is worth that this field agreed."""
        return math.log2(clamp(self.m[field_name]) / clamp(self.u[field_name]))

    def disagreement_weight(self, field_name: str) -> float:
        """log2((1-m)/(1-u)): what it costs that this field did not agree.

        Leaving this out is the common shortcut and it inflates every score, because a pair is then
        only ever rewarded for what matched and never charged for what did not.
        """
        return math.log2(clamp(1.0 - self.m[field_name]) / clamp(1.0 - self.u[field_name]))

    @property
    def prior_weight(self) -> float:
        return math.log2(clamp(self.prior) / clamp(1.0 - self.prior))


@dataclass(frozen=True, slots=True)
class Comparison:
    """One pair, and which fields agreed."""

    left_id: str
    right_id: str
    agreed: frozenset[str]

    def pattern(self, fields: Sequence[str]) -> tuple[bool, ...]:
        return tuple(name in self.agreed for name in fields)


@dataclass(frozen=True, slots=True)
class Prediction:
    """A scored pair. `weight` is the evidence; `probability` is the evidence plus the prior."""

    left_id: str
    right_id: str
    weight: float
    probability: float
    agreed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """How each parameter was arrived at. Read this before reading the numbers.

    A caller that records a match probability without recording how m and u were obtained has
    written down a number nobody can check.
    """

    method_m: str
    method_u: str
    prior: float
    iterations: int = 0
    converged: bool = False
    #: Fields whose m probability cannot be learned because they were used for blocking: inside a
    #: block they always agree, so they carry no information.
    untrainable: tuple[str, ...] = ()
    pairs_compared: int = 0
    pairs_sampled: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


__all__ = [
    "EPSILON",
    "Comparison",
    "Parameters",
    "Prediction",
    "Record",
    "TrainingReport",
    "clamp",
]
