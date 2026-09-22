"""Probabilistic record linkage by the Fellegi-Sunter method, in the standard library.

Written from the method published in Fellegi and Sunter (1969). No code is copied from any
existing implementation; see README.md.
"""

from syntel_linkage.blocking import BlockingReport, blocked_fields, candidate_pairs, key_of
from syntel_linkage.clustering import clusters, suspicious
from syntel_linkage.estimate import default_prior, estimate_m, estimate_u, train
from syntel_linkage.linker import LinkageResult, link, score
from syntel_linkage.model import Comparison, Parameters, Prediction, Record, TrainingReport

__version__ = "0.1.0"

__all__ = [
    "BlockingReport",
    "Comparison",
    "LinkageResult",
    "Parameters",
    "Prediction",
    "Record",
    "TrainingReport",
    "__version__",
    "blocked_fields",
    "candidate_pairs",
    "clusters",
    "default_prior",
    "estimate_m",
    "estimate_u",
    "key_of",
    "link",
    "score",
    "suspicious",
    "train",
]
