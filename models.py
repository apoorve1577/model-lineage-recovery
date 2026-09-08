"""
Data model for the blast radius prototype.

A ModelRecord represents one node in the model lineage graph: a base model,
or a model derived from one or more parents via fine-tuning, quantization,
or merging. This mirrors the fields an OpenSSF Model Signing attestation
plus a fine-tuning manifest would realistically carry.
"""
from dataclasses import dataclass, asdict
from typing import List


VALID_OPERATIONS = {"root", "fine-tune", "quantize", "merge", "compose"}

# Whether an operation is INTENDED to preserve the model's learned behaviour.
# Approximate and method-dependent, and a statement about utility, not safety:
# a quantization can introduce behaviour its parent did not have.
SEMANTICALLY_PRESERVING = {
    "root": False,
    "fine-tune": False,
    "quantize": True,
    "merge": False,
    "compose": False,   # an adapter deliberately changes behaviour
}

# Whether the DERIVATION STEP can be replayed without training, given clean
# inputs. This is deliberately NOT the same set as SEMANTICALLY_PRESERVING, and
# it is a property of the step alone: rebuilding a compromised PARENT may still
# require training, but that cost belongs to the parent's own edge, not to this
# one. Merging is in this set because standard weight averaging and most
# adapter-merging methods combine existing weights without any training; only
# fine-tuning inherently requires it.
CHEAP_TO_REBUILD = {"quantize", "compose", "merge"}

REBUILD_COST = {
    "quantize": "cheap: re-derive from a clean parent; determinism depends on "
                "method, configuration and environment",
    "compose": "cheap: re-serve the adapter against a clean base, no training",
    "merge": "cheap to re-execute: combines existing weights without training. "
             "Cost lies in obtaining clean parents, not in the merge itself",
    "fine-tune": "expensive: retraining required",
    "root": "not applicable: no parent to rebuild from",
}


@dataclass
class ModelRecord:
    id: str
    hash: str
    parent_ids: List[str]
    operation: str
    signed: bool
    is_patient_zero: bool
    generation: int

    def __post_init__(self):
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(f"unknown operation {self.operation!r} for {self.id}")
        expected_parent_count = {
            "root": 0,
            "fine-tune": 1,
            "quantize": 1,
            "compose": 1,   # the base model the adapter is served against
        }
        if self.operation in expected_parent_count:
            n = expected_parent_count[self.operation]
            if len(self.parent_ids) != n:
                raise ValueError(
                    f"{self.id} is a {self.operation} node but has "
                    f"{len(self.parent_ids)} parents, expected {n}"
                )
        elif self.operation == "merge" and len(self.parent_ids) < 2:
            raise ValueError(f"{self.id} is a merge node but has fewer than 2 parents")

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return ModelRecord(**d)
