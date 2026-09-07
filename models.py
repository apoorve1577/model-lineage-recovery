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

# Whether an operation preserves the model's learned behaviour.
SEMANTICALLY_PRESERVING = {
    "root": False,
    "fine-tune": False,
    "quantize": True,
    "merge": False,
    "compose": False,   # an adapter deliberately changes behaviour
}

# Whether a derivation can be replayed without retraining. This is what
# makes recovery cost asymmetric, and it is deliberately NOT the same set
# as SEMANTICALLY_PRESERVING. Re-composing a LoRA adapter against a clean
# base changes behaviour relative to the base, yet costs nothing to redo:
# the adapter itself was never the compromised artifact.
CHEAP_TO_REBUILD = {"quantize", "compose"}

REBUILD_COST = {
    "quantize": "cheap: re-derive deterministically from clean parent",
    "compose": "cheap: re-serve the existing adapter against a clean base, no retraining",
    "fine-tune": "expensive: retraining required",
    "merge": "expensive: re-merge, and every parent must be clean first",
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
