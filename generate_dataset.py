"""
Generates a synthetic model lineage dataset: several family trees of
fine-tuned/quantized/merged models, a few rooted at a "patient zero"
compromised base model.

Two graphs come out of this, and the gap between them is the whole point:

  true_lineage.json    - every relationship that actually happened.
                          Used ONLY by the evaluator, as ground truth.
  tracked_lineage.json - what a real system would actually have on record.
                          Each true edge has a chance of never being
                          submitted (no signing attestation, no fine-tuning
                          manifest filed), mirroring real adoption gaps.
                          This is the ONLY input the blast radius query sees.

The evaluation question the paper cares about: given only the tracked
graph, how much of the true blast radius can we still find, and how much
silently disappears because the edge was never recorded?
"""
import json
import random
from models import ModelRecord

SEED = 42
N_FAMILIES = 7
N_ROOT_PATIENT_ZEROS = 2      # compromise introduced at the base model
N_MIDCHAIN_PATIENT_ZEROS = 3  # compromise introduced partway down a lineage
MAX_GENERATIONS = 4
UNTRACKED_EDGE_PROBABILITY = 0.15  # true edges never recorded in the tracked system

random.seed(SEED)


def fake_hash(rng):
    return "sha256:" + "".join(rng.choice("0123456789abcdef") for _ in range(12))


def generate(seed=SEED, n_families=N_FAMILIES,
             n_root_pz=N_ROOT_PATIENT_ZEROS, n_mid_pz=N_MIDCHAIN_PATIENT_ZEROS):
    rng = random.Random(seed)
    records = {}
    all_ids_by_family = []

    for fam_idx in range(n_families):
        family_ids = []

        root_id = f"fam{fam_idx}-root"
        root = ModelRecord(
            id=root_id,
            hash=fake_hash(rng),
            parent_ids=[],
            operation="root",
            signed=True,
            is_patient_zero=False,
            generation=0,
        )
        records[root_id] = root
        family_ids.append(root_id)

        frontier = [root_id]
        for gen in range(1, MAX_GENERATIONS + 1):
            next_frontier = []
            for parent_id in frontier:
                n_children = rng.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1])[0]
                for _ in range(n_children):
                    op = rng.choices(
                        ["fine-tune", "quantize"], weights=[0.7, 0.3]
                    )[0]
                    child_id = f"fam{fam_idx}-g{gen}-{len(records)}"
                    child = ModelRecord(
                        id=child_id,
                        hash=fake_hash(rng),
                        parent_ids=[parent_id],
                        operation=op,
                        signed=rng.random() < 0.8,
                        is_patient_zero=False,
                        generation=gen,
                    )
                    records[child_id] = child
                    family_ids.append(child_id)
                    next_frontier.append(child_id)
            frontier = next_frontier
            if not frontier:
                break

        all_ids_by_family.append(family_ids)

    # A handful of cross-family merges: combine a node from one family with
    # a node from another. If either parent is in a compromised family's
    # descendant set, the merge inherits the compromise, this is exactly
    # the case existing tools handle worst.
    n_merges = max(2, round(6 * n_families / N_FAMILIES))
    all_ids = list(records.keys())
    for i in range(n_merges):
        fam_a, fam_b = rng.sample(range(n_families), 2)
        parent_a = rng.choice(all_ids_by_family[fam_a])
        parent_b = rng.choice(all_ids_by_family[fam_b])
        gen = max(records[parent_a].generation, records[parent_b].generation) + 1
        merge_id = f"merge-{i}-{len(records)}"
        merge_node = ModelRecord(
            id=merge_id,
            hash=fake_hash(rng),
            parent_ids=[parent_a, parent_b],
            operation="merge",
            signed=rng.random() < 0.6,
            is_patient_zero=False,
            generation=gen,
        )
        records[merge_id] = merge_node
        all_ids.append(merge_id)

    # LoRA adapters, served against a base rather than merged into it.
    # Signed is forced True: the point of this case is that the adapter's
    # own signature verifies perfectly while the base underneath it is
    # compromised. Per-artifact verification cannot see this at all.
    n_adapters = max(2, round(5 * n_families / N_FAMILIES))
    base_candidates = [r.id for r in records.values() if r.operation in ("root", "fine-tune")]
    for i in range(n_adapters):
        base_id = rng.choice(base_candidates)
        adapter_id = f"lora-{i}-{len(records)}"
        records[adapter_id] = ModelRecord(
            id=adapter_id,
            hash=fake_hash(rng),
            parent_ids=[base_id],
            operation="compose",
            signed=True,
            is_patient_zero=False,
            generation=records[base_id].generation + 1,
        )

    assign_patient_zeros(records, rng, n_root_pz, n_mid_pz)
    return records


def assign_patient_zeros(records, rng,
                         n_root_pz=N_ROOT_PATIENT_ZEROS,
                         n_mid_pz=N_MIDCHAIN_PATIENT_ZEROS):
    """Places compromises at two structurally different positions.

    Root patient zeros model a poisoned foundation model or dataset: every
    descendant is affected and there is no clean ancestor to roll back to.
    Mid-chain patient zeros model a compromise introduced during one team's
    fine-tune: descendants are affected, but a clean ancestor still exists
    upstream, so rollback is actually possible.

    The distinction matters because it determines whether recovery means
    'roll back' or 'rebuild from scratch', and no existing tooling
    distinguishes the two.
    """
    roots = [r for r in records.values() if r.operation == "root"]
    non_roots = [
        r for r in records.values()
        if r.operation != "root" and r.generation >= 1
    ]

    for r in rng.sample(roots, min(n_root_pz, len(roots))):
        r.is_patient_zero = True

    # Only pick mid-chain nodes that aren't already downstream of a root PZ,
    # otherwise the two cases confound each other in the evaluation.
    root_pz_ids = {r.id for r in records.values() if r.is_patient_zero}
    contaminated = set()
    frontier = list(root_pz_ids)
    child_map = {}
    for rid, rec in records.items():
        for p in rec.parent_ids:
            child_map.setdefault(p, []).append(rid)
    while frontier:
        cur = frontier.pop()
        for child in child_map.get(cur, []):
            if child not in contaminated:
                contaminated.add(child)
                frontier.append(child)

    # Only pick mid-chain nodes that actually have descendants. A patient
    # zero with no children has a blast radius of one and tells us nothing
    # about recovery, which is the whole thing being measured.
    descendant_count = {}
    for rid in records:
        seen, stack = set(), list(child_map.get(rid, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(child_map.get(cur, []))
        descendant_count[rid] = len(seen)

    eligible = [
        r for r in non_roots
        if r.id not in contaminated
        and not r.is_patient_zero
        and descendant_count.get(r.id, 0) >= 2
    ]
    for r in rng.sample(eligible, min(n_mid_pz, len(eligible))):
        r.is_patient_zero = True


def build_tracked_edges(records, rng, drop_p=UNTRACKED_EDGE_PROBABILITY):
    """Drop each true edge independently with probability drop_p."""
    tracked = {}
    for rid, rec in records.items():
        tracked_parents = [
            p for p in rec.parent_ids if rng.random() >= drop_p
        ]
        tracked[rid] = tracked_parents
    return tracked


if __name__ == "__main__":
    records = generate()
    rng = random.Random(SEED + 1)
    tracked_parents = build_tracked_edges(records, rng)

    true_out = {rid: r.to_dict() for rid, r in records.items()}
    with open("data/true_lineage.json", "w") as f:
        json.dump(true_out, f, indent=2)

    tracked_out = {}
    for rid, r in records.items():
        d = r.to_dict()
        d["parent_ids"] = tracked_parents[rid]
        tracked_out[rid] = d
    with open("data/tracked_lineage.json", "w") as f:
        json.dump(tracked_out, f, indent=2)

    n_models = len(records)
    pz = [r for r in records.values() if r.is_patient_zero]
    n_root_pz = sum(1 for r in pz if r.operation == "root")
    n_mid_pz = len(pz) - n_root_pz
    n_true_edges = sum(len(r.parent_ids) for r in records.values())
    n_tracked_edges = sum(len(p) for p in tracked_parents.values())
    print(f"Generated {n_models} models across {N_FAMILIES} families")
    print(f"Patient zeros: {len(pz)} total "
          f"({n_root_pz} at roots, {n_mid_pz} mid-chain)")
    print(f"True edges: {n_true_edges}, tracked edges: {n_tracked_edges} "
          f"({n_true_edges - n_tracked_edges} dropped, "
          f"{100 * (n_true_edges - n_tracked_edges) / n_true_edges:.1f}% untracked)")


# Per-operation untracked rates. The uniform model in build_tracked_edges
# assumes every kind of derivation is equally likely to go unrecorded, which
# is the assumption the paper's limitations section flags as unrealistic:
# an automated quantization step is plausibly attested at a very different
# rate than a deliberate, reviewed model merge. These scenarios are each
# calibrated against the measured edge mix (fine-tune 0.520, quantize 0.207,
# merge 0.193, compose 0.080) so that every one drops the same ~15% of edges
# overall. Any difference in outcome is therefore attributable to the
# STRUCTURE of the missingness, not to its magnitude.
DROP_SCENARIOS = {
    # Every derivation equally likely to go unrecorded. The paper's default.
    "uniform": {"fine-tune": 0.15, "quantize": 0.15, "merge": 0.15, "compose": 0.15},
    # Routine automated steps under-reported; deliberate reviewed ones logged.
    "routine_ops_underreported": {
        "fine-tune": 0.08, "quantize": 0.35, "merge": 0.05, "compose": 0.35},
    # The inverse: ad-hoc merges escape the pipeline, automation always logs.
    "deliberate_ops_underreported": {
        "fine-tune": 0.13, "quantize": 0.02, "merge": 0.40, "compose": 0.02},
    # Adapter serving is outside every signing pipeline that exists today,
    # so composition edges are the ones that mostly do not get recorded.
    "composition_blind_spot": {
        "fine-tune": 0.0935, "quantize": 0.0935, "merge": 0.0935, "compose": 0.80},
}


def build_tracked_edges_by_op(records, rng, rates):
    """Drop each true edge with a probability that depends on the operation
    that produced the child. Edge (parent -> child) is typed by the child's
    operation, which is exactly the derivation whose attestation would be
    missing."""
    tracked = {}
    for rid, rec in records.items():
        p = rates.get(rec.operation, 0.0)
        tracked[rid] = [par for par in rec.parent_ids if rng.random() >= p]
    return tracked
