"""
Generates a synthetic model lineage dataset.

Two graphs come out of this, and the gap between them is the whole point:

  true_lineage.json    - every relationship that actually happened.
                          Used ONLY by the evaluator, as ground truth.
  tracked_lineage.json - what a real system would actually have on record.
                          Each true edge has a chance of never being
                          submitted (no signing attestation, no fine-tuning
                          manifest filed), mirroring real adoption gaps.
                          This is the ONLY input the blast radius query sees.

The evaluation question: given only the tracked graph, how much of the true
blast radius can we still find, and how much silently disappears because the
edge was never recorded?

TOPOLOGY NOTE (rewritten 2026-09-07). An earlier version of this generator
grew the fine-tune/quantize families to completion and only then bolted merge
and adapter nodes on top. That made every merge and every compose node a
LEAF, which silently rigged two results:

  * A merge edge could cost at most one node when unrecorded, because the
    merge had no descendants to lose. The measured criticality gap between
    single-parent and multi-parent edges was therefore an artifact of
    construction rather than a property of the propagation rule.
  * The planner's `blocked_on_compromised_merge_parent` path was never
    exercised beyond one hop, and the symmetric planner error - tracked says
    recoverable, truth says blocked because the merge's other-parent edge was
    dropped - was impossible to observe at all.

Generations are now interleaved: merges and adapters are created alongside
fine-tunes at each generation and re-enter the frontier, so anything can be
a parent of anything created later. Acyclicity still holds by construction,
since an edge only ever runs from an existing node to a newly created one.
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

N_MERGES = 6      # cross-family merges, spread across generations
N_ADAPTERS = 5    # LoRA adapters served against a base, spread across generations

random.seed(SEED)


def fake_hash(rng):
    return "sha256:" + "".join(rng.choice("0123456789abcdef") for _ in range(12))


def generate(seed=SEED, n_families=N_FAMILIES,
             n_root_pz=N_ROOT_PATIENT_ZEROS, n_mid_pz=N_MIDCHAIN_PATIENT_ZEROS):
    rng = random.Random(seed)
    records = {}
    # Which root families each node descends from. Used to keep merges
    # cross-family, and it is a set because a merge descends from several.
    families = {}

    scale = n_families / N_FAMILIES
    n_merges = max(2, round(N_MERGES * scale))
    n_adapters = max(2, round(N_ADAPTERS * scale))

    for fam_idx in range(n_families):
        rid = f"fam{fam_idx}-root"
        records[rid] = ModelRecord(
            id=rid, hash=fake_hash(rng), parent_ids=[], operation="root",
            signed=True, is_patient_zero=False, generation=0,
        )
        families[rid] = {fam_idx}

    frontier = list(records.keys())

    # Merges and adapters are spread over generations 1..MAX_GENERATIONS so
    # that ones created early still have generations left to acquire children.
    merge_budget = _spread(n_merges, MAX_GENERATIONS)
    adapter_budget = _spread(n_adapters, MAX_GENERATIONS)

    for gen in range(1, MAX_GENERATIONS + 1):
        created = []

        # 1. Ordinary single-parent derivations from the previous frontier.
        for parent_id in frontier:
            for _ in range(rng.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1])[0]):
                op = rng.choices(["fine-tune", "quantize"], weights=[0.7, 0.3])[0]
                cid = f"fam{min(families[parent_id])}-g{gen}-{len(records)}"
                records[cid] = ModelRecord(
                    id=cid, hash=fake_hash(rng), parent_ids=[parent_id],
                    operation=op, signed=rng.random() < 0.8,
                    is_patient_zero=False, generation=gen,
                )
                families[cid] = set(families[parent_id])
                created.append(cid)

        pool = [r for r in records if records[r].generation < gen]

        # 2. Merges. Most join separate lineages, but a third join two
        #    branches that share an ancestor -- merging two fine-tunes of one
        #    base is among the most common real cases. That matters here: if
        #    every merge were cross-family, no merge could ever have two
        #    parents affected by the same patient zero, and the planner's
        #    `blocked_on_compromised_merge_parent` class would be unreachable
        #    in the only situation it was written for.
        for _ in range(merge_budget[gen - 1]):
            same_family = rng.random() < 0.35
            pair = (_pick_same_family_pair(pool, families, rng) if same_family
                    else _pick_cross_family_pair(pool, families, rng))
            pair = pair or _pick_cross_family_pair(pool, families, rng)
            if pair is None:
                continue
            a, b = pair
            mid = f"merge-g{gen}-{len(records)}"
            records[mid] = ModelRecord(
                id=mid, hash=fake_hash(rng), parent_ids=[a, b], operation="merge",
                signed=rng.random() < 0.6, is_patient_zero=False, generation=gen,
            )
            families[mid] = families[a] | families[b]
            created.append(mid)

        # 3. LoRA adapters, served against a base rather than merged into it.
        #    signed is forced True: the point of this case is that the
        #    adapter's own signature verifies perfectly while the base
        #    underneath it is compromised.
        bases = [r for r in pool if records[r].operation != "compose"]
        for _ in range(adapter_budget[gen - 1]):
            if not bases:
                break
            base = rng.choice(bases)
            aid = f"lora-g{gen}-{len(records)}"
            records[aid] = ModelRecord(
                id=aid, hash=fake_hash(rng), parent_ids=[base], operation="compose",
                signed=True, is_patient_zero=False, generation=gen,
            )
            families[aid] = set(families[base])
            created.append(aid)

        # Everything created this generation - merges and adapters included -
        # can be derived from next generation. This is the fix.
        frontier = created
        if not frontier:
            break

    assign_patient_zeros(records, rng, n_root_pz, n_mid_pz)
    return records


def _spread(total, buckets):
    """Distribute `total` items over `buckets` generations as evenly as
    possible, front-loaded, leaving the last generation empty so that
    late-created merges and adapters still have a chance at children."""
    usable = max(1, buckets - 1)
    out = [total // usable] * usable + [0] * (buckets - usable)
    for i in range(total % usable):
        out[i] += 1
    return out


def _pick_cross_family_pair(pool, families, rng, tries=25):
    for _ in range(tries):
        a, b = rng.choice(pool), rng.choice(pool)
        if a != b and not (families[a] & families[b]):
            return a, b
    return None


def _pick_same_family_pair(pool, families, rng, tries=25):
    """Two nodes from the same family: sibling branches off a shared ancestor."""
    for _ in range(tries):
        a, b = rng.choice(pool), rng.choice(pool)
        if a != b and (families[a] & families[b]):
            return a, b
    return None


def assign_patient_zeros(records, rng,
                         n_root_pz=N_ROOT_PATIENT_ZEROS,
                         n_mid_pz=N_MIDCHAIN_PATIENT_ZEROS):
    """Places compromises at two structurally different positions.

    Root patient zeros model a poisoned foundation model or dataset: every
    descendant is affected and there is no clean ancestor to roll back to.
    Mid-chain patient zeros model a compromise introduced during one team's
    derivation, so a clean ancestor still exists upstream and rollback is
    actually possible.

    Mid-chain candidates deliberately include merge and compose nodes, not
    just fine-tunes: a compromise discovered in a merged model, or in an
    adapter, is a case the recoverability taxonomy has to handle.
    """
    roots = [r for r in records.values() if r.operation == "root"]
    for r in rng.sample(roots, min(n_root_pz, len(roots))):
        r.is_patient_zero = True

    child_map = {}
    for rid, rec in records.items():
        for p in rec.parent_ids:
            child_map.setdefault(p, []).append(rid)

    # Nodes already downstream of a root patient zero are excluded, otherwise
    # the root and mid-chain cases confound each other in the evaluation.
    contaminated, stack = set(), [r.id for r in records.values() if r.is_patient_zero]
    while stack:
        for child in child_map.get(stack.pop(), []):
            if child not in contaminated:
                contaminated.add(child)
                stack.append(child)

    descendant_count = {}
    for rid in records:
        seen, stack = set(), list(child_map.get(rid, []))
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(child_map.get(cur, []))
        descendant_count[rid] = len(seen)

    eligible = [
        r for r in records.values()
        if r.operation != "root" and r.generation >= 1
        and r.id not in contaminated and not r.is_patient_zero
        and descendant_count.get(r.id, 0) >= 2
    ]
    for r in rng.sample(eligible, min(n_mid_pz, len(eligible))):
        r.is_patient_zero = True


def build_tracked_edges(records, rng, drop_p=UNTRACKED_EDGE_PROBABILITY):
    """Drop each true edge independently with probability drop_p."""
    return {
        rid: [p for p in rec.parent_ids if rng.random() >= drop_p]
        for rid, rec in records.items()
    }


# Per-operation untracked rates. The uniform model above assumes every kind of
# derivation is equally likely to go unrecorded, which is unrealistic: an
# automated quantization step is plausibly attested at a very different rate
# than a deliberate, reviewed model merge. Each scenario is calibrated against
# the measured edge mix so all of them drop the same fraction overall, which is
# what makes any difference in outcome attributable to the STRUCTURE of the
# missingness rather than to its magnitude.
#
# Calibrated 2026-09-07 against the CURRENT generator's mix (fine-tune 0.571,
# quantize 0.245, merge 0.130, compose 0.054, over 200 seeds). Expected total
# drop: uniform 0.1500, routine 0.1491, deliberate 0.1503, composition 0.1494.
# An earlier set was calibrated against a previous generator and drifted to
# 13.2-15.9% once the topology changed, at which point the scenarios differed
# in magnitude too and the comparison no longer isolated structure. Recalibrate
# whenever the generator changes; sweep.py reports realized drop fractions, so
# the drift is visible rather than silent.
DROP_SCENARIOS = {
    "uniform": {"fine-tune": 0.15, "quantize": 0.15, "merge": 0.15, "compose": 0.15},
    "routine_ops_underreported": {
        "fine-tune": 0.0633, "quantize": 0.35, "merge": 0.0633, "compose": 0.35},
    "deliberate_ops_underreported": {
        "fine-tune": 0.14, "quantize": 0.04, "merge": 0.45, "compose": 0.04},
    "composition_blind_spot": {
        "fine-tune": 0.1122, "quantize": 0.1122, "merge": 0.1122, "compose": 0.80},
}


def build_tracked_edges_by_op(records, rng, rates):
    """Drop each true edge with a probability that depends on the operation
    that produced the child. Edge (parent -> child) is typed by the child's
    operation, which is exactly the derivation whose attestation is missing."""
    return {
        rid: [p for p in rec.parent_ids if rng.random() >= rates.get(rec.operation, 0.0)]
        for rid, rec in records.items()
    }


# An adversary who wants to stay hidden does not drop edges at random: they
# decline to attest the derivations that would expose them. This withholds the
# edges immediately below patient zero, which is the cheapest possible way to
# truncate a blast radius, and it is the missingness model the threat model
# implies but the uniform sweep does not test.
def build_tracked_edges_adversarial(records, rng, drop_p, hops=2, focus=0.9):
    """Edges within `hops` of a patient zero are withheld preferentially, at a
    rate up to `focus`, with the remainder of the budget spread over the rest
    of the graph so the overall expected drop fraction still equals drop_p.

    `focus` is a CEILING, not the realized rate. The near set is usually large
    enough that spending the whole budget on it would exceed `focus * n_near`,
    so the near rate is capped at `budget / n_near` and the far rate falls to
    roughly zero. Measured over 300 seeds at drop_p=0.15: the cap binds in
    about 90% of them, giving a mean near-edge rate of 0.60 and a mean
    far-edge rate of 0.003. The arm is therefore best described as "spend
    the entire recording budget within `hops` of every patient zero", not as
    "90% withholding against a benign background".

    Note what this assumes about the adversary. Edges below patient zero are
    attested by whoever performed those derivations, so an adversary who
    merely publishes a poisoned artifact cannot withhold them. This models
    capability 3 - control of derivation pipelines - or an operator concealing
    their own exposure. It is also not the optimal strategy: an adversary
    spending the budget on hop-1 edges first, and never on a hop-2 edge below
    an already-severed hop-1 edge, would do strictly more damage. The measured
    figure is therefore a lower bound on adversarial impact."""
    child_map = {}
    for rid, rec in records.items():
        for p in rec.parent_ids:
            child_map.setdefault(p, []).append(rid)

    near, frontier = set(), [r.id for r in records.values() if r.is_patient_zero]
    for _ in range(hops):
        nxt = []
        for node in frontier:
            for c in child_map.get(node, []):
                if (node, c) not in near:
                    near.add((node, c))
                    nxt.append(c)
        frontier = nxt

    all_edges = [(p, rid) for rid, rec in records.items() for p in rec.parent_ids]
    n_near = sum(1 for e in all_edges if e in near)
    n_far = len(all_edges) - n_near
    budget = drop_p * len(all_edges)

    # Spend the budget on near-patient-zero edges first, but never overspend
    # it: if the near set alone would exceed the budget, cap the near rate so
    # the adversary drops exactly as many edges as benign loss would. Without
    # this the adversarial arm drops more edges overall and the comparison
    # measures budget, not strategy.
    near_p = min(focus, budget / n_near) if n_near else 0.0
    far_p = 0.0 if n_far == 0 else max(0.0, min(1.0, (budget - near_p * n_near) / n_far))

    return {
        rid: [p for p in rec.parent_ids
              if rng.random() >= (near_p if (p, rid) in near else far_p)]
        for rid, rec in records.items()
    }


if __name__ == "__main__":
    records = generate()
    rng = random.Random(SEED + 1)
    tracked_parents = build_tracked_edges(records, rng)

    with open("data/true_lineage.json", "w") as f:
        json.dump({rid: r.to_dict() for rid, r in records.items()}, f, indent=2)

    tracked_out = {}
    for rid, r in records.items():
        d = r.to_dict()
        d["parent_ids"] = tracked_parents[rid]
        tracked_out[rid] = d
    with open("data/tracked_lineage.json", "w") as f:
        json.dump(tracked_out, f, indent=2)

    pz = [r for r in records.values() if r.is_patient_zero]
    n_true = sum(len(r.parent_ids) for r in records.values())
    n_tracked = sum(len(p) for p in tracked_parents.values())
    ops = {}
    for r in records.values():
        ops[r.operation] = ops.get(r.operation, 0) + 1
    print(f"Generated {len(records)} models across {N_FAMILIES} families")
    print(f"  by operation: {ops}")
    print(f"Patient zeros: {len(pz)} "
          f"({sum(1 for r in pz if r.operation == 'root')} at roots, "
          f"{sum(1 for r in pz if r.operation != 'root')} mid-chain; "
          f"ops: {sorted(r.operation for r in pz)})")
    print(f"True edges: {n_true}, tracked edges: {n_tracked} "
          f"({n_true - n_tracked} dropped, "
          f"{100 * (n_true - n_tracked) / n_true:.1f}% untracked)")
