"""
The recovery half. Blast radius tells an operator what is affected;
this tells them what to actually do about each affected model.

For every model in the blast radius, find the nearest ancestor that is
both outside the blast radius and independently verifiable (signed), then
report the operation chain needed to rebuild from it and what that costs.

The interesting result is that this is not always possible. When patient
zero is a root, no clean ancestor exists anywhere above an affected model,
so rollback is undefined and the only option is rebuilding from scratch.
When patient zero sits mid-chain, a clean ancestor does exist upstream and
rollback is a real option. Existing tooling makes no distinction between
these two cases, and they demand completely different incident responses.
"""
import networkx as nx
from models import CHEAP_TO_REBUILD, REBUILD_COST


def nearest_clean_ancestor(graph, node_id, affected):
    """Breadth-first walk upward for the closest ancestor that is outside
    the blast radius and signed. Returns (ancestor_id, path) or (None, None).

    Path is the operation chain from that ancestor down to node_id, which
    is what an operator would actually have to re-execute.
    """
    from collections import deque

    queue = deque([(node_id, [])])
    seen = {node_id}

    while queue:
        current, path_down = queue.popleft()
        for parent in graph.predecessors(current):
            if parent in seen:
                continue
            seen.add(parent)
            op = graph.nodes[current].get("operation")
            new_path = [(parent, current, op)] + path_down
            parent_clean = parent not in affected
            parent_signed = graph.nodes[parent].get("signed", False)
            if parent_clean and parent_signed:
                return parent, new_path
            queue.append((parent, new_path))

    return None, None


def recovery_plan(graph, patient_zero, affected, strict=False):
    """Builds a per-model recovery plan for an entire blast radius.

    `strict` selects the blocking rule; see the comment in the merge branch
    below. False (default) is the paper's definition: only a parent that the
    rebuild plan does not itself repair can block. True is the conservative
    variant, under which a `recoverable` verdict is provably never wrong.
    """
    plans = []

    for model_id in sorted(affected):
        if model_id == patient_zero:
            plans.append({
                "model": model_id,
                "status": "patient_zero",
                "action": "quarantine and investigate: this is the origin",
                "recovery_target": None,
                "rebuild_steps": None,
                "rebuild_cost": None,
            })
            continue

        ancestor, path = nearest_clean_ancestor(graph, model_id, affected)

        if ancestor is None:
            plans.append({
                "model": model_id,
                "status": "unrecoverable_by_rollback",
                "action": "no clean signed ancestor exists: full rebuild from a trusted base required",
                "recovery_target": None,
                "rebuild_steps": None,
                "rebuild_cost": "maximum: nothing upstream can be trusted",
            })
            continue

        ops = [op for _, _, op in path]
        all_cheap = all(op in CHEAP_TO_REBUILD for op in ops)

        # A merge can only be rebuilt once EVERY one of its parents is clean.
        # Finding one clean ancestor through a single parent is not enough,
        # and treating it as sufficient is a real correctness trap: the
        # upward search happily returns a clean parent from an unrelated
        # family while the merge's other parent is still compromised.
        #
        # WHICH parent counts is the subtle part, and getting it wrong changes
        # what the planner claims. The parent that lies ON the rebuild path is
        # rebuilt by this very plan, so it does not block anything; only a
        # parent the plan does not touch can. That is the `strict=False`
        # default and it is what the paper defines.
        #
        # `strict=True` also counts the path-parent, which is always affected
        # inside a blast radius. That makes the classifier conservative: it
        # blocks work that could in fact proceed, in exchange for a provable
        # guarantee that a `recoverable` verdict is never wrong (a path with no
        # merge-typed node inside the blast radius has in-degree 1 throughout,
        # so the true planner walks the same path and returns the same answer).
        # The precision cost is real - at 15% untracked edges, 3.2% of strict
        # `blocked` verdicts have every off-path parent clean - so the two
        # settings trade precision against provability rather than one
        # dominating the other.
        blocked_merges = [
            child for parent_on_path, child, op in path
            if op == "merge" and any(
                p in affected and (strict or p != parent_on_path)
                for p in graph.predecessors(child)
            )
        ]

        if blocked_merges:
            plans.append({
                "model": model_id,
                "status": "blocked_on_compromised_merge_parent",
                "action": (
                    f"cannot rebuild yet: merge node(s) {blocked_merges} have a "
                    f"compromised parent that this plan does not rebuild. "
                    f"Those parents must be recovered first."
                ),
                "recovery_target": None,
                "rebuild_steps": None,
                "rebuild_cost": "deferred: blocked behind another model's recovery",
                "blocked_on_merge_parents": blocked_merges,
            })
            continue

        plans.append({
            "model": model_id,
            "status": "recoverable_by_rollback",
            "action": f"rebuild from {ancestor}",
            "recovery_target": ancestor,
            "rebuild_steps": [f"{a} --{op}--> {b}" for a, b, op in path],
            "rebuild_cost": (
                "cheap: no retraining required, only replay or re-composition"
                if all_cheap else
                "; ".join(sorted({REBUILD_COST.get(op, op) for op in ops}))
            ),
            "blocked_on_merge_parents": None,
        })

    return plans


def summarize_plans(plans):
    total = len(plans)
    by_status = {}
    for p in plans:
        by_status[p["status"]] = by_status.get(p["status"], 0) + 1
    recoverable = by_status.get("recoverable_by_rollback", 0)
    non_pz = total - by_status.get("patient_zero", 0)
    return {
        "models_in_blast_radius": total,
        "by_status": by_status,
        "rollback_recoverable_fraction": (
            round(recoverable / non_pz, 4) if non_pz else None
        ),
    }
