"""
Evaluation harness, two questions.

1. Detection: given only the tracked lineage graph (what a real system
   would actually have on record), how much of the true blast radius do we
   find, and how much silently disappears because an edge was never
   recorded? False negatives are the dangerous failure mode: a model gets
   treated as safe when it isn't.

2. Recovery: for the models we do find, can rollback actually fix them?
   This splits by where the compromise was introduced, and the split is
   the point.
"""
import json
from collections import defaultdict
from lineage_graph import load_graph, blast_radius
from recovery import recovery_plan, summarize_plans

TRUE_PATH = "data/true_lineage.json"
TRACKED_PATH = "data/tracked_lineage.json"


def run_evaluation():
    return evaluate_graphs(load_graph(TRUE_PATH), load_graph(TRACKED_PATH))


def evaluate_graphs(true_graph, tracked_graph):
    patient_zeros = [n for n, d in true_graph.nodes(data=True) if d.get("is_patient_zero")]

    results = []
    for pz in patient_zeros:
        true_affected, _ = blast_radius(true_graph, pz)
        tracked_affected, elapsed = blast_radius(tracked_graph, pz)

        false_negatives = true_affected - tracked_affected
        false_positives = tracked_affected - true_affected
        found = true_affected & tracked_affected
        completeness = len(found) / len(true_affected) if true_affected else 1.0

        plans = recovery_plan(tracked_graph, pz, tracked_affected)
        plan_summary = summarize_plans(plans)

        # The same missing edges that hide affected models also hide the
        # clean ancestors those models could roll back to. Recompute the
        # plan against ground truth to count how many models get reported
        # as unrecoverable when a valid rollback target actually existed.
        true_plans = recovery_plan(true_graph, pz, true_affected)
        true_status = {p["model"]: p["status"] for p in true_plans}
        false_unrecoverable = [
            p["model"] for p in plans
            if p["status"] == "unrecoverable_by_rollback"
            and true_status.get(p["model"]) == "recoverable_by_rollback"
        ]

        # Validate each PROPOSED plan directly against true dependencies.
        #
        # This is the quantity the paper cares about, and it is NOT the same as
        # comparing the tracked planner's verdict to the true planner's. The
        # two planners select targets independently, so an added edge can make
        # the true planner pick a different, nearer target whose plan happens
        # to be blocked; that says nothing about whether the plan actually
        # proposed reuses a compromised input. Walk the submitted path instead,
        # treating nodes the plan has already rebuilt as repaired, and ask
        # whether any TRUE parent of a rebuilt node is compromised and left
        # unrepaired. That is the mechanism "reintroduces the compromise"
        # names.
        unsafe_plans = []
        for pl in plans:
            if pl["status"] != "recoverable_by_rollback" or not pl["rebuild_steps"]:
                continue
            rebuilt, bad = set(), False
            for step in pl["rebuild_steps"]:
                a, rest = step.split(" --", 1)
                _, b = rest.split("--> ")
                a, b = a.strip(), b.strip()
                rebuilt.add(a)
                for par in true_graph.predecessors(b):
                    if par in true_affected and par not in rebuilt:
                        bad = True
                rebuilt.add(b)
            if bad:
                unsafe_plans.append(pl["model"])

        # The symmetric error, and the more dangerous one. The tracked graph
        # says a model can be rolled back; the truth is that it cannot, because
        # an edge that would have revealed a compromised merge parent, or the
        # compromise of the target itself, was never recorded. Acting on this
        # verdict reintroduces the compromise. The previous generator made it
        # unobservable by construction, since merges had no descendants.
        false_recoverable = [
            p["model"] for p in plans
            if p["status"] == "recoverable_by_rollback"
            and true_status.get(p["model"]) in (
                "unrecoverable_by_rollback", "blocked_on_compromised_merge_parent")
        ]

        # Both classifications also computed under the conservative rule, so
        # the precision cost of provability can be reported rather than
        # asserted.
        strict_plans = recovery_plan(tracked_graph, pz, tracked_affected, strict=True)
        strict_status = {p["model"]: p["status"] for p in strict_plans}
        strict_blocked_relaxable = [
            m for m, st in strict_status.items()
            if st == "blocked_on_compromised_merge_parent"
            and {p["model"]: p["status"] for p in plans}.get(m)
            == "recoverable_by_rollback"
        ]
        false_recoverable_strict = [
            p["model"] for p in strict_plans
            if p["status"] == "recoverable_by_rollback"
            and true_status.get(p["model"]) in (
                "unrecoverable_by_rollback", "blocked_on_compromised_merge_parent")
        ]

        # Planner soundness: whenever the tracked planner proposes a rollback
        # target, is that target genuinely outside the TRUE blast radius? A
        # violation means the plan rebuilds from something itself compromised.
        unsound_targets = [
            (p["model"], p["recovery_target"]) for p in plans
            if p["status"] == "recoverable_by_rollback"
            and p["recovery_target"] in true_affected
        ]

        results.append({
            "patient_zero": pz,
            "pz_position": true_graph.nodes[pz].get("operation"),
            "pz_generation": true_graph.nodes[pz].get("generation"),
            "true_blast_radius_size": len(true_affected),
            "tracked_blast_radius_size": len(tracked_affected),
            "false_negatives": sorted(false_negatives),
            "false_positives": sorted(false_positives),
            "completeness": round(completeness, 4),
            "false_negative_count": len(false_negatives),
            "query_time_ms": round(elapsed * 1000, 4),
            "recovery": plan_summary,
            "false_unrecoverable": false_unrecoverable,
            "false_unrecoverable_count": len(false_unrecoverable),
            "false_recoverable": false_recoverable,
            "false_recoverable_count": len(false_recoverable),
            "unsafe_plan_count": len(unsafe_plans),
            "unsafe_plans": unsafe_plans,
            "false_recoverable_count_strict": len(false_recoverable_strict),
            "strict_blocked_relaxable_count": len(strict_blocked_relaxable),
            "strict_blocked_total": sum(
                1 for st in strict_status.values()
                if st == "blocked_on_compromised_merge_parent"),
            "unsound_targets": unsound_targets,
            "unsound_target_count": len(unsound_targets),
            "recovery_plans": plans,
        })

    return results


def summarize(results):
    n = len(results)
    total_true = sum(r["true_blast_radius_size"] for r in results)
    total_fn = sum(r["false_negative_count"] for r in results)
    total_fp = sum(len(r["false_positives"]) for r in results)

    by_position = defaultdict(
        lambda: {"patient_zeros": 0, "recoverable": 0, "unrecoverable": 0, "blocked": 0}
    )
    for r in results:
        pos = "root" if r["pz_position"] == "root" else "mid-chain"
        stats = r["recovery"]["by_status"]
        by_position[pos]["patient_zeros"] += 1
        by_position[pos]["recoverable"] += stats.get("recoverable_by_rollback", 0)
        by_position[pos]["unrecoverable"] += stats.get("unrecoverable_by_rollback", 0)
        by_position[pos]["blocked"] += stats.get("blocked_on_compromised_merge_parent", 0)

    return {
        "n_patient_zeros_evaluated": n,
        "average_completeness": round(sum(r["completeness"] for r in results) / n, 4),
        "total_true_affected_models": total_true,
        "total_false_negatives": total_fn,
        "total_false_positives": total_fp,
        "overall_recall": round((total_true - total_fn) / total_true, 4) if total_true else 1.0,
        "average_query_time_ms": round(sum(r["query_time_ms"] for r in results) / n, 4),
        "recovery_by_pz_position": {k: dict(v) for k, v in by_position.items()},
        "total_false_unrecoverable": sum(r["false_unrecoverable_count"] for r in results),
        "total_false_recoverable": sum(r["false_recoverable_count"] for r in results),
        "total_unsafe_plans": sum(r["unsafe_plan_count"] for r in results),
        "total_false_recoverable_strict": sum(
            r["false_recoverable_count_strict"] for r in results),
        "total_unsound_targets": sum(r["unsound_target_count"] for r in results),
    }


if __name__ == "__main__":
    results = run_evaluation()
    summary = summarize(results)

    with open("results/evaluation.json", "w") as f:
        json.dump({"per_patient_zero": results, "summary": summary}, f, indent=2)

    print("DETECTION\n")
    for r in results:
        pos = "root" if r["pz_position"] == "root" else f"mid-chain (gen {r['pz_generation']})"
        print(f"  {r['patient_zero']} [{pos}]: true = {r['true_blast_radius_size']}, "
              f"found = {r['true_blast_radius_size'] - r['false_negative_count']}, "
              f"missed = {r['false_negative_count']}, "
              f"completeness = {r['completeness'] * 100:.1f}%")

    print("\nRECOVERY\n")
    for r in results:
        pos = "root" if r["pz_position"] == "root" else "mid-chain"
        st = r["recovery"]["by_status"]
        print(f"  {r['patient_zero']} [{pos}]: "
              f"recoverable = {st.get('recoverable_by_rollback', 0)}, "
              f"unrecoverable = {st.get('unrecoverable_by_rollback', 0)}, "
              f"blocked on merge = {st.get('blocked_on_compromised_merge_parent', 0)}"
              + (f"  <- {r['false_unrecoverable_count']} FALSELY reported unrecoverable"
                 if r["false_unrecoverable_count"] else ""))

    print("\nSUMMARY\n")
    for k, v in summary.items():
        print(f"  {k}: {v}")
