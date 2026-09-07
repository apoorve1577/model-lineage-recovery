"""
Statistical sweep over the one parameter the whole result depends on:
how much of the true lineage a system actually has on record.

The single-seed evaluation in evaluate.py demonstrates the mechanism on one
draw. It cannot support a claim about how the architecture behaves, because
every number it reports is one sample of a stochastic process: the graph
structure is random, and which edges go unrecorded is random on top of that.

This runs the whole pipeline N times per untracked-edge rate, over an
independently regenerated lineage each trial, and reports means with 95%
confidence intervals. Three things it establishes that one seed cannot:

  1. Recall degrades with the untracked-edge rate along a measurable curve,
     not at a single point. The curve is what an operator needs, because
     their own recording gap is not 15%, it is whatever it is.
  2. False positives are identically zero at every rate, over every trial.
     That is the Soundness proposition holding empirically, and at
     drop_p = 0 the harness is also validated: recall must be exactly 1.0.
  3. Not all lineage edges are worth the same. Single-parent derivations
     are cut points; merge edges are largely redundant. That is an
     instrumentation priority an operator can act on.
  4. False-unrecoverable verdicts, the more damaging error, grow
     SUPERLINEARLY in the detection miss rate. Each marginal unit of
     recording gap does disproportionately more damage to recovery than to
     detection, so partial lineage capture degrades worse than linearly.

run_scenario_sweep, drops the independence-and-uniformity
assumption entirely and re-runs everything under per-operation missingness.
"""
import json
import math
import time

from generate_dataset import generate, build_tracked_edges
from lineage_graph import graph_from_records, blast_radius
from evaluate import evaluate_graphs
import random

DROP_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_TRIALS = 300
SCALE_FAMILIES = [7, 25, 100, 400]
SCALE_TRIALS = 15
SCALE_DROP = 0.15


def build_pair(seed, drop_p, n_families=7):
    """One trial: a freshly generated true lineage, and the tracked view of
    it after independently dropping each edge with probability drop_p."""
    records = generate(seed=seed, n_families=n_families)
    rng = random.Random(seed * 7919 + 13)
    tracked_parents = build_tracked_edges(records, rng, drop_p=drop_p)

    true_recs = {rid: r.to_dict() for rid, r in records.items()}
    tracked_recs = {}
    for rid, r in records.items():
        d = r.to_dict()
        d["parent_ids"] = tracked_parents[rid]
        tracked_recs[rid] = d

    n_true_edges = sum(len(r.parent_ids) for r in records.values())
    n_tracked_edges = sum(len(p) for p in tracked_parents.values())
    return (graph_from_records(true_recs), graph_from_records(tracked_recs),
            n_true_edges, n_tracked_edges)


def trial_metrics(results):
    """Collapse one trial's per-patient-zero results into trial-level scalars.
    Pooled over patient zeros within the trial, so each trial contributes one
    independent observation to the confidence interval."""
    total_true = sum(r["true_blast_radius_size"] for r in results)
    total_fn = sum(r["false_negative_count"] for r in results)
    total_fp = sum(len(r["false_positives"]) for r in results)
    total_false_unrec = sum(r["false_unrecoverable_count"] for r in results)

    # Denominator for the false-unrecoverable rate: models the operator was
    # actually handed a verdict on, i.e. inside the tracked blast radius and
    # not the patient zero itself.
    n_verdicts = sum(
        r["recovery"]["models_in_blast_radius"]
        - r["recovery"]["by_status"].get("patient_zero", 0)
        for r in results
    )

    rec = {"root": {"recoverable": 0, "unrecoverable": 0, "blocked": 0},
           "mid-chain": {"recoverable": 0, "unrecoverable": 0, "blocked": 0}}
    for r in results:
        pos = "root" if r["pz_position"] == "root" else "mid-chain"
        st = r["recovery"]["by_status"]
        rec[pos]["recoverable"] += st.get("recoverable_by_rollback", 0)
        rec[pos]["unrecoverable"] += st.get("unrecoverable_by_rollback", 0)
        rec[pos]["blocked"] += st.get("blocked_on_compromised_merge_parent", 0)

    return {
        "recall": (total_true - total_fn) / total_true if total_true else 1.0,
        "false_positives": total_fp,
        "false_unrecoverable": total_false_unrec,
        "false_unrecoverable_rate": total_false_unrec / n_verdicts if n_verdicts else 0.0,
        "n_verdicts": n_verdicts,
        "true_affected": total_true,
        "query_time_ms": sum(r["query_time_ms"] for r in results) / len(results),
        "recovery_by_position": rec,
    }


def mean_ci(xs, confidence=0.95):
    """Mean with a normal-approximation 95% CI over trial-level observations.
    N is large enough (300) that the t-correction is immaterial."""
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0, m, m
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    z = 1.96
    return m, z * se, m - z * se, m + z * se


def run_drop_sweep():
    rows = []
    for drop_p in DROP_RATES:
        trials = []
        actual_drop = []
        for i in range(N_TRIALS):
            seed = 1000 + i
            tg, kg, n_true_e, n_tracked_e = build_pair(seed, drop_p)
            trials.append(trial_metrics(evaluate_graphs(tg, kg)))
            actual_drop.append((n_true_e - n_tracked_e) / n_true_e)

        recalls = [t["recall"] for t in trials]
        fur = [t["false_unrecoverable_rate"] for t in trials]
        m_r, h_r, lo_r, hi_r = mean_ci(recalls)
        m_f, h_f, lo_f, hi_f = mean_ci(fur)

        # Pooled recoverable fraction by patient-zero position.
        pooled = {"root": {"recoverable": 0, "unrecoverable": 0, "blocked": 0},
                  "mid-chain": {"recoverable": 0, "unrecoverable": 0, "blocked": 0}}
        for t in trials:
            for pos in pooled:
                for k in pooled[pos]:
                    pooled[pos][k] += t["recovery_by_position"][pos][k]

        rows.append({
            "drop_p": drop_p,
            "n_trials": N_TRIALS,
            "mean_actual_untracked_fraction": round(sum(actual_drop) / len(actual_drop), 4),
            "recall_mean": round(m_r, 4),
            "recall_ci95_halfwidth": round(h_r, 4),
            "recall_ci95": [round(lo_r, 4), round(hi_r, 4)],
            "recall_min": round(min(recalls), 4),
            "total_false_positives_across_all_trials": sum(t["false_positives"] for t in trials),
            "false_unrecoverable_rate_mean": round(m_f, 4),
            "false_unrecoverable_rate_ci95": [round(lo_f, 4), round(hi_f, 4)],
            "total_false_unrecoverable": sum(t["false_unrecoverable"] for t in trials),
            "recovery_pooled_by_pz_position": pooled,
        })
        print(f"  drop_p={drop_p:.2f}  recall={m_r:.3f} +/- {h_r:.3f}  "
              f"FP={sum(t['false_positives'] for t in trials)}  "
              f"false-unrec rate={m_f:.4f} +/- {h_f:.4f}")
    return rows


def run_scale_sweep():
    rows = []
    for n_fam in SCALE_FAMILIES:
        nodes, edges, qtimes = [], [], []
        for i in range(SCALE_TRIALS):
            t0 = time.perf_counter()
            tg, kg, n_true_e, _ = build_pair(2000 + i, SCALE_DROP, n_families=n_fam)
            res = evaluate_graphs(tg, kg)
            nodes.append(kg.number_of_nodes())
            edges.append(kg.number_of_edges())
            qtimes.append(sum(r["query_time_ms"] for r in res) / max(len(res), 1))
        m_q, h_q, _, _ = mean_ci(qtimes)
        rows.append({
            "n_families": n_fam,
            "mean_nodes": round(sum(nodes) / len(nodes), 1),
            "mean_edges": round(sum(edges) / len(edges), 1),
            "mean_blast_radius_query_ms": round(m_q, 5),
            "query_ms_ci95_halfwidth": round(h_q, 5),
        })
        print(f"  families={n_fam:>4}  nodes={rows[-1]['mean_nodes']:>7}  "
              f"query={m_q * 1000:.2f} us +/- {h_q * 1000:.2f}")
    return rows



# ---------------------------------------------------------------------------
# Non-uniform missingness. Everything above assumes each edge is equally
# likely to go unrecorded, which the paper flags as its weakest assumption.
# This tests whether the results survive when they are not.
# ---------------------------------------------------------------------------

from generate_dataset import DROP_SCENARIOS, build_tracked_edges_by_op


def build_pair_by_op(seed, rates, n_families=7):
    records = generate(seed=seed, n_families=n_families)
    rng = random.Random(seed * 7919 + 13)
    tracked_parents = build_tracked_edges_by_op(records, rng, rates)

    true_recs = {rid: r.to_dict() for rid, r in records.items()}
    tracked_recs = {}
    for rid, r in records.items():
        d = r.to_dict()
        d["parent_ids"] = tracked_parents[rid]
        tracked_recs[rid] = d

    n_true = sum(len(r.parent_ids) for r in records.values())
    n_tracked = sum(len(p) for p in tracked_parents.values())
    return (graph_from_records(true_recs), graph_from_records(tracked_recs),
            n_true, n_tracked)


def run_scenario_sweep(n_trials=N_TRIALS):
    rows = []
    for name, rates in DROP_SCENARIOS.items():
        trials, actual = [], []
        for i in range(n_trials):
            tg, kg, n_true_e, n_tracked_e = build_pair_by_op(1000 + i, rates)
            trials.append(trial_metrics(evaluate_graphs(tg, kg)))
            actual.append((n_true_e - n_tracked_e) / n_true_e)

        m_r, h_r, lo_r, hi_r = mean_ci([t["recall"] for t in trials])
        m_f, h_f, lo_f, hi_f = mean_ci([t["false_unrecoverable_rate"] for t in trials])
        rows.append({
            "scenario": name,
            "per_operation_drop_rates": rates,
            "n_trials": n_trials,
            "mean_actual_untracked_fraction": round(sum(actual) / len(actual), 4),
            "recall_mean": round(m_r, 4),
            "recall_ci95": [round(lo_r, 4), round(hi_r, 4)],
            "false_unrecoverable_rate_mean": round(m_f, 4),
            "false_unrecoverable_rate_ci95": [round(lo_f, 4), round(hi_f, 4)],
            "total_false_positives_across_all_trials": sum(t["false_positives"] for t in trials),
        })
        print(f"  {name:<30} untracked={rows[-1]['mean_actual_untracked_fraction']:.3f}  "
              f"recall={m_r:.3f} +/- {h_r:.3f}  "
              f"false-unrec={m_f:.4f} +/- {h_f:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Edge criticality. The scenario sweep shows that WHICH edges go missing
# changes the outcome, so the obvious next question is why. This measures the
# marginal cost of losing exactly one edge, by operation type: regenerate the
# true blast radius, delete a single edge, and count how many models fall out.
# ---------------------------------------------------------------------------

def run_edge_criticality(n_graphs=120):
    """Mean models dropped from the true blast radius per single lost edge.

    The result is that single-parent derivations (fine-tune, quantize) are
    cut points: losing one severs everything below it. Merge edges are not,
    because compromise propagates as a logical OR across parents, so the
    other parent still reaches the node. The same parental redundancy that
    makes merges hard for per-artifact verification also makes their lineage
    edges the cheapest ones to lose."""
    from collections import defaultdict
    cost = defaultdict(list)

    for seed in range(1000, 1000 + n_graphs):
        records = generate(seed=seed)
        recs = {rid: r.to_dict() for rid, r in records.items()}
        full = graph_from_records(recs)
        pzs = [n for n, a in full.nodes(data=True) if a.get("is_patient_zero")]
        base = {pz: blast_radius(full, pz)[0] for pz in pzs}

        for rid, r in records.items():
            for par in r.parent_ids:
                perturbed = {k: dict(v) for k, v in recs.items()}
                perturbed[rid]["parent_ids"] = [
                    p for p in recs[rid]["parent_ids"] if p != par
                ]
                g2 = graph_from_records(perturbed)
                cost[r.operation].append(
                    sum(len(base[pz] - blast_radius(g2, pz)[0]) for pz in pzs)
                )

    rows = []
    for op in ("fine-tune", "quantize", "merge", "compose"):
        v = cost[op]
        m, h, lo, hi = mean_ci(v)
        rows.append({
            "edge_type": op,
            "n_edges_perturbed": len(v),
            "mean_models_lost_per_missing_edge": round(m, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "fraction_of_edges_whose_loss_costs_nothing":
                round(sum(1 for x in v if x == 0) / len(v), 4),
            "is_cut_point": op in ("fine-tune", "quantize"),
        })
        print(f"  {op:<12} mean models lost = {m:.3f} +/- {h:.3f}   "
              f"({100 * rows[-1]['fraction_of_edges_whose_loss_costs_nothing']:.1f}% cost nothing)")
    return rows


if __name__ == "__main__":
    print(f"DROP-RATE SWEEP ({N_TRIALS} independent lineages per rate)\n")
    drop_rows = run_drop_sweep()

    print(f"\nSCALE SWEEP ({SCALE_TRIALS} trials per size, drop_p={SCALE_DROP})\n")
    scale_rows = run_scale_sweep()

    print(f"\nNON-UNIFORM MISSINGNESS ({N_TRIALS} trials per scenario, "
          f"all calibrated to ~15% of edges dropped overall)\n")
    scenario_rows = run_scenario_sweep()

    print("\nEDGE CRITICALITY (marginal cost of one missing edge, by type)\n")
    criticality_rows = run_edge_criticality()

    zero = next(r for r in drop_rows if r["drop_p"] == 0.0)
    checks = {
        "recall_is_exactly_1_at_zero_drop": zero["recall_mean"] == 1.0,
        "no_false_unrecoverable_at_zero_drop": zero["total_false_unrecoverable"] == 0,
        "zero_false_positives_at_every_rate": all(
            r["total_false_positives_across_all_trials"] == 0 for r in drop_rows
        ),
        "zero_false_positives_under_non_uniform_missingness": all(
            r["total_false_positives_across_all_trials"] == 0 for r in scenario_rows
        ),
    }
    print("\nSANITY CHECKS\n")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    with open("results/sweep.json", "w") as f:
        json.dump({"drop_rate_sweep": drop_rows,
                   "scale_sweep": scale_rows,
                   "non_uniform_missingness_sweep": scenario_rows,
                   "edge_criticality": criticality_rows,
                   "sanity_checks": checks}, f, indent=2)
    print("\nWrote results/sweep.json")
