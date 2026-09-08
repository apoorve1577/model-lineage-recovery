"""
Statistical evaluation. Every trial regenerates the lineage from scratch and
redraws which edges go unrecorded, so each trial is an independent observation
rather than a resample of one graph.

Six experiments:

  1. run_drop_sweep       recall and both planner error classes against the
                          fraction of unrecorded edges
  2. run_edge_criticality marginal cost of one missing edge, by operation type
  3. run_scenario_sweep   non-uniform missingness, calibrated to a fixed total
  4. run_adversarial      strategic missingness: an adversary declines to
                          attest the derivations that would expose them
  5. run_stress           merge density raised, to check that the symmetric
                          error grows and plan soundness survives
  6. bootstrap_slopes     log-log slope of each error class against the
                          detection miss rate, with a CI that respects the
                          paired design

TWO METHODOLOGICAL NOTES, both of which an earlier draft of this evaluation
got wrong.

Common random numbers. Every drop rate reuses trial seeds 1000..1000+N and the
same drop RNG stream, so the set of edges dropped at p=0.10 is a strict subset
of the set dropped at p=0.20. This is deliberate - it is a variance-reduction
design that makes conditions comparable - but it means the drop-rate conditions
are PAIRED, not independent, and a naive regression across them understates
uncertainty. bootstrap_slopes therefore resamples TRIALS, recomputing every
condition from the same resampled trial set, which is the unit that is actually
independent.

Rate versus count. The false-unrecoverable RATE has the number of tracked
verdicts as its denominator, and that denominator shrinks as recording degrades
- fewer artifacts are found, so fewer receive a verdict at all. The rate can
therefore rise while the COUNT of misdirected rebuilds falls. Any claim about
operational or financial cost is a claim about the count. Both are reported.
"""
import json
import math
import random
from collections import defaultdict

from generate_dataset import (generate, build_tracked_edges,
                              build_tracked_edges_by_op,
                              build_tracked_edges_adversarial, DROP_SCENARIOS)
from lineage_graph import graph_from_records, blast_radius
from evaluate import evaluate_graphs

DROP_RATES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
N_TRIALS = 300
N_BOOTSTRAP = 2000


def _pair_from(records, tracked_parents):
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


def build_pair(seed, drop_p, n_families=7):
    records = generate(seed=seed, n_families=n_families)
    rng = random.Random(seed * 7919 + 13)
    return _pair_from(records, build_tracked_edges(records, rng, drop_p=drop_p))


def build_pair_by_op(seed, rates, n_families=7):
    records = generate(seed=seed, n_families=n_families)
    rng = random.Random(seed * 7919 + 13)
    return _pair_from(records, build_tracked_edges_by_op(records, rng, rates))


def build_pair_adversarial(seed, drop_p, n_families=7):
    records = generate(seed=seed, n_families=n_families)
    rng = random.Random(seed * 7919 + 13)
    return _pair_from(records,
                      build_tracked_edges_adversarial(records, rng, drop_p))


def trial_metrics(results):
    """Collapse one trial into scalars, keeping the patient-zero position split.

    Root patient zeros cannot produce a false-unrecoverable verdict: there is
    no clean ancestor above a root, so the truth is `unrecoverable` too and the
    tracked verdict cannot be wrong in that direction. They nonetheless supply
    about two-thirds of the pooled denominator (70% at p=0, falling to 57% at
    p=0.50), diluting the pooled rate roughly threefold relative to the
    mid-chain population where the error can occur at all. Reporting the pooled
    figure alone understates the error where it exists.
    """
    agg = {"root": defaultdict(int), "mid-chain": defaultdict(int)}
    for r in results:
        pos = "root" if r["pz_position"] == "root" else "mid-chain"
        a = agg[pos]
        a["true"] += r["true_blast_radius_size"]
        a["fn"] += r["false_negative_count"]
        a["fp"] += len(r["false_positives"])
        a["fu"] += r["false_unrecoverable_count"]
        a["fr"] += r["false_recoverable_count"]
        a["fr_strict"] += r["false_recoverable_count_strict"]
        a["unsafe"] += r["unsafe_plan_count"]
        a["strict_blocked"] += r["strict_blocked_total"]
        a["strict_relaxable"] += r["strict_blocked_relaxable_count"]
        a["recoverable"] += r["recovery"]["by_status"].get(
            "recoverable_by_rollback", 0)
        a["unsound"] += r["unsound_target_count"]
        a["verdicts"] += (r["recovery"]["models_in_blast_radius"]
                          - r["recovery"]["by_status"].get("patient_zero", 0))

    tot = {k: sum(agg[p][k] for p in agg)
           for k in ("true", "fn", "fp", "fu", "fr", "fr_strict", "unsound",
                     "verdicts", "recoverable", "strict_blocked",
                     "strict_relaxable", "unsafe")}
    return {
        "recall": (tot["true"] - tot["fn"]) / tot["true"] if tot["true"] else 1.0,
        "false_positives": tot["fp"],
        "fu_count": tot["fu"],
        "fr_count": tot["fr"],
        "fr_count_strict": tot["fr_strict"],
        "unsafe_plans": tot["unsafe"],
        "recoverable": tot["recoverable"],
        "strict_blocked": tot["strict_blocked"],
        "strict_relaxable": tot["strict_relaxable"],
        "unsound": tot["unsound"],
        "verdicts": tot["verdicts"],
        "true_affected": tot["true"],
        "fn": tot["fn"],
        "by_position": {p: dict(agg[p]) for p in agg},
    }


def mean_ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, 1.96 * math.sqrt(var / n)


def _slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx, my = sum(lx) / n, sum(ly) / n
    den = sum((a - mx) ** 2 for a in lx)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / den if den else float("nan")


def _points(trials_by_p, idx):
    """Pooled miss rate, FU rate and FU count at each drop rate, over the
    trials named by idx. Pooling (rather than averaging trial ratios) keeps
    the denominator meaningful."""
    miss, rate, count = [], [], []
    for p in DROP_RATES:
        if p == 0.0:
            continue
        ts = [trials_by_p[p][i] for i in idx]
        true_ = sum(t["true_affected"] for t in ts)
        fn = sum(t["fn"] for t in ts)
        fu = sum(t["fu_count"] for t in ts)
        v = sum(t["verdicts"] for t in ts)
        miss.append(fn / true_ if true_ else 0.0)
        rate.append(fu / v if v else 0.0)
        count.append(fu)
    return miss, rate, count


def bootstrap_slopes(trials_by_p, rng=None):
    """Resample TRIALS, not conditions. Because every drop rate is evaluated on
    the same trial seeds, a resampled trial set must be applied to all
    conditions at once; that is what respects the pairing."""
    rng = rng or random.Random(7)
    n = N_TRIALS
    base_miss, base_rate, base_count = _points(trials_by_p, range(n))
    obs = {"rate": _slope(base_miss, base_rate),
           "count": _slope(base_miss, base_count)}

    draws = {"rate": [], "count": []}
    for _ in range(N_BOOTSTRAP):
        idx = [rng.randrange(n) for _ in range(n)]
        m, r, c = _points(trials_by_p, idx)
        if min(m) <= 0 or min(r) <= 0 or min(c) <= 0:
            continue
        draws["rate"].append(_slope(m, r))
        draws["count"].append(_slope(m, c))

    out = {}
    for k, v in draws.items():
        v.sort()
        lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v))]
        out[k] = {"slope": round(obs[k], 4),
                  "ci95": [round(lo, 4), round(hi, 4)],
                  "n_bootstrap": len(v)}
    return out


def run_drop_sweep():
    trials_by_p, rows = {}, []
    for p in DROP_RATES:
        trials = [trial_metrics(evaluate_graphs(*build_pair(1000 + i, p)[:2]))
                  for i in range(N_TRIALS)]
        trials_by_p[p] = trials

        recalls = [t["recall"] for t in trials]
        m_r, h_r = mean_ci(recalls)
        pooled = {}
        for pos in ("root", "mid-chain"):
            fu = sum(t["by_position"][pos]["fu"] for t in trials)
            v = sum(t["by_position"][pos]["verdicts"] for t in trials)
            pooled[pos] = {"fu_count": fu, "verdicts": v,
                           "fu_rate": round(fu / v, 4) if v else None}
        fu_tot = sum(t["fu_count"] for t in trials)
        v_tot = sum(t["verdicts"] for t in trials)

        rows.append({
            "drop_p": p,
            "n_trials": N_TRIALS,
            "recall_mean": round(m_r, 4),
            "recall_ci95_halfwidth": round(h_r, 4),
            "total_false_positives": sum(t["false_positives"] for t in trials),
            "false_unrecoverable_count": fu_tot,
            "false_unrecoverable_rate_pooled": round(fu_tot / v_tot, 4) if v_tot else 0.0,
            "verdicts": v_tot,
            "by_pz_position": pooled,
            "recoverable_verdicts": sum(t["recoverable"] for t in trials),
            # NOT an unsafety rate. The tracked and full-graph planners select
            # targets independently, so an added edge can make the full-graph
            # planner choose a different, nearer target whose plan happens to
            # be blocked. That is disagreement, not danger. `unsafe_plan_*`
            # below is the quantity that answers "does the proposed plan reuse
            # a compromised input".
            "verdict_disagreement_count": sum(t["fr_count"] for t in trials),
            "verdict_disagreement_rate": round(
                sum(t["fr_count"] for t in trials)
                / max(1, sum(t["recoverable"] for t in trials)), 4),
            "false_recoverable_count_strict": sum(t["fr_count_strict"] for t in trials),
            "unsafe_plan_count": sum(t["unsafe_plans"] for t in trials),
            "unsafe_plan_rate": round(
                sum(t["unsafe_plans"] for t in trials)
                / max(1, sum(t["recoverable"] for t in trials)), 5),
            "strict_blocked_total": sum(t["strict_blocked"] for t in trials),
            "strict_blocked_relaxable": sum(t["strict_relaxable"] for t in trials),
            "unsound_target_count": sum(t["unsound"] for t in trials),
        })
        r = rows[-1]
        print(f"  p={p:.2f}  recall={m_r:.3f}+/-{h_r:.3f}  "
              f"FU count={fu_tot:>5}  rate={r['false_unrecoverable_rate_pooled']:.4f}  "
              f"(mid-chain only {pooled['mid-chain']['fu_rate']})  "
              f"disagree={r['verdict_disagreement_count']:>3}/{r['recoverable_verdicts']:<5}"
              f"={r['verdict_disagreement_rate']:.4f}  "
              f"UNSAFE={r['unsafe_plan_count']:>3}  "
              f"unsound={r['unsound_target_count']}")
    return rows, trials_by_p


def run_edge_criticality(n_graphs=120):
    """Mean artifacts dropped from the true blast radius per single lost edge.
    Now meaningful: merge and compose nodes have descendants, so a multi-parent
    edge is no longer capped at one node by construction."""
    cost = defaultdict(list)
    # Why an edge costs nothing: because its parent is not exposed to any
    # sampled incident at all, or because an alternate path still reaches the
    # child. Only the second is the redundancy story.
    zero_out_of_scope = defaultdict(int)
    zero_alternate_path = defaultdict(int)
    for seed in range(1000, 1000 + n_graphs):
        records = generate(seed=seed)
        recs = {rid: r.to_dict() for rid, r in records.items()}
        full = graph_from_records(recs)
        pzs = [n for n, a in full.nodes(data=True) if a.get("is_patient_zero")]
        base = {pz: blast_radius(full, pz)[0] for pz in pzs}
        exposed = set().union(*base.values()) if base else set()
        for rid, r in records.items():
            for par in r.parent_ids:
                perturbed = {k: dict(v) for k, v in recs.items()}
                perturbed[rid]["parent_ids"] = [
                    x for x in recs[rid]["parent_ids"] if x != par]
                g2 = graph_from_records(perturbed)
                lost = sum(len(base[pz] - blast_radius(g2, pz)[0]) for pz in pzs)
                cost[r.operation].append(lost)
                if lost == 0:
                    if par not in exposed:
                        zero_out_of_scope[r.operation] += 1
                    else:
                        zero_alternate_path[r.operation] += 1

    rows = []
    for op in ("fine-tune", "quantize", "merge", "compose"):
        v = cost[op]
        m, h = mean_ci(v)
        nz = [x for x in v if x > 0]
        m_nz, h_nz = mean_ci(nz) if nz else (0.0, 0.0)
        rows.append({
            "edge_type": op, "n_edges_perturbed": len(v),
            "mean_models_lost_per_missing_edge": round(m, 4),
            "ci95_halfwidth": round(h, 4),
            "fraction_costing_nothing": round(sum(1 for x in v if x == 0) / len(v), 4),
            # The unconditional mean mixes two effects. These separate them.
            "conditional_mean_given_nonzero": round(m_nz, 4),
            "conditional_ci95_halfwidth": round(h_nz, 4),
            "conditional_median_given_nonzero": (
                sorted(nz)[len(nz) // 2] if nz else 0),
            "single_parent": op in ("fine-tune", "quantize", "compose"),
            "zero_because_parent_unexposed": zero_out_of_scope[op],
            "zero_despite_exposed_parent": zero_alternate_path[op],
        })
        r = rows[-1]
        print(f"  {op:<11} mean {m:.3f}+/-{h:.3f}   "
              f"{100 * r['fraction_costing_nothing']:.0f}% cost nothing   "
              f"given nonzero: {m_nz:.2f} (median {r['conditional_median_given_nonzero']})   "
              f"zeros: {r['zero_because_parent_unexposed']} unexposed / "
              f"{r['zero_despite_exposed_parent']} alternate-path")
    return rows


def run_scenario_sweep(n_trials=N_TRIALS):
    rows = []
    for name, rates in DROP_SCENARIOS.items():
        trials, actual = [], []
        for i in range(n_trials):
            tg, kg, nt, nk = build_pair_by_op(1000 + i, rates)
            trials.append(trial_metrics(evaluate_graphs(tg, kg)))
            actual.append((nt - nk) / nt)
        m_r, h_r = mean_ci([t["recall"] for t in trials])
        fu = sum(t["fu_count"] for t in trials)
        v = sum(t["verdicts"] for t in trials)
        rows.append({
            "scenario": name, "per_operation_drop_rates": rates,
            "mean_actual_untracked_fraction": round(sum(actual) / len(actual), 4),
            "recall_mean": round(m_r, 4), "recall_ci95_halfwidth": round(h_r, 4),
            "false_unrecoverable_count": fu,
            "false_unrecoverable_rate": round(fu / v, 4) if v else 0.0,
            "total_false_positives": sum(t["false_positives"] for t in trials),
        })
        print(f"  {name:<30} dropped={rows[-1]['mean_actual_untracked_fraction']:.3f}  "
              f"recall={m_r:.3f}+/-{h_r:.3f}  FU={fu}")
    return rows


def run_adversarial(n_trials=N_TRIALS, drop_p=0.15):
    """The threat model has an adversary who waits. Withholding attestations
    near patient zero is strictly cheaper for them than any random loss, and
    it is what the uniform sweep never tests. Calibrated to the same overall
    drop fraction so the comparison is like-for-like."""
    rows = []
    for label, builder in (("benign uniform", build_pair),
                           ("adversarial", build_pair_adversarial)):
        trials, actual = [], []
        for i in range(n_trials):
            tg, kg, nt, nk = builder(1000 + i, drop_p)
            trials.append(trial_metrics(evaluate_graphs(tg, kg)))
            actual.append((nt - nk) / nt)
        m_r, h_r = mean_ci([t["recall"] for t in trials])
        rows.append({
            "missingness": label,
            "mean_actual_untracked_fraction": round(sum(actual) / len(actual), 4),
            "recall_mean": round(m_r, 4), "recall_ci95_halfwidth": round(h_r, 4),
            "total_false_positives": sum(t["false_positives"] for t in trials),
        })
        print(f"  {label:<16} dropped={rows[-1]['mean_actual_untracked_fraction']:.3f}  "
              f"recall={m_r:.3f} +/- {h_r:.3f}")
    return rows


def run_stress(n_trials=200, drop_p=0.30):
    """Raise merge density well above the default and re-measure both planner
    error classes.

    This exists because the symmetric error - the tracked planner calling an
    artifact recoverable when the truth is blocked - is only reachable when a
    merge can have two parents affected by the same patient zero. Denser merge
    graphs make that configuration commoner, so if the error is real it should
    grow here, and if the planner's soundness property is real it should
    survive here.
    """
    import generate_dataset as gd
    original = gd.N_MERGES
    rows = []
    try:
        for n_merges in (6, 12, 20):
            gd.N_MERGES = n_merges
            trials = [trial_metrics(evaluate_graphs(*build_pair(3000 + i, drop_p)[:2]))
                      for i in range(n_trials)]
            rec = sum(t["recoverable"] for t in trials)
            fr = sum(t["fr_count"] for t in trials)
            rows.append({
                "n_merges": n_merges,
                "n_trials": n_trials,
                "drop_p": drop_p,
                "recoverable_verdicts": rec,
                "verdict_disagreement_count": fr,
                "verdict_disagreement_rate": round(fr / max(1, rec), 4),
                "false_recoverable_count_strict": sum(t["fr_count_strict"] for t in trials),
            "unsafe_plan_count": sum(t["unsafe_plans"] for t in trials),
            "unsafe_plan_rate": round(
                sum(t["unsafe_plans"] for t in trials)
                / max(1, sum(t["recoverable"] for t in trials)), 5),
                "unsound_target_count": sum(t["unsound"] for t in trials),
            })
            print(f"  merges={n_merges:>3}  recoverable={rec:>5}  "
                  f"disagree={fr:>4} ({rows[-1]['verdict_disagreement_rate']:.4f})  "
                  f"UNSAFE={rows[-1]['unsafe_plan_count']}  "
                  f"unsound={rows[-1]['unsound_target_count']}")
    finally:
        gd.N_MERGES = original
    return rows


if __name__ == "__main__":
    print(f"DROP-RATE SWEEP ({N_TRIALS} independent lineages per rate)\n")
    drop_rows, trials_by_p = run_drop_sweep()

    print("\nSLOPES vs detection miss rate (bootstrap over trials, "
          "respecting the paired design)\n")
    slopes = bootstrap_slopes(trials_by_p)
    for k, v in slopes.items():
        print(f"  false-unrecoverable {k:<6} slope {v['slope']:+.3f}  "
              f"95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

    print("\nEDGE CRITICALITY (marginal cost of one missing edge)\n")
    criticality_rows = run_edge_criticality()

    print(f"\nNON-UNIFORM MISSINGNESS ({N_TRIALS} trials, all ~15% dropped)\n")
    scenario_rows = run_scenario_sweep()

    print(f"\nADVERSARIAL MISSINGNESS ({N_TRIALS} trials, both ~15% dropped)\n")
    adversarial_rows = run_adversarial()

    print("\nMERGE-DENSITY STRESS (does the symmetric error grow, "
          "and does plan soundness survive?)\n")
    stress_rows = run_stress()

    zero = next(r for r in drop_rows if r["drop_p"] == 0.0)
    checks = {
        "recall_is_exactly_1_at_zero_drop": zero["recall_mean"] == 1.0,
        "no_false_unrecoverable_at_zero_drop": zero["false_unrecoverable_count"] == 0,
        "no_false_positives_anywhere": all(
            r["total_false_positives"] == 0
            for r in drop_rows + scenario_rows + adversarial_rows),
        "no_unsound_rollback_targets_anywhere": all(
            r["unsound_target_count"] == 0 for r in drop_rows + stress_rows),
        "root_pz_never_false_unrecoverable": all(
            r["by_pz_position"]["root"]["fu_count"] == 0 for r in drop_rows),
    }
    print("\nHARNESS ASSERTIONS "
          "(these test the experiment, not the system)\n")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    with open("results/sweep.json", "w") as f:
        json.dump({"drop_rate_sweep": drop_rows,
                   "slopes": slopes,
                   "edge_criticality": criticality_rows,
                   "non_uniform_missingness_sweep": scenario_rows,
                   "adversarial_missingness": adversarial_rows,
                   "merge_density_stress": stress_rows,
                   "harness_assertions": checks}, f, indent=2)
    print("\nWrote results/sweep.json")
