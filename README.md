# Blast Radius and Recovery for Compromised AI Model Lineage

**License:** Apache-2.0 · **Status:** research prototype · **Reproduces in ~7s**

Reference implementation for the paper *Beyond Detection: A Recovery-Oriented
Architecture for Compromised AI Model Remediation* (preprint in preparation).

Existing AI supply chain tooling verifies a model artifact at the moment you
adopt it. Signing answers *"is this artifact what it claims to be."* It cannot
answer *"what else is affected, and what do I do about it,"* because
contamination is a relationship between artifacts and a signature is a property
of one artifact in isolation.

This prototype addresses what happens afterwards: when a model or dataset is
discovered to be compromised months later, which downstream models are affected,
and what can actually be done about each one.

> **Motivating case.** In December 2023 the Stanford Internet Observatory found
> validated CSAM in LAION-5B and recommended that models trained on it be
> withdrawn. That recommendation was largely unactionable: no queryable graph
> connected the dataset to its derivatives. The query was never the hard part.
> The graph did not exist.
## What it does

1. Builds a lineage DAG of model artifacts with typed edges (`fine-tune`, `quantize`, `merge`, `compose`)
2. Computes **blast radius**: everything downstream of a compromised artifact
3. Computes a **recovery plan** per affected model: the nearest clean signed ancestor, the operation chain to rebuild from it, and what that costs
4. Evaluates both against ground truth the queries never see

## The experimental design

Two graphs are generated from the same underlying lineage:

- `data/true_lineage.json` — every relationship that actually happened. Ground truth, used only by the evaluator.
- `data/tracked_lineage.json` — what a real system would have on record. Each true edge independently has a 15% chance of never being recorded, modeling the fact that not every fine-tune submits a signing attestation or manifest.

The queries only ever see the tracked graph. This is what makes the evaluation meaningful: it measures what an operator would actually get, not what a complete-information system could theoretically achieve.

Patient zeros are placed at two structurally different positions: at lineage **roots** (poisoned foundation model or dataset) and **mid-chain** (compromise introduced during one team's fine-tune). These turn out to demand completely different incident responses.

## Results

Two evaluations. `evaluate.py` runs a single fully-inspectable case at seed 42;
`sweep.py` runs the statistical evaluation the claims actually rest on.

### The worked case (`evaluate.py`, seed 42, 46 artifacts, 17.8% of edges untracked)

| Metric | Value |
|---|---|
| True affected artifacts | 23 across 5 patient zeros |
| Blast radius recall | 69.6% |
| False positives | 0 |
| Artifacts falsely reported unrecoverable | 3 |
| Signed LoRA adapters contaminated via compromised base | 2 |

This is one draw. It is kept because every model in it can be traced by hand,
which is what makes the failure modes legible. It is not evidence on its own,
and the paper does not treat it as such.

### The statistical evaluation (`sweep.py`)

Four experiments. Every trial regenerates the lineage from scratch and redraws
which edges go unrecorded, so each trial is an independent observation.
300 trials per condition, means with 95% confidence intervals. Runs in ~7s.

**1. Recall degrades along a measurable curve, and false positives are always zero.**

| Untracked edges | Blast radius recall (95% CI) | False positives | False-unrecoverable rate (95% CI) |
|---|---|---|---|
| 0% | 1.000 | 0 | 0.000 |
| 5% | 0.905 ± 0.012 | 0 | 0.023 ± 0.008 |
| 10% | 0.821 ± 0.015 | 0 | 0.046 ± 0.012 |
| 15% | 0.747 ± 0.016 | 0 | 0.078 ± 0.016 |
| 20% | 0.674 ± 0.017 | 0 | 0.102 ± 0.018 |
| 30% | 0.563 ± 0.016 | 0 | 0.152 ± 0.020 |
| 50% | 0.395 ± 0.014 | 0 | 0.257 ± 0.027 |

Zero false positives across all 2,700 trials at every rate. That is the
Soundness proposition holding empirically, not a favourable seed. At 0%
untracked, recall is exactly 1.000 and no artifact is falsely reported
unrecoverable, which validates the harness itself.

**2. Recovery degrades faster than detection does.** Regressing the
false-unrecoverable rate on the detection miss rate in log-log space gives a
slope of **1.30**: the more damaging error grows *superlinearly* in the less
damaging one. Partial lineage capture is not merely proportionally worse, it
is disproportionately worse in the direction that costs money.

**3. Not all lineage edges are worth the same.** Marginal cost of one missing
edge, measured by deleting exactly one edge and recounting the blast radius:

| Edge type | Mean models lost per missing edge (95% CI) | Cut point? |
|---|---|---|
| `fine-tune` | 1.446 ± 0.077 | yes |
| `quantize` | 1.460 ± 0.123 | yes |
| `compose` | 0.600 ± 0.046 | no (leaf) |
| `merge` | 0.524 ± 0.029 | no (redundant parent) |

Single-parent derivations are cut points: losing one severs everything below
it. Merge edges cost **2.8x less**, because compromise propagates as a logical
OR across parents, so the surviving parent still reaches the node. The same
parental redundancy that makes merges hard for per-artifact verification makes
their lineage edges the cheapest ones to lose. `fine-tune` and `quantize` are
statistically indistinguishable in *detection* criticality; they differ only in
*recovery* cost, which is exactly why the edge type is carried on the graph.

This yields an instrumentation priority an operator can act on: if you can only
attest part of your pipeline, attest the single-parent derivations first.

**4. The results survive non-uniform missingness.** The uniform per-edge drop
model is this evaluation's weakest assumption, so it is tested rather than
merely disclaimed. Four scenarios, each calibrated against the measured edge mix
to drop the same ~15% of edges overall, so any difference is attributable to the
*structure* of the missingness and not its magnitude:

| Scenario | Actual untracked | Recall (95% CI) | False-unrecoverable rate |
|---|---|---|---|
| Uniform | 15.2% | 0.747 ± 0.016 | 0.078 ± 0.016 |
| Routine ops under-reported | 15.4% | 0.736 ± 0.016 | 0.070 ± 0.014 |
| Deliberate ops under-reported | 15.7% | 0.781 ± 0.014 | 0.052 ± 0.013 |
| Composition blind spot | 15.6% | 0.781 ± 0.014 | 0.043 ± 0.011 |

Recall moves only within 0.736-0.781 across radically different missingness
structures, and zero false positives hold throughout. The headline number is
driven by the marginal recording rate, not by which derivations go unrecorded.
The spread that does exist is fully explained by experiment 3: scenarios that
spare `fine-tune` edges do better, because those are the cut points.

**5. Query cost tracks blast radius, not registry size.**

| Families | Mean nodes | Blast radius query |
|---|---|---|
| 7 | 65 | 3.11 us |
| 25 | 227 | 3.13 us |
| 100 | 896 | 3.19 us |
| 400 | 3,555 | 3.36 us |

Flat across a 55x increase in registry size. This is not a clever
implementation, it is the complexity: forward reachability is O(affected), and
a compromise in one lineage does not reach further because the registry got
bigger. The honest claim is that registry growth alone does not slow the query.

### The qualitative findings

**Valid signatures do not mean uncontaminated artifacts.** Two LoRA adapters,
both correctly signed and never modified, landed inside blast radii because the
base models they are served against were compromised. The deployed model is base
plus adapter, and only the adapter was signed. Per-artifact verification passes
and the artifact is still unsafe.

**Compromise position determines whether recovery is even possible.** Root
patient zeros left 0 affected artifacts recoverable by rollback, since no clean
ancestor exists above them. Mid-chain compromises leave a clean signed ancestor
upstream, so rollback is genuinely available. No existing tool distinguishes
these cases.

**Missing lineage edges fail twice, and the second failure is worse.** Artifacts
get reported unrecoverable when a valid rollback target existed, because the
edge upward to the clean ancestor was never recorded. A detection miss leaves
you unaware. A false unrecoverable verdict actively pushes an operator toward an
expensive retrain when a cheap deterministic rollback was available. Experiment
2 quantifies how much faster this error grows.

**Naive nearest-clean-ancestor search is wrong on merge nodes.** It returns a
clean parent from an unrelated lineage while the merge's other parent is still
compromised. Merge-descended models are classified
`blocked_on_compromised_merge_parent` and sequenced behind their parents'
recovery instead.

## Running it

```
python3 -m venv .venv
./.venv/bin/pip install networkx
./.venv/bin/python generate_dataset.py
./.venv/bin/python evaluate.py   # the worked case  -> results/evaluation.json
./.venv/bin/python sweep.py      # the statistics   -> results/sweep.json
```

## Files

| File | Purpose |
|---|---|
| `models.py` | `ModelRecord` schema, operation semantics, rebuild cost model |
| `generate_dataset.py` | Synthetic lineage generator, true vs tracked graphs |
| `lineage_graph.py` | DAG construction and blast radius query |
| `recovery.py` | Recovery planner and ancestor search |
| `evaluate.py` | Single-case evaluation harness for detection and recovery |
| `sweep.py` | Statistical evaluation: drop-rate curve, edge criticality, non-uniform missingness, scale |

## Limitations

Synthetic lineage throughout. No real model registry exposes the ground-truth
graph needed to compute recall, which is why the true-versus-tracked construction
is synthetic by necessity rather than by convenience.

The independence assumption is partially addressed rather than fully resolved:
`sweep.py` shows results hold under four different non-uniform missingness
structures, but all four still drop edges independently. Correlated gaps, where
one team's entire pipeline goes unattested at once, would concentrate loss on
connected subgraphs and are not modelled. Calibrating any of these against real
adoption data remains the main open item.

Graph structure comes from one generative model (branching factor, operation mix,
cross-family merge rate). The scale sweep varies size but not shape, so nothing
here establishes behaviour on a real registry's topology.

Federated learning is handled only at round granularity; per-client attribution
is out of scope.

## Citing

A preprint is in preparation. Until it is posted, please cite this repository:

```
Bhargava, A. Blast Radius and Recovery for Compromised AI Model Lineage.
https://github.com/apoorve1577/model-lineage-recovery
```

## License

Apache License 2.0. See `LICENSE`.
