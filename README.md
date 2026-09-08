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

Two evaluations. `evaluate.py` runs a single fully-inspectable case;
`sweep.py` runs the statistical evaluation the claims rest on. 300 trials per
condition, 95% confidence intervals, ~11 seconds end to end.

### Detection

| Untracked edges | Recall (95% CI) | False-unrecoverable count | rate (pooled) | rate (mid-chain) |
|---|---|---|---|---|
| 0% | 1.000 | 0 | 0.000 | 0.000 |
| 5% | 0.904 ± 0.010 | 199 | 0.017 | 0.053 |
| 15% | 0.734 ± 0.015 | 429 | 0.046 | 0.136 |
| 30% | 0.524 ± 0.015 | 588 | 0.095 | 0.257 |
| 50% | 0.344 ± 0.013 | 649 | 0.191 | 0.446 |

![Recall and falsely-unrecoverable verdicts against the fraction of unrecorded lineage edges](figures/degradation.png)

![Marginal cost of one missing edge, by derivation type](figures/criticality.png)

### The findings

**Plan soundness.** Any rollback target the planner proposes lies outside the
*true* blast radius, even though the planner only ever sees the incomplete
graph. Proved, and zero violations across 2,700 trials plus 1,200 stress trials
at triple merge density. So the planner can be wrong about *whether* recovery is
possible, but never about whether a rollback it proposes is safe to run.

**The two error measures disagree.** The falsely-unrecoverable *rate* grows
superlinearly in the detection miss rate (log-log slope 1.26, bootstrap CI
[1.11, 1.46]). The *count* grows **sublinearly** (0.655, CI [0.51, 0.84]),
because the population receiving any verdict collapses from 13,557 to 3,402.
Cost is about the count. The defensible claim is that a larger *share* of the
advice an operator receives is wrong, not that the volume of wrong advice grows
disproportionately.

**The error is structurally confined.** Root patient zeros are half the verdict
denominator and cannot produce this error at all — no clean ancestor exists
above a root, so `unrecoverable` is simply true. Pooling dilutes the rate about
threefold.

**Edge criticality is bimodal.** `merge` (1.367 ± 0.151) and `compose`
(1.407 ± 0.232) edges cost *more* on average to lose than `fine-tune`
(1.147 ± 0.052) or `quantize` (1.068 ± 0.070) — while being far more often
free: 70% and 67% cost nothing at all, against 51% and 53%. Redundant parents
mean losing one edge usually costs nothing; but a merge reached only through the
dropped edge takes an aggregated subtree with it.

> An earlier version of this repo reported the opposite — merge edges as 2.8×
> *cheaper* to lose. That was an artifact of a generator in which merges had no
> descendants, so a merge edge could not cost more than one node. The flaw was
> found in external review and is documented in `generate_dataset.py`. It is
> worth knowing how much a result like this depends on synthetic topology.

**Strategic missingness breaks the benign model.** An adversary who withholds
attestations near patient zero, rather than losing them at random, halves recall
at an identical budget: **0.338 vs 0.734**. Attestation coverage measured in
aggregate says nothing about coverage where an adversary chooses to hide.

**Valid signatures do not mean uncontaminated artifacts.** A signed, unmodified
LoRA adapter is contaminated through the base model it is served against. The
deployed model is base plus adapter; only the adapter was signed.

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
