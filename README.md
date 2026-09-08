# Blast Radius and Recovery for Compromised AI Model Lineage

**License:** Apache-2.0 · **Status:** research prototype · **Reproduces in ~13s**

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

`evaluate.py` runs a single fully-inspectable case; `sweep.py` runs the
statistical evaluation the claims rest on — 300 trials per condition, 95%
confidence intervals, ~13 seconds end to end.

![Recall and falsely-unrecoverable verdicts against the fraction of unrecorded lineage edges](figures/degradation.png)

![Marginal cost of one missing edge, by derivation type](figures/criticality.png)

| Untracked | Recall (95% CI) | False-unrec. count | rate (pooled) | rate (mid-chain) | False-recov. rate |
|---|---|---|---|---|---|
| 0% | 1.000 | 0 | 0.000 | 0.000 | 0.000 |
| 5% | 0.904 ± 0.010 | 199 | 0.017 | 0.053 | 0.014 |
| 15% | 0.734 ± 0.015 | 437 | 0.047 | 0.139 | 0.031 |
| 30% | 0.524 ± 0.015 | 589 | 0.096 | 0.258 | 0.050 |
| 50% | 0.344 ± 0.013 | 656 | 0.193 | 0.451 | 0.076 |

### The findings

**Both planner errors are real, and they trade off against each other.** The
planner can wrongly say a model cannot be rolled back (*falsely unrecoverable* —
costly, sends you to retrain unnecessarily) or wrongly say it can (*falsely
recoverable* — dangerous, the rebuild reintroduces the compromise). The second
runs at 3.1% of recoverable verdicts at 15% untracked edges, rising to 7.6% at
50%, and grows with merge density (4.7% → 8.3% as merges per graph go 6 → 20).

**There is a conservative setting that makes the dangerous error provably
zero.** `recovery_plan(..., strict=True)` also treats the rebuild path's own
parent as blocking. Under it the falsely-recoverable rate is 0 by construction
— a `recoverable` verdict is then provably correct — at the cost of precision:
3.8% of its `blocked` verdicts have every off-path parent clean and could
safely have proceeded. Precision and provability trade against each other here;
neither setting dominates.

**Rollback targets are always safe, under either setting.** Whatever the
planner proposes as a rollback target lies outside the *true* blast radius,
even though the planner only ever sees the incomplete graph. Proved, and zero
violations in every configuration tested.

**The two measures of the falsely-unrecoverable error disagree.** Its *rate*
grows superlinearly in the detection miss rate (fitted log-log exponent 1.27,
bootstrap CI [1.12, 1.46]); its *count* grows **sublinearly** (0.66,
CI [0.52, 0.84]), because the population receiving any verdict collapses from
13,557 to 3,402. Cost is about the count.

**The error is structurally confined.** Root patient zeros supply about
two-thirds of the verdict denominator (70% at p=0, 57% at p=0.50) and cannot
produce this error at all — no clean ancestor exists above a root, so
`unrecoverable` is simply true. Pooling dilutes the rate roughly threefold.

**Edge cost is zero-inflated, not bimodal.** Every edge type has one mode at
zero and a decreasing tail. Separating the two effects: multi-parent and
composition edges are about *half as likely to cost anything* (70% and 67% cost
nothing, against 51% and 53%) and about *twice as costly when they do* (mean
given nonzero: merge 4.55, compose 4.28, against fine-tune 2.34, quantize
2.25). Redundant parents mean losing one edge usually costs nothing; but a
merge reached only through the dropped edge takes an aggregated subtree with it.

> An earlier version of this repo reported merge edges as 2.8× *cheaper* to
> lose. That was an artifact of a generator in which merges had no descendants,
> so a merge edge could not cost more than one node. The flaw was found in
> external review and is documented in `generate_dataset.py`.

**Strategic missingness breaks the benign model.** An adversary who spends the
recording budget withholding attestations within two hops of patient zero,
rather than losing edges at random, halves recall at an identical budget:
**0.338 vs 0.734**. This assumes an adversary who controls those derivations
(capability 3), not merely one who published a poisoned artifact. It is also
not the optimal strategy, so it is a lower bound on adversarial damage.

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
| `sweep.py` | Statistical evaluation: drop-rate curve, edge criticality, non-uniform and adversarial missingness, merge-density stress |
| `make_figures.py` | The two figures, from `results/sweep.json` |

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

Graph structure comes from one generative model: branching factor, operation
mix, and the rate at which merges join branches that share an ancestor (0.35 as
a parameter, about 21% realized, since generation-1 merges can only be
cross-family). Nothing here establishes behaviour on a real registry's
topology, and that is the main threat to these results — model hubs publish
enough structure that real topology with synthetic missingness is the
experiment that should replace this one.

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
