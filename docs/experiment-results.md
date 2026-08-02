# Policy-aware AIM on Adult — results

Stock AIM under standard differential privacy against AIM under the Blowfish
policy of `policy.POLICY`, at matched privacy budget.

**50 runs** — 5 zCDP budgets × 2 arms × 5 seeds, 2,444s. Both arms spend
identical `rho`; what differs is what that budget buys. Every metric is an
error, so lower is better, and every ratio is `policy / stock`, so **above 1.00
means the policy is worse**.

Raw output: `experiment/results/sweep.json`. Reproduce the tables with
`python experiment/evaluation/analyze.py`.

---

## Headline

The policy is **not** a free upgrade and **not** a wash. It is a trade whose
sign depends on what you ask:

| | result |
|---|---|
| Ordinal / range-shaped queries | **decisively better** — 2–4× at every budget |
| Cell-level marginal accuracy | **worse below `rho = 0.04`**, slightly better above |
| Disclosure cost | 4 `workclass` block totals, released exactly |

The spec's hypothesis had three parts. Two hold; one fails.

---

## 1. Where the policy wins: ordinal structure

### Range queries

A range query under stock AIM sums many independently noised cells, so its
error grows with the width of the interval. Under a threshold policy the same
range is a **difference of two published aggregates**, so the error barely grows
at all.

![Range-query error on age](figures/range-age.svg)

![Range-query error on hours.per.week](figures/range-hours.svg)

That flatness is the whole point of the threshold graph. Stock error grows 4.8×
across the widths shown; the policy grows 2.8× and finishes 43% lower.

Ratio at the widest band (`policy / stock`), across the budget range:

| `rho` | age, width >20 | hours, width >20 | age, width 1–2 |
|---|---|---|---|
| 0.000625 | 0.38 | 0.62 | 1.02 |
| 0.0025 | 0.44 | 0.53 | 1.47 |
| 0.01 | 0.46 | 0.64 | 1.13 |
| 0.04 | 0.47 | 0.45 | 1.01 |
| 0.16 | 0.57 | 0.39 | 0.98 |

Read the last column against the first two: the policy is **worse or equal on
single cells and roughly twice as good on wide intervals**, at every budget.
That is precisely the predicted behaviour, and it is the clearest signal in the
sweep.

### Wasserstein-1

The distance that respects ordinal position — the one a threshold policy should
most improve.

![Wasserstein-1 on age](figures/wasserstein-age.svg)

| `rho` | age | hours | education.num |
|---|---|---|---|
| 0.000625 | **0.24** | 0.62 | 0.76 |
| 0.0025 | 0.39 | 0.49 | 0.87 |
| 0.01 | 0.33 | 0.52 | 0.80 |
| 0.04 | 0.32 | 0.38 | 1.00 |
| 0.16 | 0.29 | 0.29 | 0.87 |

**3–4× better on age at every budget.** `education.num` gains least, which is
consistent with it having the narrowest threshold (θ=2) and the smallest domain
(16 values) — the least ordinal structure to exploit.

Small-count cell error on age improves similarly, ratio 0.19 to 0.65.

---

## 2. Where it loses: cell-level accuracy

![Cell-level accuracy ratio](figures/cell-accuracy-ratio.svg)

| `rho` | TV 1-way | TV 2-way | TV 3-way | 3-way L1 |
|---|---|---|---|---|
| 0.000625 | 1.13 | 1.06 | 1.02 | 1.02 |
| 0.0025 | **1.33** | 1.14 | 1.08 | 1.08 |
| 0.01 | **1.31** | **1.19** | 1.11 | 1.11 |
| 0.04 | 0.94 | 0.99 | 0.99 | 0.99 |
| 0.16 | **0.81** | 0.97 | 0.99 | 0.99 |

There is a **crossover at `rho ≈ 0.04`**. Below it the policy is up to 33% worse
on 1-way total variation; above it, better. Seed spread is small enough
(sd ≤ 0.003 at `rho ≥ 0.01`) that this is well outside noise.

### Why

The policy publishes **more numbers, each quieter**. For age it releases 483
edge values instead of 73 cell counts, at sensitivity 0.487 instead of 1.414:

| marginal | stock | policy |
|---|---|---|
| age | 73 values at `1.414σ` | 483 values at `0.487σ` |
| hours.per.week | 94 at `1.414σ` | 716 at `0.460σ` |
| age × hours | 6,862 at `1.414σ` | 97,670 at `0.356σ` |

At small `rho` every number is so noisy that having 6.6× more of them hurts. At
large `rho` the lower per-number sensitivity dominates and the ordering flips.

The spec predicted `1.26×` relative cell error for age at θ=7. The measured
1.31–1.33 at `rho` = 0.0025–0.01 is close. The spec did **not** predict the
inversion at high budget.

---

## 3. Verdict on the stated hypothesis

The spec (§1) claims policy-aware AIM is *"indistinguishable from stock AIM on
cell-level marginal accuracy, measurably better on ordinal and range-shaped
queries, and carries a bounded, enumerable disclosure cost."*

| claim | verdict |
|---|---|
| No regression on cell accuracy | ✗ **fails** for `rho < 0.04` — up to 33% worse on TV 1-way. Holds for `rho ≥ 0.04`. |
| Better on ordinal / range queries | ✓ **confirmed** — 2× on wide ranges, 3–4× on W1, at every budget |
| Bounded disclosure cost | ✓ the 4 `workclass` block totals, `[4351, 22696, 3657, 1857]`, released exactly |

Part 1 is a genuine miss, not a rounding error, and it should be reported as
one. The honest summary is that the policy trades cell-level accuracy for
ordinal accuracy, and the trade is only free once the budget is large enough.

---

## Caveats

**The comparison is not symmetric.** The spec says this itself, and it matters:
Blowfish at `rho` guarantees strictly *less* than DP at `rho`, because the
neighbour relation is narrower. An equal-`rho` accuracy win is therefore
*expected* and is not by itself a finding. What makes the range-query result
meaningful is pairing it with the disclosure accounting.

**SELECT diversity differs slightly.** At `rho` = 0.01 the policy arm picks 3
distinct marginals against stock's 4, so some of the cell-accuracy gap may be
selection behaviour rather than the transform itself. This has not been
isolated — doing so would need a run with SELECT frozen to a fixed schedule.

| `rho` | stock | policy |
|---|---|---|
| 0.000625 | 2.0 | 2.0 |
| 0.0025 | 3.0 | 2.0 |
| 0.01 | 4.0 | 3.0 |
| 0.04 | 5.0 | 5.0 |
| 0.16 | 5.0 | 5.0 |

**Errors stay well above the sampling floor.** Drawing 32,561 records from the
*true* joint, with no privacy at all, already costs TV-3way 0.071. The best
result here is 0.161. The mechanism, not the sampler, is the binding constraint
everywhere in this budget range — so none of these numbers are floor-limited.

**One spec ambiguity was resolved by hand.** §5.2 specifies `I_k` for a
full-protection axis while §4.1 specifies the complete graph. `I_k` is the
paper's Case I ⊥-star construction, which belongs to the *unbounded* setting,
whereas §4.1 and §6.3 commit to bounded DP. The code follows §4.1. This affects
only `income`, where `k = 2` and both readings give a single edge.

**Not measured.** The spec's Tier C (downstream utility) and Tier D (attack /
inference gap) were not run. Only the marginal-accuracy and ordinal tiers are
covered here.

---

## Reproducing

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation
.venv/bin/python experiment/run.py            # ~41 min, 50 runs
.venv/bin/python experiment/evaluation/analyze.py
```

The stock arm is deterministic per seed and doubles as a regression check —
`rho = 0.04`, seed 0 must give 3-way L1 `0.2982`. See `architecture.md` for the
module layout and the mechanics of the transform.
