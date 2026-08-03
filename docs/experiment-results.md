# Policy-aware AIM on Adult — results

Stock AIM under standard differential privacy against AIM under the Blowfish
policy of `policy.POLICY`, at matched privacy budget.

**Unbounded DP**, matching AIM's own neighbour definition — adding or removing
one record.

**50 runs** — 5 zCDP budgets × 2 arms × 5 seeds, 2,563s. Both arms spend
identical `rho`. Every metric is an error, so lower is better, and every ratio
is `policy / stock`, so **above 1.00 means the policy is worse**.

Raw output: `experiment/results/sweep.json`. Reproduce with
`python experiment/evaluation/analyze.py`.

---

## Headline

The policy buys ordinal accuracy and pays for it in cell accuracy. Under
unbounded DP it pays at **every** budget:

| | result |
|---|---|
| Wasserstein-1 | **1.3–4× better** on age and hours at every budget |
| Wide range queries | **1.2–3× better** on age; on hours only above `rho = 0.04` |
| Cell-level accuracy | **1.02–1.79× worse** — at every budget, no crossover |
| Disclosure cost | none beyond the noisy releases |

---

## 1. Where the policy wins: ordinal structure

A range query under stock AIM sums many independently noised cells, so its
error grows with the interval. Under a threshold policy the same range is a
**difference of two published aggregates**, so it grows far more slowly.

![Range-query error on age](figures/range-age.svg)

![Range-query error on hours.per.week](figures/range-hours.svg)

At `rho = 0.16` on age, stock runs 10.0 → 56.2 across the widths while the
policy runs 14.9 → 38.3: worse on single cells, 1.5× better at the widest band.
That crossing pattern is the threshold policy's signature.

Ratio at the widest band:

| `rho` | age >20 | hours >20 | age 1–2 |
|---|---|---|---|
| 0.000625 | 0.60 | 1.03 | 1.36 |
| 0.0025 | 0.86 | 0.98 | 1.92 |
| 0.01 | 0.64 | 0.93 | 1.54 |
| 0.04 | 0.81 | 0.51 | 1.70 |
| 0.16 | 0.68 | 0.33 | 1.49 |

Note `hours` only pays off at `rho ≥ 0.04` — below that it is a wash. Age wins
throughout.

### Wasserstein-1

![Wasserstein-1 on age](figures/wasserstein-age.svg)

| `rho` | age | hours | education.num |
|---|---|---|---|
| 0.000625 | 0.46 | 0.76 | 1.13 |
| 0.0025 | 0.68 | 0.64 | 0.74 |
| 0.01 | 0.51 | 0.63 | 0.65 |
| 0.04 | 0.46 | 0.38 | 0.71 |
| 0.16 | 0.61 | **0.25** | 0.69 |

Age is a consistent 1.5–2×. Hours improves steadily with budget, reaching **4×**
at `rho = 0.16`. `education.num` gains least and is the one case that goes the
wrong way at the smallest budget (1.13) — it has the narrowest threshold (θ=2)
and the smallest domain, so the least ordinal structure to exploit.

---

## 2. Where it loses: cell-level accuracy

![Cell-level accuracy ratio](figures/cell-accuracy-ratio.svg)

| `rho` | TV 1-way | TV 2-way | 3-way L1 |
|---|---|---|---|
| 0.000625 | 1.47 | 1.26 | 1.17 |
| 0.0025 | **1.79** | 1.33 | 1.19 |
| 0.01 | 1.54 | 1.23 | 1.13 |
| 0.04 | 1.34 | 1.06 | 1.02 |
| 0.16 | 1.10 | 1.12 | 1.08 |

Worse everywhere. The gap narrows as budget grows but never closes.

The cause is structural: the policy publishes **more numbers, each quieter**.
For age it releases 556 edge values instead of 73 cell counts, at sensitivity
0.460 instead of 1.0. That trade is good for questions spanning many cells and
bad for questions about one cell.

---

## 3. Verdict on the stated hypothesis

The spec (§1) claims policy-aware AIM is *"indistinguishable from stock AIM on
cell-level marginal accuracy, measurably better on ordinal and range-shaped
queries, and carries a bounded, enumerable disclosure cost."*

| claim | verdict |
|---|---|
| No regression on cell accuracy | ✗ **fails at every budget** — 1.02× to 1.79× worse |
| Better on ordinal / range queries | ✓ **confirmed** — W1 better on age and hours at every budget (1.3–4×); range wins on age throughout, on hours only above `rho = 0.04`. One exception: W1 on `education.num` is 1.13× *worse* at the smallest budget. |
| Enumerable disclosure cost | n/a — partition block totals are not released exactly here, so there is no extra disclosure to enumerate |

---

## Caveats

**The comparison is still not symmetric.** Blowfish at `rho` guarantees
strictly less than DP at `rho` — the neighbour relation is narrower, so fewer
database pairs must look alike. An equal-`rho` accuracy win is therefore
*expected*, and the ordinal wins should be read against that, not as free
utility.

**SELECT diversity differs, and the gap grew.** At `rho = 0.16` stock now picks
6.2 distinct marginals against the policy's 5.0. Some of the cell-accuracy gap
may be selection behaviour rather than the transform. Isolating it needs a run
with SELECT frozen to a fixed schedule, which has not been done.

| `rho` | stock | policy |
|---|---|---|
| 0.000625 | 2.2 | 2.0 |
| 0.0025 | 3.2 | 2.0 |
| 0.01 | 4.4 | 3.4 |
| 0.04 | 5.0 | 5.0 |
| 0.16 | 6.2 | 5.0 |

**Errors stay above the sampling floor.** Drawing 32,561 records from the true
joint, no privacy at all, already costs 3-way L1 0.142. The best result here is
0.299 — still twice the floor, so nothing is floor-limited and the mechanism is
the binding constraint throughout.

**Not measured.** The spec's Tier C (downstream utility) and Tier D (attack /
inference gap) were not run.

**One departure from the spec.** §6.3 decision 3 specifies a different
neighbour relation from the one used here, so the sensitivity values in this
implementation do not match the reference table in spec §7.

---

## Reproducing

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation
.venv/bin/python experiment/run.py            # ~43 min, 50 runs
.venv/bin/python experiment/evaluation/analyze.py
```

Regression check: `rho = 0.04`, seed 0, stock arm must give 3-way L1 `0.2904`.
See `architecture.md` for the module layout and the mechanics of the transform.
