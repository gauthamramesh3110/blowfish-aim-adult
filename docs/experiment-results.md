# Policy-aware AIM on Adult — results

Stock AIM under standard differential privacy, against AIM under a Blowfish
policy, at matched privacy budget.

**The policy** puts threshold graphs on the three ordinal attributes — `age`
θ=7, `hours.per.week` θ=8, `education.num` θ=2 — a 4-block partition on
`workclass`, and full protection on `income`. Full definition in
[the spec](policy-aware-aim-spec.md#4-the-policy).

Neighbours differ by adding or removing one record, as in AIM.

**50 runs** — 5 zCDP budgets × 2 arms × 5 seeds, 43 minutes. Both arms spend
identical `rho` on the same seeds, so every comparison below is paired.

Raw output: `experiment/results/sweep.json`. Reproduce with
`python experiment/evaluation/analyze.py`.

---

## What is being measured

Each run produces a **fitted joint** — a histogram over the same
`73 × 94 × 16 × 9 × 2` = 1,976,256 cells as the real data, summing to the same
32,561 records. Every metric compares that against the true joint. Two metrics,
and deliberately only two.

### Metric A — workload error, for cell-level accuracy

Sum the joint down to each of the ten 3-way marginals (`C(5,3) = 10`) and take
the total absolute discrepancy, normalised by `n`:

```
workload error = mean over the 10 marginals of  ‖x_S − x̂_S‖₁ / n
```

**Scale.** Both histograms sum to `n`, so this runs from 0 to 2. Every
misplaced record is counted twice — once where it is missing, once where it is
wrongly added — so **0.30 means roughly 15% of the population sits in the wrong
cell**.

*Source:* **AIM** (McKenna et al., VLDB 2022). This is AIM's own metric on its
own all-3-way workload, which is what makes the column comparable to published
numbers.

### Metric B — range error, for interval queries

A range query asks *"how many people are aged between 30 and 45?"* — a
contiguous run of a 1-way marginal. Every such interval is evaluated
exhaustively, with no sampling: **2,701** on `age`, **4,465** on
`hours.per.week`, **136** on `education.num`. All of them fall out of one
cumulative sum — with `D = cumsum(x̂_a − x_a)` prefixed by zero, the error of
`[lo, hi)` is `|D[hi] − D[lo]|`.

**Units: records.** An error of 56 means that interval's estimated headcount
was off by 56 people. These are *not* normalised.

**Stratified by width, never pooled.** Intervals are grouped by how many
adjacent values they span:

| width band | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|
| intervals on `age` | 145 | 210 | 330 | 585 | 1,431 |

The policy is worse on narrow intervals and better on wide ones. Pooling them
into a single average lets the two cancel and reports nothing, so every ratio
below is read **per band**.

*Source:* **Blowfish Design paper** (Haney, Machanavajjhala, Ding, 2015) —
range queries under threshold policies are its motivating workload. AIM has no
metric that can express what a threshold policy changes, so this one comes from
the Blowfish line the transform itself is taken from. Neither metric is
invented for this project.

### How to read the tables

- Every number is an **error**: lower is better.
- Every ratio is **policy / stock**: **above 1.00 means the policy is worse**.
- **θ** is the policy's threshold — the width of the band of values it declines
  to distinguish. `age` θ=7, `hours.per.week` θ=8, `education.num` θ=2.
- Both arms run the **same 5 seeds** at each of 5 budgets, so every comparison
  is paired. "0 of 25" means 5 budgets × 5 seeds, all lost; "0 of 50" adds a
  second attribute.

---

## Headline

The policy is **uniformly worse per cell** and **conditionally better per
range** — better only above a width threshold, and only once the budget is
large enough for the transform to matter.

| | result |
|---|---|
| Cell accuracy | **1.02–1.19× worse**, at every budget. Policy wins **0 of 25** paired runs. |
| Narrow ranges (width 1–2) | **1.4–2.0× worse** on age and hours. Policy wins **0 of 50** paired comparisons. |
| Wide ranges (width >20) | **1.2–1.7× better** on age at every budget; on hours only above `rho = 0.04`, reaching **3.0×** |
| `education.num` (θ=2) | better at **every** width once `rho ≥ 0.04` |

---

## 1. Cell-level accuracy — the cost

![3-way workload error](figures/workload-error.svg)

*Both arms improve as budget grows, and the orange policy line sits above the
blue stock line at every point. The gap narrows but never closes.*

| `rho` | stock | sd | policy | sd | ratio | policy wins |
|---|---|---|---|---|---|---|
| 0.000625 | 0.5312 | 0.0220 | 0.6237 | 0.0158 | 1.17 | 0/5 |
| 0.0025 | 0.4159 | 0.0118 | 0.4954 | 0.0068 | 1.19 | 0/5 |
| 0.01 | 0.3491 | 0.0067 | 0.3947 | 0.0138 | 1.13 | 0/5 |
| 0.04 | 0.3279 | 0.0020 | 0.3361 | 0.0017 | **1.02** | 0/5 |
| 0.16 | 0.2993 | 0.0084 | 0.3243 | 0.0025 | 1.08 | 0/5 |

Worse everywhere, and not marginally — the policy loses every single paired
run, 25 for 25. With seed-level standard deviations of 0.002–0.022 against gaps
of 0.008–0.093, this is not seed noise.

The gap narrows as budget grows but does not close monotonically: it peaks at
1.19× and is smallest at `rho = 0.04` (1.02×), then widens again to 1.08×. Two
metrics cannot say why. The most likely cause is that the two arms select
different marginals — see the caveats.

The cause of the loss itself is structural. The policy publishes **more
numbers, each quieter**: for age it releases 556 edge values instead of 73 cell
counts, at sensitivity 0.460 instead of 1.0. Good for questions spanning many
cells, bad for questions about one cell.

---

## 2. Range accuracy — the purchase

![Range-query error on age](figures/range-age.svg)

![Range-query error on hours.per.week](figures/range-hours.svg)

*Mean error in records, against interval width, at the largest budget. Stock
(blue) starts lower and climbs steeply; policy (orange) starts higher and
stays flat. **The crossing is the result.***

Stock error grows steeply with interval width, because a range query under
stock AIM sums many independently noised cells and their errors accumulate. The
policy's curve is flatter, because the same range is closer to a difference of
two published aggregates. The two cross.

Mean error in records at `rho = 0.16` on `age`, across the five width bands:

| | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|
| stock | **10.0** | **21.2** | 37.3 | 57.4 | 56.2 |
| policy | 14.9 | 23.5 | **28.9** | **35.7** | **38.3** |

Read along each row: stock's error grows **5.7×** across the bands (10.0 →
57.4), the policy's only **2.6×** (14.9 → 38.3). That difference in slope is
what produces the crossing.

### The ratio, and where it crosses parity

![Range error ratio by width](figures/range-ratio.svg)

*The same data as a ratio. All three lines fall left to right — the wider the
query, the better the policy does. `education.num` (green, θ=2) is already
below parity at the narrowest band, because almost every interval on a 16-value
attribute is wider than its threshold of 2.*

Error ratio at `rho = 0.16`, with paired seed wins for the policy in brackets:

| width band | age (θ=7) | hours (θ=8) | education.num (θ=2) |
|---|---|---|---|
| 1–2 | 1.49 (0/5) | 1.37 (0/5) | **0.67** (4/5) |
| 3–5 | 1.11 (0/5) | 1.01 (2/5) | **0.57** (4/5) |
| 6–10 | **0.78** (4/5) | **0.77** (5/5) | **0.61** (4/5) |
| 11–20 | **0.62** (5/5) | **0.51** (5/5) | **0.71** (4/5) |
| >20 | **0.68** (4/5) | **0.33** (5/5) | — |

Widest-band ratio across the whole sweep:

| `rho` | age >20 | hours >20 | education.num 11–20 | age 1–2 |
|---|---|---|---|---|
| 0.000625 | 0.60 | 1.03 | 1.15 | 1.36 |
| 0.0025 | 0.86 | 0.98 | 0.84 | 1.92 |
| 0.01 | 0.64 | 0.93 | 0.91 | 1.54 |
| 0.04 | 0.81 | 0.51 | 0.79 | 1.70 |
| 0.16 | 0.68 | 0.33 | 0.71 | 1.49 |

Two patterns hold across the sweep:

- **Narrow ranges never win.** On age and hours the 1–2 band is worse at every
  budget, in every seed — 0 of 50 paired comparisons. A θ-band graph spends its
  budget describing windowed aggregates, so the narrowest queries pay for
  structure they do not use.
- **`hours` needs budget; `age` does not.** Age wins at the widest band at all
  five budgets. Hours is a wash below `rho = 0.04` (1.03, 0.98, 0.93) and then
  improves sharply to 0.33. Note `hours.per.week` is spiked at 40, so an
  interval benchmark there partly measures "did the interval contain cell 40" —
  a cell-count question in disguise. Age is the cleaner attribute.

---

## 3. What the two metrics together license

**The sign of the effect flips within a single metric.** Range error is worse
at narrow widths and better at wide widths, in the same runs, on the same
seeds. That flip is the finding, and it needs no second metric to establish.
A mechanism that were uniformly noisier would lose at every width; one that
were uniformly better would win at every width. Neither happened: on `age` at
`rho = 0.16` the ratio runs 1.49, 1.11, 0.78, 0.62, 0.68 — it crosses. The
policy is moving error from wide queries to narrow ones, which is what a
threshold policy is built to do.

**Cell error shows the price is real and always paid.** Workload error is the
control. If the policy were winning on ranges merely by being quietly weaker
overall, cell error would be uniformly *better*, not worse. It is worse in all
25 paired runs, which is what a genuine trade looks like rather than a
sensitivity bug — a cell-error ratio below 0.9× would have indicated the
latter.

**The smallest-θ attribute behaves differently, in the predicted direction.**
`education.num` has θ=2, so almost every measured interval is wider than its
threshold — and it is the one attribute below parity at every width once
`rho ≥ 0.04`. That is consistent with the crossover being governed by θ.

---

## 4. What they do not license

**The crossover location is not resolvable at 5 seeds.** For each seed
separately, take the five band ratios, find the pair of adjacent bands where
the ratio crosses 1.0, and interpolate log-linearly between their mean widths
to get that run's crossover width. At `rho = 0.16` this gives a median of 4.8
for age (θ=7, seed range 4.4–8.7) and 5.1 for hours (θ=8, range 2.1–7.3). The
spreads overlap almost entirely, and across budgets the age median wanders from
4.8 to 15.6. **The crossover exists; its position cannot be pinned to θ with
this data.** The spec's falsifiable prediction — crossover at width ≈ θ — is
therefore neither confirmed nor refuted here. More seeds, and width strata
normalised to θ rather than the current absolute bands, would settle it.

**The cell-accuracy loss is not attributed.** These two metrics cannot separate
the transform's cost from SELECT choosing different marginals in the two arms.
See the caveats.

**Nothing is said about ordinal structure beyond ranges, downstream utility, or
the privacy price.** Those need metrics not reported here.

---

## Caveats

**The comparison is not symmetric.** Blowfish at `rho` guarantees strictly less
than DP at `rho` — the neighbour relation is narrower, so fewer database pairs
must look alike. An equal-`rho` accuracy win is therefore *expected*, and the
range wins should be read against that, not as free utility. This is the single
most important qualifier on section 2.

**SELECT diversity differs.** At `rho = 0.16` stock picks 6.2 distinct
marginals against the policy's 5.0. Some of the cell-accuracy gap may be
selection behaviour rather than the transform. Isolating it needs a run with
SELECT frozen to a fixed schedule, which has not been done.

| `rho` | stock | policy |
|---|---|---|
| 0.000625 | 2.2 | 2.0 |
| 0.0025 | 3.2 | 2.0 |
| 0.01 | 4.4 | 3.4 |
| 0.04 | 5.0 | 5.0 |
| 0.16 | 6.2 | 5.0 |

**Width bands are absolute, not θ-normalised.** `metrics.BANDS` is
1–2 / 3–5 / 6–10 / 11–20 / >20 for all three attributes, but θ is 7, 8 and 2.
The crossover therefore falls in a different band for each attribute, and
`education.num` has no band strictly below its own threshold. Strata of
`[1, θ/2] … >4θ` would fix it, at the cost of a re-run — see spec §11. This is
the main reason section 4's crossover question stays open.

**Errors are measured on the fitted joint, not on sampled records.** Drawing
32,561 records from the true joint, no privacy at all, already costs 3-way L1
0.142 — about half what the mechanism loses at the top of the budget range — so
sampled metrics would spend much of their range measuring the sampler. The
consequence is that these numbers are not directly comparable to published AIM
figures, and that the largest open question — whether the range gain survives
projection onto the log-linear model and out to sampled records — is not
answered here. See spec §11.

**Errors stay above the sampling floor.** The best result here is 0.299 against
that 0.142 floor, so nothing is floor-limited and the mechanism is the binding
constraint throughout.

**Only 1-D ranges.** Rectangle queries over attribute pairs were not run.

**Not measured.** Downstream utility (TSTR) and the attack / inference gap were
not run. Earlier revisions also scored TV distance, Wasserstein-1, max workload
error and small-count cell error; those are no longer computed, because the
width-stratified range metric already carries the finding. They survive only as
extra keys in the committed `sweep.json`.

---

## Reproducing

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation
.venv/bin/python experiment/run.py                    # ~43 min, 50 runs
.venv/bin/python experiment/evaluation/analyze.py     # all metrics, tables
.venv/bin/python experiment/evaluation/figures.py     # the four figures above
```

Regression check: `rho = 0.04`, seed 0, stock arm. `aim.py`'s smoke test must
give age-1way L1 `0.0323` and `(0,2,4)` L1 `0.2904`; `metrics.workload_error`,
which averages all ten 3-way marginals, must give `0.3303`.
See `architecture.md` for the module layout and the mechanics of the transform.
