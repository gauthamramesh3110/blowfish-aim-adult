# Policy-aware AIM on Adult — results

Stock AIM under standard differential privacy against AIM under the Blowfish
policy of `policy.POLICY`, at matched privacy budget.

**Unbounded DP**, matching AIM's own neighbour definition — adding or removing
one record.

**50 runs** — 5 zCDP budgets × 2 arms × 5 seeds, 2,563s. Both arms spend
identical `rho`, on the same seeds, so every comparison below is paired.

Raw output: `experiment/results/sweep.json`. Reproduce with
`python experiment/evaluation/analyze.py`.

---

## The two metrics

This report is deliberately built on two metrics, no more.

| metric | definition | source |
|---|---|---|
| **Workload error** | mean over all ten 3-way marginals of `‖x_S − x̂_S‖₁ / n` | **AIM** (McKenna et al., VLDB 2022) — its native metric, same 3-way workload |
| **Range error, by width** | `D = cumsum(x̂_a − x_a)`; error of `[lo, hi)` is `|D[hi] − D[lo]|`; averaged within width bands, over every interval on each ordinal attribute | **Blowfish Design paper** (Haney, Machanavajjhala, Ding, 2015) — range queries under threshold policies are its motivating workload |

Neither is invented for this project. AIM supplies the cell-level metric; it has
no metric that can express what a threshold policy changes, so the range-query
benchmark comes from the Blowfish line the transform itself is taken from.

Every number is an error, so lower is better, and every ratio is
`policy / stock` — **above 1.00 means the policy is worse**.

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

Stock error grows steeply with interval width — a range query under stock AIM
sums many independently noised cells. The policy's curve is flatter, because
the same range is closer to a difference of two published aggregates. The two
cross.

At `rho = 0.16` on age, stock runs 10.0 → 56.2 across the width bands while the
policy runs 14.9 → 38.3.

### The ratio, and where it crosses parity

![Range error ratio by width](figures/range-ratio.svg)

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
seeds. That flip is the finding, and it needs no second metric family to
establish: a mechanism that were simply noisier, or simply better, could not
produce it. The policy is not trading accuracy for accuracy at random — it is
moving error from wide queries to narrow ones, which is what a threshold policy
is supposed to do.

**Cell error tells you the price is real and always paid.** Workload error is
the control. If the policy were winning on ranges by being quietly weaker
overall, cell error would not be uniformly worse — it would be uniformly
better. It is uniformly worse, 25 runs for 25, which is what a genuine
trade looks like rather than a sensitivity bug. A cell-error ratio below 0.9×
would have indicated the latter.

**The smallest-θ attribute behaves differently, in the predicted direction.**
`education.num` has θ=2, so almost every measured interval is wider than its
threshold — and it is the one attribute below parity at every width once
`rho ≥ 0.04`. That is consistent with the crossover being governed by θ.

---

## 4. What they do not license

**The crossover location is not resolvable at 5 seeds.** Interpolating each
paired run's parity crossing gives, at `rho = 0.16`, a median width of 4.8 for
age (θ=7, seed range 4.4–8.7) and 5.1 for hours (θ=8, range 2.1–7.3). The
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
