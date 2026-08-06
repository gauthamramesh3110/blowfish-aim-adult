# Policy-aware AIM on Adult — results

Measurements and what follows from them. Each section states its readings
first, then what they support; Section 7 collects the conclusions and Section 8
the next steps.

---

## 1. What was run

A 2×2 over the two things the experiment varies — the **privacy definition**
and what the mechanism **releases**:

| | stock — release the marginal | policy — release `x_G` |
|---|---|---|
| **bounded DP** | `bounded-stock` | `bounded-policy` |
| **unbounded DP** | `unbounded-stock` | `unbounded-policy` |

- **bounded DP** — `n` is fixed and public; a neighbour move changes one
  record's value. A raw marginal has L2 sensitivity **√2**.
- **unbounded DP** — a neighbour move adds or removes one record; `n` is not
  public and is measured. A raw marginal has L2 sensitivity **1**.
- **stock** releases the marginal itself.
- **policy** releases `x_G = P_G⁻¹x`, weights on the edges of the Blowfish
  policy graph. The policy is threshold θ=7 on `age`, θ=8 on `hours.per.week`,
  θ=2 on `education.num`, a 4-block partition on `workclass`, full protection
  on `income`. Full definition in [the spec](policy-aware-aim-spec.md#4-the-policy).

**100 runs** — 5 zCDP budgets × 4 arms × 5 seeds, 4,716s. All four arms run the
same five seeds at each budget, so every comparison is paired.

Raw output: `experiment/results/sweep.json`. Reproduce with
`python experiment/evaluation/analyze.py`.

---

## 2. What is being measured

Each run produces a **fitted joint** — a histogram over the same
`73 × 94 × 16 × 9 × 2` = 1,976,256 cells as the real data, summing to the same
32,561 records. Both metrics compare that against the true joint.

### Metric A — workload error, for cell-level accuracy

Sum the joint down to each of the ten 3-way marginals (`C(5,3) = 10`) and take
the total absolute discrepancy, normalised by `n`:

```
workload error = mean over the 10 marginals of  ‖x_S − x̂_S‖₁ / n
```

Both histograms sum to `n`, so this runs from 0 to 2. Every misplaced record is
counted twice — once where it is missing, once where it is wrongly added — so
0.30 corresponds to roughly 15% of the population sitting in the wrong cell.

*Source:* **AIM** (McKenna et al., VLDB 2022) — its native metric, on its own
all-3-way workload.

### Metric B — range error, for interval queries

A range query asks *"how many people are aged between 30 and 45?"* — a
contiguous run of a 1-way marginal. Every such interval is evaluated
exhaustively, with no sampling: **2,701** on `age`, **4,465** on
`hours.per.week`, **136** on `education.num`. All fall out of one cumulative
sum — with `D = cumsum(x̂_a − x_a)` prefixed by zero, the error of `[lo, hi)` is
`|D[hi] − D[lo]|`.

**Units: records.** An error of 56 means that interval's estimated headcount was
off by 56 people. These are *not* normalised.

Intervals are grouped by width — how many adjacent values they span — and
reported per band, never pooled:

| width band | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|
| intervals on `age` | 145 | 210 | 330 | 585 | 1,431 |

*Source:* **Blowfish Design paper** (Haney, Machanavajjhala, Ding, 2015) —
range queries under threshold policies are its motivating workload.

### Reading the tables

- Every number is an **error**: lower is better.
- Every ratio is **policy / stock within the same privacy definition**, so it
  isolates the effect of the release. Above 1.00 means the policy arm is worse.
- **(n/5)** is the paired seed count where the policy arm beat stock.
- **θ** is the policy threshold: `age` 7, `hours.per.week` 8,
  `education.num` 2.

**Why the 2×2 and not a single comparison.** Two things differ between stock
AIM and policy-aware AIM: the *release* (marginal counts, or edge weights) and
the *privacy definition*. Changing both at once cannot tell them apart. Reading
**across a row** — stock vs policy at a fixed definition — isolates the
release. Comparing that same ratio **between rows** shows how much the
definition changes what the release is worth.

---

## 3. Metric A — workload error

![3-way workload error, all four arms](figures/workload-error.svg)

*Colour is the privacy definition (blue bounded, orange unbounded); dashed is
stock, solid is policy. Mean over 5 seeds.*

Mean (sd) over 5 seeds, and the policy/stock ratio within each definition:

| `rho` | bounded stock | bounded policy | ratio | unbounded stock | unbounded policy | ratio |
|---|---|---|---|---|---|---|
| 0.000625 | 0.5839 (0.0083) | 0.5814 (0.0213) | 1.00 (3/5) | 0.5312 (0.0220) | 0.6237 (0.0158) | 1.17 (0/5) |
| 0.0025 | 0.4457 (0.0059) | 0.4744 (0.0098) | 1.06 (0/5) | 0.4159 (0.0118) | 0.4954 (0.0068) | 1.19 (0/5) |
| 0.01 | 0.3627 (0.0016) | 0.3722 (0.0175) | 1.03 (2/5) | 0.3491 (0.0067) | 0.3947 (0.0138) | 1.13 (0/5) |
| 0.04 | 0.3328 (0.0017) | 0.3334 (0.0019) | 1.00 (1/5) | 0.3279 (0.0020) | 0.3361 (0.0017) | 1.02 (0/5) |
| 0.16 | 0.3244 (0.0005) | 0.3260 (0.0019) | 1.00 (1/5) | 0.2993 (0.0084) | 0.3243 (0.0025) | 1.08 (0/5) |

For scale: resampling 32,561 records from the **true** joint, with no privacy
at all, costs 0.1420 on this metric.

**What the table shows.** The transform's cell-level cost depends almost
entirely on the privacy definition. Averaged over the five budgets the
policy/stock ratio is **1.02 under bounded DP** and **1.12 under unbounded DP**;
under bounded it is 1.00 at three of five budgets and never exceeds 1.06, while
under unbounded the policy arm loses every one of the 25 paired runs.

The mechanism-level explanation is in Section 5: the transform publishes many
more numbers than there are cells, and reading one cell means combining many of
them. What changes between the definitions is the *baseline* those extra
numbers are measured against — bounded DP's stock arm already carries √2
sensitivity, unbounded DP's carries 1.

---

## 4. Metric B — range error

### 4.1 Absolute error against width, `rho = 0.16`

![Range error on age, by interval width](figures/range-age.svg)

![Range error on hours.per.week, by interval width](figures/range-hours.svg)

*Mean error in records at the largest budget. Same colour and dash scheme as
above.*

`age`, `rho = 0.16`, mean records of error:

| arm | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|
| bounded stock | 12.3 | 23.4 | 36.4 | 50.2 | 58.5 |
| bounded policy | 13.3 | 23.0 | 31.1 | 30.3 | 39.6 |
| unbounded stock | 10.0 | 21.2 | 37.3 | 57.4 | 56.2 |
| unbounded policy | 14.9 | 23.5 | 28.9 | 35.7 | 38.3 |

`hours.per.week`, `rho = 0.16`:

| arm | 1–2 | 3–5 | 6–10 | 11–20 | >20 |
|---|---|---|---|---|---|
| bounded stock | 12.4 | 22.8 | 37.4 | 61.9 | 92.2 |
| bounded policy | 20.8 | 26.9 | 32.3 | 35.7 | 30.9 |
| unbounded stock | 11.1 | 20.8 | 34.6 | 57.4 | 83.0 |
| unbounded policy | 15.2 | 21.0 | 26.5 | 29.2 | 27.7 |

**What the tables show.** Both policy arms trade the narrow bands for the wide
ones. On `age` at this budget stock error grows about 4.8× across the bands
(12.3 → 58.5 bounded) while the policy arm grows about 3.0× (13.3 → 39.6); on
`hours.per.week` stock grows 7.4× against the policy arm's 1.5×. The crossing
point sits between the 3–5 and 6–10 bands in three of the four arms.

### 4.2 Ratio against width, by privacy definition

![Range error ratio, policy / stock, within each privacy definition](figures/range-ratio-2x2.svg)

*Left panel bounded, right panel unbounded. Below the dashed line the policy arm
has lower error than stock under the same definition. `rho = 0.16`.*

**What the panels show.** All six lines fall left to right, so the direction of
the effect is the same under both definitions — it is a property of what is
released, not of the guarantee. The panels differ in where the lines sit, not in
their shape.

### 4.3 Every budget × every width band

![Range error ratio grid](figures/range-ratio-grid.svg)

*One cell per attribute × privacy definition × budget × width band — 140 in
all. Blue where the policy arm has lower error than stock under the same
definition, orange where it is higher, deeper colour further from parity. The
number is the ratio. Blank cells: `education.num` has 16 values, so no interval
is wider than 20.*

**What the grid shows.** Three patterns, none of which depends on picking a
budget:

1. **Every panel gets bluer left to right.** The transform's advantage grows
   with interval width, at all five budgets, on all three attributes, under both
   privacy definitions. Counting cells below parity: `age` 19/25 bounded and
   10/25 unbounded, `hours.per.week` 14/25 and 7/25, `education.num` 16/20 and
   12/20.
2. **The bounded panels are bluer than the unbounded ones.** Comparing the same
   attribute-budget-band cell across the two definitions, **the bounded ratio is
   better in 60 of 70 cells.**
3. **`education.num` flips wholesale at higher budgets.** With θ=2 on a 16-value
   domain almost every interval is wider than the threshold, and from
   `rho = 0.04` upward the whole panel is blue — reaching 0.23 under bounded DP.

Exact values including paired seed counts: `python experiment/evaluation/analyze.py`.

---

## 5. Mechanism-level reference values

Computed from the policy alone, before any data is touched. `Δ₂` is the L2
sensitivity of the released quantity; the SELECT penalty is
`sum_i sqrt(deg_i)`; cell error is `Δ₂ × penalty`, the expected L1
reconstruction error at unit `σ`.

| attribute | k | | **bounded** edges | Δ₂ | penalty | cell err | | **unbounded** edges | Δ₂ | penalty | cell err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `age` | 73 | | 484 | 0.4872 | 265.0 | 129.1 | | 556 | 0.4604 | 274.7 | 126.5 |
| `hours.per.week` | 94 | | 717 | 0.4604 | 366.2 | 168.6 | | 810 | 0.4376 | 377.9 | 165.4 |
| `education.num` | 16 | | 30 | 0.7862 | 30.6 | 24.1 | | 45 | 0.6802 | 34.3 | 23.3 |
| `workclass` | 9 | | 11 | 1.0000 | 12.5 | 12.5 | | 16 | 1.0000 | 14.2 | 14.2 |
| `income` | 2 | | 2 | 1.0000 | 2.4 | 2.4 | | 3 | 0.8165 | 2.8 | 2.3 |

Corresponding stock values: `Δ₂ = √2` and penalty = `k` under bounded,
`Δ₂ = 1` and penalty = `k` under unbounded.

**What the table shows.** These are computed from the policy alone, so they
predict the cell-level cost before any run. Dividing policy cell error by stock
cell error gives the expected ratio: on `age`, **1.25 under bounded** against
**1.73 under unbounded**; on `education.num`, 1.06 against 1.46. The measured
ratios in Section 3 are smaller than both — the fit pools measurements across
overlapping marginals and recovers some of the loss — but the ordering between
the two definitions is the same, and it comes entirely from the stock baseline:
√2 × k versus 1 × k.

The row that isolates it: `age` releases **484 edge weights bounded and 556
unbounded to encode 73 cells**, at `Δ₂` of 0.4872 and 0.4604. The transform's
own cost barely moves between definitions (129.1 vs 126.5). What moves is what
it is being compared against.

---

## 6. SELECT diagnostic

Distinct marginals chosen out of 10 rounds — recorded because the two arms may
not explore the same candidates.

| `rho` | bounded stock | bounded policy | unbounded stock | unbounded policy |
|---|---|---|---|---|
| 0.000625 | 2.0 | 2.0 | 2.2 | 2.0 |
| 0.0025 | 3.0 | 2.0 | 3.2 | 2.0 |
| 0.01 | 4.0 | 3.8 | 4.4 | 3.4 |
| 0.04 | 5.0 | 5.0 | 5.0 | 5.0 |
| 0.16 | 5.0 | 5.0 | 6.2 | 5.0 |

**What the table shows.** The policy arms explore slightly fewer distinct
marginals than their stock counterparts at most budgets — most visibly 6.2
against 5.0 under unbounded DP at `rho = 0.16`. The generalised SELECT penalty
is larger for transformed marginals, which steers budget away from them. Some
part of the cell-level gap in Section 3 may therefore be selection behaviour
rather than the transform itself; these two metrics cannot separate the two.

---

## 7. Conclusions

### 7.1 The transform trades cell accuracy for range accuracy

This holds under **both** privacy definitions, so it is a property of what is
released, not of the guarantee.

It charges on cells because it publishes far more numbers than there are cells —
556 edge weights for 73 `age` cells — and reading one cell means combining many
of them, so their noise accumulates. It pays on ranges because each of those
numbers is individually quieter, and a wide interval reads enough of them for
that to dominate.

**The sign flip is the substantive result.** A mechanism that were uniformly
noisier would lose at every width; one uniformly better would win at every
width. Neither happens in any of the 140 cells of the grid. Error is being
*moved* from wide queries to narrow ones, which is what a threshold policy is
built to do.

### 7.2 The relaxation is worth something, and only bounded DP can spend it

This is what the 2×2 was added to test, and it is the clearest result here.

- **Cell cost**: 1.02× under bounded DP, 1.12× under unbounded (Section 3).
- **Range benefit**: the bounded ratio beats the unbounded ratio in **60 of 70**
  attribute × budget × band cells (Section 4.3).

The reason is what each baseline was already paying for. A Blowfish policy
relaxes obligations about **substitutions** — "you needn't hide age 30 from age
60":

| baseline | substitutions | room to relax |
|---|---|---|
| **bounded DP** | every pair `u→v` is one neighbour step, sensitivity √2 | **large** — drop the distant pairs |
| **unbounded DP** | none directly; a substitution is remove-then-add, two steps | **almost none** — already cheap |

Under bounded DP the threshold policy cuts a raw marginal's sensitivity from √2
to 0.4872 on `age`. Under unbounded DP the baseline is already 1 and the policy
has nothing to give back.

**A caveat that should be checked before it is relied on.** `policy.Graph`
attaches a bottom edge to *every* cell in the unbounded construction, so that
policy's neighbour relation appears to *contain* unbounded DP's — every
add/remove is a single step, plus nearby substitutions. If so the unbounded
policy arm is at least as private as unbounded stock rather than less, and the
usual "an equal-`rho` win is expected because Blowfish guarantees less" caveat
does not apply to it. This is an argument about privacy definitions, not a
measurement, and it should be checked against the Blowfish paper's formal
statement.

### 7.3 The gain is smaller than the textbook picture, for a structural reason

The standard account is that a range query becomes a *difference of two running
totals*, so the interior cancels and error stops growing with width.

**This construction does not do that.** A pure path graph with one grounded
endpoint does produce running totals — verified on `education.num`, where the
edge weights come out as the suffix sums. But when every cell carries its own
edge to `⊥`, each cell has a direct shortcut to ground and nothing accumulates
along the axis. A width-`w` query then reads about `w+1` edges rather than 2,
because every bottom edge inside the range contributes.

So the observed gain is **not** "reads fewer numbers". It is "reads numbers that
are individually quieter" — a weaker effect, and the reason the measured ratios
stay far from flat-in-width.

### 7.4 What these results do not support

- **No claim about the crossover location.** Width bands are absolute
  (1–2 … >20) while θ is 7, 8 and 2, so the crossover falls in a different band
  per attribute and `education.num` has no band below its own threshold. That a
  threshold *exists* is supported; where it sits is not.
- **No quantitative model of the range gain.** The per-number-noise argument
  gives the direction, not the size. On `hours.per.week` the widest band reaches
  0.33–0.34, below its own `Δ₂` of ~0.44, which per-number noise alone cannot
  produce.
- **No attribution of the cell-level cost.** SELECT chooses somewhat different
  marginals in the two arms (Section 6), so part of the gap may be selection.
- **No cross-definition utility comparison.** Bounded and unbounded arms carry
  different guarantees; only within-definition ratios are meaningful.
- **Nothing about sampled output, downstream utility, or attack resistance.**

### 7.5 Scope of these measurements

Facts about what was and was not measured.

- Utility is scored on the **fitted joint**, not on records sampled from it.
- Only **1-D** range queries were evaluated; no rectangle queries over
  attribute pairs.
- Width bands are **absolute** (1–2 / 3–5 / 6–10 / 11–20 / >20), identical for
  all three ordinal attributes, while θ is 7, 8 and 2.
- **5 seeds** per arm per budget.
- The workload (`metrics.THREE_WAY`) is used only for scoring. It does not
  reach the mechanism: SELECT weights every candidate equally.
- The policy transform is applied to **every** measured marginal when
  `use_policy` is set, including `workclass`, `income` and
  `workclass × income`, none of which is ordinal.
- Downstream utility (TSTR) and attack / inference-gap measurements were not
  run.

---

## 8. Next steps

In rough order of value per unit of effort.

1. **Settle the guarantee question in §7.2.** It decides whether the unbounded
   arm's range gains are "expected" or free, and it is a short formal check
   against the Blowfish paper rather than a new run.
2. **Stop transforming non-ordinal marginals.** `workclass`, `income` and
   `workclass × income` are transformed today at a predicted 1.52× cell cost
   with no range query ever asked of them. `mle.fit` already mixes transformed
   and plain measurements, so this is a change in `aim.run`, not new theory.
3. **θ-normalised width bands and more seeds.** Strata of `[1, θ/2] … >4θ`
   would make the crossover *location* testable at all; 5 seeds cannot resolve
   it today.
4. **Search for a better strategy than `P_G⁻¹`.** The transformational
   equivalence permits *any* standard-DP mechanism on `(W_G, x_G)`; `P_G⁻¹` is
   the laziest choice, releasing 556 numbers to encode 73. A hierarchical
   strategy was checked and is **not** the answer — worse on cells and only
   marginally better at extreme widths — so this is genuine strategy selection.
5. **Close the two validity gaps**: score on sampled records rather than the
   fitted joint, and freeze SELECT to a fixed schedule to remove the selection
   confound in Section 6.

---

## 9. Reproducing

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation
.venv/bin/python experiment/run.py                    # ~79 min, 100 runs
.venv/bin/python experiment/evaluation/analyze.py     # tables
.venv/bin/python experiment/evaluation/figures.py     # the four figures above
```

Regression check: `rho = 0.04`, seed 0, `unbounded-stock`. `aim.py`'s smoke test
gives age-1way L1 `0.0323` and `(0,2,4)` L1 `0.2904`; `metrics.workload_error`
gives `0.3303`.

See `architecture.md` for the module layout and `policy-aware-aim-spec.md` for
the mechanism.
