# Policy-Aware AIM on Adult

What happens to a state-of-the-art DP synthetic data mechanism when you swap
its privacy definition for a more specific one?

This runs [AIM](docs/papers) — an adaptive select–measure–fit mechanism for
differentially private synthetic data — under a **Blowfish policy** instead of
standard differential privacy, and measures what that buys and what it costs on
the Adult census dataset.

## The idea in one paragraph

Standard DP promises that *any* two neighbouring databases look alike. A
Blowfish policy says which pairs actually need to: it is a graph per attribute,
where an edge between two values is a promise that those values stay
indistinguishable. Put a **threshold graph** on `age` — every pair within 7
years joined — and you have declined to hide whether someone is 30 or 34, while
still hiding whether they are 30 or 60. The hypothesis under test is that
queries spanning many ages get cheaper, because the mechanism no longer has to
protect distinctions inside that band.

The mechanism releases weights on the policy graph's **edges** rather than
counts on its **cells** (`x_G = P_G⁻¹x`, from the Design paper). Noise goes
there, where it stays isotropic, and is never reconstructed back.

## What is measured

Two metrics, both taken from the literature rather than invented here:

- **Workload error** — the mean L1 error over all ten 3-way marginals,
  normalised by `n`. AIM's own metric; it measures *cell-level* accuracy.
- **Range error** — the error of every interval query on an ordinal attribute
  ("how many people are aged 30–45?"), in records, grouped by the interval's
  **width**: how many adjacent values it spans. Reported per width band, never
  pooled.

## What is run

A 2×2 over the two things that vary — the privacy definition, and what the
mechanism releases:

| | stock — release the marginal | policy — release `x_G` |
|---|---|---|
| **bounded DP** | `bounded-stock` | `bounded-policy` |
| **unbounded DP** | `unbounded-stock` | `unbounded-policy` |

**100 runs** — 5 zCDP budgets × 4 arms × 5 seeds, all arms on the same seeds so
every comparison is paired.

![Range error ratio, policy / stock, within each privacy definition](docs/figures/range-ratio-2x2.svg)

*Range error ratio, policy ÷ stock, within each privacy definition, at
`rho = 0.16`. Below the dashed line the policy arm has the lower error.*

Figures, tables, the full budget × width grid, what follows from them, and the
next steps are all in **[experiment-results.md](docs/experiment-results.md)**.

## Docs

| | |
|---|---|
| **[experiment-results.md](docs/experiment-results.md)** | Readings, figures, the full grid, conclusions and next steps |
| **[policy-aware-aim-spec.md](docs/policy-aware-aim-spec.md)** | The specification: data, policy, the `P_G` transform, sensitivity, protocol, metrics, reference values |
| **[architecture.md](docs/architecture.md)** | Module layout, the privacy boundary, and the mechanics of the transform |
| **[docs/papers/](docs/papers)** | The six source papers |

## Running it

Needs `numpy`, and nothing else.

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation

python experiment/run.py                    # the sweep -- 100 runs, ~79 min
python experiment/evaluation/analyze.py     # tables
python experiment/evaluation/figures.py     # figures -> docs/figures/
```

`run.py` writes `experiment/results/sweep.json` after each budget block, so
partial results survive an interrupt. The committed `sweep.json` is what the
results doc reports, so `analyze.py` and `figures.py` work without re-running.

Each module has a self-test:

```bash
python experiment/mechanism/data.py      # dataset invariants
python experiment/mechanism/policy.py    # graph sizes, sensitivity, round-trip
python experiment/mechanism/aim.py       # end-to-end smoke test
```

## Layout

```
dataset/adult.csv          32,561 records
experiment/
  mechanism/               the DP algorithm -- sees real data, spends budget
    data.py                load Adult into a dense joint histogram
    policy.py              Blowfish policy graphs: build, transform, sensitivity
    mle.py                 fit a joint to noisy measurements
    aim.py                 the SELECT / MEASURE / refit loop
  evaluation/              scoring -- post-processing, sees ground truth
    metrics.py             workload error, range error by width
    analyze.py             sweep.json -> tables
    figures.py             sweep.json -> SVGs
  run.py                   the harness, the only module importing both sides
```

The `mechanism/` / `evaluation/` split is the **privacy boundary**, not just
tidiness: `mechanism/` is what would ship, `evaluation/` reads ground truth to
score the output — something you could not do in a real deployment. Nothing in
`mechanism/` imports `evaluation/`, and that is checkable.
