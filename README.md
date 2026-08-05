# Policy-Aware AIM on Adult

What happens to a state-of-the-art DP synthetic data mechanism when you swap
its privacy definition for a weaker, more specific one?

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
still hiding whether they are 30 or 60. The payoff should be that queries
spanning many ages get cheaper, because the mechanism no longer has to protect
distinctions inside that band.

The mechanism releases weights on the policy graph's **edges** rather than
counts on its **cells** (`x_G = P_G⁻¹x`, from the Design paper). Noise goes
there, where it stays isotropic, and is never reconstructed back.

## What we found

Both arms spend identical budget on identical seeds — 50 paired runs across
five budgets.

**The policy is uniformly worse per cell and conditionally better per range.**
The sign of the effect flips with query width, which is exactly the shape a
threshold policy predicts:

![Range error ratio by interval width](docs/figures/range-ratio.svg)

| | result |
|---|---|
| Cell accuracy (AIM's own metric) | **1.02–1.19× worse**, at every budget — the policy wins 0 of 25 paired runs |
| Narrow ranges (width 1–2) | **1.4–2.0× worse** — 0 of 50 paired comparisons |
| Wide ranges (width >20) | **1.2–1.7× better** on `age` at every budget; up to **3.0×** on `hours.per.week` |

**Read this against the caveat, not around it:** Blowfish at `rho` guarantees
strictly less than DP at `rho`, so an equal-budget accuracy win is *expected*.
What makes the result informative is that the gain has a shape — it reverses
with query width — rather than being a uniform lift.

Full numbers, paired statistics and limitations: **[experiment-results.md](docs/experiment-results.md)**.

## Docs

| | |
|---|---|
| **[experiment-results.md](docs/experiment-results.md)** | What was measured, what it showed, and what these two metrics do and do not license |
| **[policy-aware-aim-spec.md](docs/policy-aware-aim-spec.md)** | The specification: data, policy, the `P_G` transform, sensitivity, protocol, metrics, reference values |
| **[architecture.md](docs/architecture.md)** | Module layout, the privacy boundary, and the mechanics of the transform |
| **[docs/papers/](docs/papers)** | The six source papers |

## Running it

Needs `numpy`, and nothing else.

```bash
export PYTHONPATH=experiment/mechanism:experiment/evaluation

python experiment/run.py                    # the sweep -- 50 runs, ~43 min
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
