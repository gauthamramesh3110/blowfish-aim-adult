"""Run the sweep.  Writes experiment/results/sweep.json.

The harness, and the only module that imports from both sides:

    mechanism/   the DP algorithm.  Sees the real data, spends the budget.
    evaluation/  scores its output against ground truth.  Post-processing
                 only, and something you could not do in a real deployment.

Four arms at every budget -- a 2x2 over the two things the experiment varies:

                     |  stock (release the marginal)  |  policy (release x_G)
    -----------------+--------------------------------+----------------------
    bounded DP       |  bounded-stock                 |  bounded-policy
    unbounded DP     |  unbounded-stock               |  unbounded-policy

The columns isolate the **mechanism**: what the P_G transform does, holding the
privacy definition fixed.  The rows isolate the **relaxation**: what a Blowfish
policy is worth, and that depends entirely on the baseline.  Bounded DP must
protect every substitution in one step, so a threshold policy excuses it from
most of them.  Unbounded DP never protected substitutions cheaply to begin
with, so there is nothing there to relax.

Five zCDP budgets x four arms x five seeds = 100 runs, all on the same seeds so
every comparison is paired.

    python experiment/run.py            # from the project root
"""
import json
import time

import aim
import metrics
from data import load

RHOS = [0.000625, 0.0025, 0.01, 0.04, 0.16]
SEEDS = list(range(5))
ARMS = [(b, p) for b in (True, False) for p in (False, True)]
OUT = "experiment/results/sweep.json"


def label(bounded, policy):
    return f"{'bounded' if bounded else 'unbounded'}-{'policy' if policy else 'stock'}"


def evaluate(true, est):
    """Both metrics for one fitted joint, against the true joint."""
    return {
        "wl_mean": metrics.workload_error(true, est),
        "range": {str(a): metrics.range_by_stratum(true, est, a)
                  for a in metrics.ORDINAL},
    }


def main():
    _, true = load()
    rows = []
    t0 = time.time()

    for rho in RHOS:
        for bounded, pol in ARMS:
            for seed in SEEDS:
                est, picked = aim.run(true, rho, seed,
                                      use_policy=pol, bounded=bounded)
                row = evaluate(true, est)
                row.update(rho=rho, seed=seed, policy=pol, bounded=bounded,
                           arm=label(bounded, pol),
                           distinct=len(set(picked)),
                           picked=[list(S) for S in picked])
                rows.append(row)
                print(f"[{time.time()-t0:6.0f}s] rho={rho:<9} "
                      f"{label(bounded, pol):<18} seed={seed} "
                      f"wl={row['wl_mean']:.4f}", flush=True)
        json.dump(rows, open(OUT, "w"))

    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
