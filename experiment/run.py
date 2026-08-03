"""Run the sweep.  Writes experiment/results/sweep.json.

The harness, and the only module that imports from both sides:

    mechanism/   the DP algorithm.  Sees the real data, spends the budget.
    evaluation/  scores its output against ground truth.  Post-processing
                 only, and something you could not do in a real deployment.

Two arms at every budget -- stock standard DP, and the Blowfish policy in
policy.POLICY -- run on the same seeds so the comparison is paired.  Five zCDP
budget points x two arms x five seeds = 50 runs.  Run-to-run variance in AIM is
high enough that single-run numbers carry little information.

Both arms spend exactly the same rho.  What differs is what that budget buys:
stock releases the marginal, the policy arm releases weights on the policy
graph's edges -- more numbers, each with less noise.

    python experiment/run.py            # from the project root
"""
import json
import time

import aim
import metrics
from data import load

RHOS = [0.000625, 0.0025, 0.01, 0.04, 0.16]
SEEDS = list(range(5))
ARMS = [False, True]                    # stock, then policy
OUT = "experiment/results/sweep.json"


def evaluate(true, est):
    """Every metric for one fitted joint, against the true joint."""
    mean3, max3 = metrics.workload_error(true, est)
    return {
        "wl_mean": mean3, "wl_max": max3,
        "tv1": metrics.tv_distance(true, est, 1),
        "tv2": metrics.tv_distance(true, est, 2),
        "range": {str(a): metrics.range_by_stratum(true, est, a) for a in metrics.ORDINAL},
        "w1": {str(a): metrics.wasserstein1(true, est, a) for a in metrics.ORDINAL},
        "small": {str(a): metrics.small_cell_error(true, est, a) for a in (0, 1)},
    }


def main():
    _, true = load()
    rows = []
    t0 = time.time()

    for rho in RHOS:
        for pol in ARMS:
            for seed in SEEDS:
                est, picked = aim.run(true, rho, seed, use_policy=pol)
                row = evaluate(true, est)
                row.update(rho=rho, seed=seed, policy=pol,
                           distinct=len(set(picked)),
                           picked=[list(S) for S in picked])
                rows.append(row)
                print(f"[{time.time()-t0:6.0f}s] rho={rho:<9} "
                      f"{'policy' if pol else 'stock':<7} seed={seed} "
                      f"wl={row['wl_mean']:.4f}", flush=True)
        json.dump(rows, open(OUT, "w"))

    print(f"done in {time.time()-t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
