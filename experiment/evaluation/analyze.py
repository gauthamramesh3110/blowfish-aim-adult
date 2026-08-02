"""Aggregate the sweep into tables.

Every number is an error, so lower is better and every column should fall as
rho rises.  Two arms are reported side by side -- stock standard DP and the
Blowfish policy -- with the ratio policy/stock, so above 1.00 means the policy
arm is worse on that metric.

The policy is expected to lose slightly on cell-level metrics and win on range
queries: it releases more numbers, each carrying less noise.  Read the ratio
per metric; there is no single overall verdict.

The sampling floor at the bottom is what you would get by drawing 32,561
records from the true joint -- a reminder that these errors are measured on the
fitted joint, not on sampled records.

    python experiment/evaluation/analyze.py     # from the project root
"""
import json

import numpy as np

RES = json.load(open("experiment/results/sweep.json"))
RHOS = sorted({r["rho"] for r in RES})
ARMS = [(False, "stock"), (True, "policy")]
ORD = {"0": "age", "1": "hours", "2": "education.num"}
BANDS = ["1-2", "3-5", "6-10", "11-20", ">20"]


def stat(rho, pol, path):
    """Mean and sd over seeds of one metric, at one budget, for one arm."""
    vals = []
    for r in RES:
        if r["rho"] != rho or bool(r.get("policy", False)) != pol:
            continue
        try:
            v = r
            for k in path:
                v = v[k]
        except (KeyError, TypeError):
            # metrics.range_by_stratum omits empty bands -- education.num has
            # only 16 values, so no interval is wider than 20.
            continue
        if v == v:                      # skip NaN
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def table(title, path, fmt="{:.4f}"):
    print(f"\n{title}")
    print(f"  {'rho':>9}{'stock':>11}{'sd':>9}{'policy':>11}{'sd':>9}{'ratio':>9}")
    for rho in RHOS:
        s, ssd = stat(rho, False, path)
        p, psd = stat(rho, True, path)
        ratio = p / s if s else float("nan")
        print(f"  {rho:>9}{fmt.format(s):>11}{fmt.format(ssd):>9}"
              f"{fmt.format(p):>11}{fmt.format(psd):>9}{ratio:>9.2f}")


print("=" * 68)
print("MARGINAL ACCURACY         ratio = policy / stock, >1 means policy worse")
print("=" * 68)
table("3-way workload error (AIM's own target)", ["wl_mean"])
table("max 3-way workload error", ["wl_max"])
table("TV distance, 1-way", ["tv1"])
table("TV distance, 2-way", ["tv2"])
table("TV distance, 3-way", ["tv3"])

print("\n" + "=" * 68)
print("ORDINAL STRUCTURE         where the threshold policy should pay off")
print("=" * 68)
for ax, name in ORD.items():
    print(f"\nRange query error on {name}, by interval width")
    print(f"  {'rho':>9}  {'arm':<8}" + "".join(f"{b:>11}" for b in BANDS))
    for rho in RHOS:
        for pol, label in ARMS:
            row = f"  {rho:>9}  {label:<8}"
            for b in BANDS:
                m, _ = stat(rho, pol, ["range", ax, b])
                row += f"{m:>11.1f}" if m == m else f"{'--':>11}"
            print(row)

for ax, name in ORD.items():
    table(f"Wasserstein-1, {name}", ["w1", ax])
for ax, name in [("0", "age"), ("1", "hours")]:
    table(f"Small-count cell error, {name}", ["small", ax], "{:.2f}")

print("\n" + "=" * 68)
print("SELECT diversity (distinct marginals chosen, of 10)")
print("=" * 68)
print(f"  {'rho':>9}{'stock':>11}{'policy':>11}")
for rho in RHOS:
    s, _ = stat(rho, False, ["distinct"])
    p, _ = stat(rho, True, ["distinct"])
    print(f"  {rho:>9}{s:>11.1f}{p:>11.1f}")

print("\nfor scale -- resampling 32,561 records from the TRUE joint, no DP:")
print("  3-way workload 0.1420   TV2 0.0286   TV3 0.0710   W1 age 0.0648")
