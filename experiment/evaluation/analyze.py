"""Aggregate the sweep into tables.

Two metrics, matching docs/experiment-results.md: AIM's 3-way workload error,
and range-query error stratified by interval width.  Every number is an error,
so lower is better and every column should fall as rho rises.  Two arms are
reported side by side -- stock standard DP and the Blowfish policy -- with the
ratio policy/stock, so above 1.00 means the policy arm is worse.

The policy is expected to lose on workload error and to win on range queries,
but only above a width threshold.  Read the range table across a row: the
ratio starts above 1.00 at narrow widths and drops below it at wide ones.  That
sign flip is the finding, so the range strata are never pooled.

Reads sweep.json and nothing else -- no import from mechanism/, so the tables
can be regenerated without the domain or the policy on the path.

    python experiment/evaluation/analyze.py     # from the project root
"""
import json

import numpy as np

RES = json.load(open("experiment/results/sweep.json"))
RHOS = sorted({r["rho"] for r in RES})
ARMS = [(False, "stock"), (True, "policy")]
BANDS = ["1-2", "3-5", "6-10", "11-20", ">20"]

# theta per ordinal attribute, mirroring policy.POLICY -- kept as a literal so
# this module stays JSON-only and never imports from mechanism/.
ORD = {"0": ("age", 7), "1": ("hours", 8), "2": ("education.num", 2)}


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
print("CELL-LEVEL ACCURACY       ratio = policy / stock, >1 means policy worse")
print("=" * 68)
table("3-way workload error (AIM's own metric)", ["wl_mean"])

print("\n" + "=" * 68)
print("RANGE QUERIES             where the threshold policy should pay off")
print("=" * 68)
for ax, (name, theta) in ORD.items():
    print(f"\nRange query error on {name} (theta = {theta}), by interval width")
    print(f"  {'rho':>9}  {'arm':<8}" + "".join(f"{b:>11}" for b in BANDS))
    for rho in RHOS:
        means = {}
        for pol, label in ARMS:
            row = f"  {rho:>9}  {label:<8}"
            means[pol] = []
            for b in BANDS:
                m, _ = stat(rho, pol, ["range", ax, b])
                means[pol].append(m)
                row += f"{m:>11.1f}" if m == m else f"{'--':>11}"
            print(row)
        row = f"  {'':>9}  {'ratio':<8}"
        for s, p in zip(means[False], means[True]):
            row += f"{p / s:>11.2f}" if s == s and p == p and s else f"{'--':>11}"
        print(row)

print("\n" + "=" * 68)
print("SELECT diversity (distinct marginals chosen, of 10)")
print("=" * 68)
print("a diagnostic, not a metric: the arms may not be picking the same")
print("marginals, which would confound the workload-error gap above.")
print(f"\n  {'rho':>9}{'stock':>11}{'policy':>11}")
for rho in RHOS:
    s, _ = stat(rho, False, ["distinct"])
    p, _ = stat(rho, True, ["distinct"])
    print(f"  {rho:>9}{s:>11.1f}{p:>11.1f}")

print("\nfor scale -- resampling 32,561 records from the TRUE joint, no DP:")
print("  3-way workload error 0.1420")
