"""Aggregate the sweep into tables.

Four arms, a 2x2 over the two things the experiment varies:

    bounded-stock     bounded-policy       <- policy CAN relax this baseline
    unbounded-stock   unbounded-policy     <- policy CANNOT relax this one

Read it two ways.  **Across** a row (stock vs policy at a fixed privacy
definition) isolates the mechanism: what releasing x_G instead of the marginal
buys and costs.  **Down** a column (bounded vs unbounded) shows what the
neighbour relation costs, and therefore how much room a policy has to relax it.

Every number is an error, so lower is better, and every ratio is policy/stock
within one privacy definition -- above 1.00 means the transform is worse there.

Reads sweep.json and nothing else -- no import from mechanism/.

    python experiment/evaluation/analyze.py     # from the project root
"""
import json

import numpy as np

RES = json.load(open("experiment/results/sweep.json"))
RHOS = sorted({r["rho"] for r in RES})
BANDS = ["1-2", "3-5", "6-10", "11-20", ">20"]
RELATIONS = [(True, "bounded"), (False, "unbounded")]

# theta per ordinal attribute, mirroring policy.POLICY -- kept as a literal so
# this module stays JSON-only and never imports from mechanism/.
ORD = {"0": ("age", 7), "1": ("hours", 8), "2": ("education.num", 2)}


def rows(rho, bounded, policy):
    return [r for r in RES
            if r["rho"] == rho
            and bool(r.get("bounded", False)) == bounded
            and bool(r.get("policy", False)) == policy]


def stat(rho, bounded, policy, path):
    """Mean over seeds of one metric, for one cell of the 2x2."""
    vals = []
    for r in rows(rho, bounded, policy):
        v = r
        for k in path:
            if k not in v:
                v = None
                break
            v = v[k]
        if v is not None and v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def wins(rho, bounded, path):
    """Paired seeds (out of 5) where the policy beats stock, same relation."""
    n = 0
    for seed in range(5):
        def get(pol):
            v = [r for r in rows(rho, bounded, pol) if r["seed"] == seed]
            if not v:
                return None
            v = v[0]
            for k in path:
                if k not in v:
                    return None
                v = v[k]
            return v
        s, p = get(False), get(True)
        if s is not None and p is not None and p < s:
            n += 1
    return n


def block(title, path, fmt="{:.4f}"):
    """One metric, both relations side by side, with the policy/stock ratio."""
    print(f"\n{title}")
    print(f"  {'':>9}{'BOUNDED':>34}  |{'UNBOUNDED':>34}")
    print(f"  {'rho':>9}{'stock':>10}{'policy':>10}{'ratio':>8}{'win':>6}  |"
          f"{'stock':>10}{'policy':>10}{'ratio':>8}{'win':>6}")
    for rho in RHOS:
        line = f"  {rho:>9}"
        for bounded, _ in RELATIONS:
            s = stat(rho, bounded, False, path)
            p = stat(rho, bounded, True, path)
            w = wins(rho, bounded, path)
            ratio = p / s if s == s and s else float("nan")
            line += (f"{fmt.format(s):>10}{fmt.format(p):>10}"
                     f"{ratio:>8.2f}{w:>4}/5")
            if bounded:
                line += "  |"
        print(line)


print("=" * 96)
print("CELL-LEVEL ACCURACY        ratio = policy / stock within one relation")
print("=" * 96)
block("3-way workload error (AIM's own metric)", ["wl_mean"])

print("\n" + "=" * 96)
print("RANGE QUERIES              where the threshold policy should pay off")
print("=" * 96)
for ax, (name, theta) in ORD.items():
    for b in BANDS:
        if all(stat(rho, bd, False, ["range", ax, b]) != stat(rho, bd, False, ["range", ax, b])
               for rho in RHOS for bd, _ in RELATIONS):
            continue                       # band absent for this attribute
        block(f"{name} (theta={theta}), interval width {b}",
              ["range", ax, b], "{:.1f}")

print("\n" + "=" * 96)
print("SELECT diversity (distinct marginals chosen, of 10) -- a diagnostic")
print("=" * 96)
print(f"  {'rho':>9}{'b-stock':>10}{'b-policy':>10}{'u-stock':>10}{'u-policy':>10}")
for rho in RHOS:
    print(f"  {rho:>9}"
          + "".join(f"{stat(rho, bd, pol, ['distinct']):>10.1f}"
                    for bd, _ in RELATIONS for pol in (False, True)))

print("\nfor scale -- resampling 32,561 records from the TRUE joint, no DP:")
print("  3-way workload error 0.1420")
