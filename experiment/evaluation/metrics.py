"""How close is the synthetic distribution to the real one?

Everything here compares the fitted joint against the true joint directly.

  marginals   L1 error on the 3-way workload, total variation at degree 1 and 2
  ordinal     range queries, Wasserstein-1, small-count cells

Nothing samples records.  Drawing 32,561 records from the *true* joint already
costs 3-way L1 0.142, roughly half what the DP mechanism loses at the top of
the budget range -- so metrics computed on sampled records would spend much of
their range measuring the sampler.  data.sample stays available for producing
synthetic records, it just is not what we score.
"""
import itertools

import numpy as np

from data import SIZES, marginal

ORDINAL = [0, 1, 2]                 # age, hours.per.week, education.num
THREE_WAY = list(itertools.combinations(range(len(SIZES)), 3))

# interval widths, in domain steps, for bucketing range-query error
BANDS = [("1-2", 1, 2), ("3-5", 3, 5), ("6-10", 6, 10),
         ("11-20", 11, 20), (">20", 21, 10 ** 9)]


# ------------------------------------------------------------- marginals
def workload_error(true, est):
    """Mean and max L1 error over all 3-way marginals, normalised by n."""
    e = [np.abs(marginal(true, S) - marginal(est, S)).sum() / true.sum()
         for S in THREE_WAY]
    return float(np.mean(e)), float(np.max(e))


def tv_distance(true, est, degree):
    """Mean total variation distance over all marginals of the given degree."""
    n = true.sum()
    d = [0.5 * np.abs(marginal(true, S) / n - marginal(est, S) / n).sum()
         for S in itertools.combinations(range(len(SIZES)), degree)]
    return float(np.mean(d))


# --------------------------------------------------------------- ordinal
def range_errors(true, est, axis):
    """L1 error of every interval on `axis`, with each interval's width.

    Error of [lo, hi) is |D[hi] - D[lo]| where D is the cumulative difference,
    so all k(k+1)/2 intervals come out of one cumsum.
    """
    d = np.concatenate([[0.0], np.cumsum(marginal(est, (axis,))
                                         - marginal(true, (axis,)))])
    lo, hi = np.triu_indices(len(d), k=1)
    return np.abs(d[hi] - d[lo]), hi - lo


def range_by_stratum(true, est, axis):
    """Mean interval error, bucketed by interval width."""
    err, w = range_errors(true, est, axis)
    out = {}
    for name, lo, hi in BANDS:
        m = (w >= lo) & (w <= hi)
        if m.any():
            out[name] = float(err[m].mean())
    return out


def wasserstein1(true, est, axis):
    """Earth-mover distance between the two 1-way distributions, in cells.

    Unlike TV distance this respects ordinal distance: moving mass from age 20
    to age 21 costs far less than moving it to age 60.
    """
    t = marginal(true, (axis,)); e = marginal(est, (axis,))
    return float(np.abs(np.cumsum(e / e.sum() - t / t.sum())).sum())


def small_cell_error(true, est, axis):
    """Mean relative error on cells holding under 0.1% of mass.

    Rare values are where DP noise does the most damage in relative terms, and
    where a mechanism is most likely to invent mass that was never there.
    """
    t = marginal(true, (axis,)); e = marginal(est, (axis,))
    m = t < 0.001 * true.sum()
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(e[m] - t[m]) / np.maximum(t[m], 1.0)))
