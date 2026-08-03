"""How close is the fitted joint to the real one?

Two metrics, and deliberately only two.

  workload error   mean L1 over the 3-way workload, normalised by n.  AIM's
                   own metric (McKenna et al., VLDB 2022), same 3-way workload.
  range error      L1 error of every interval on an ordinal attribute,
                   bucketed by interval width.  Range queries under threshold
                   policies are the motivating workload of the Blowfish Design
                   paper (Haney, Machanavajjhala, Ding, 2015).

The first is what the policy is expected to cost, the second what it buys.
Neither is invented here: AIM supplies the cell-level metric and has none that
can express what a threshold policy changes, so the range benchmark comes from
the Blowfish line the transform itself is taken from.

The range metric is stratified by width and never pooled.  The policy is worse
on narrow intervals and better on wide ones, so a single pooled average lets
the two cancel and reports nothing.  That sign flip, within one metric, is the
whole finding -- which is why no third metric is needed to establish it.

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


def workload_error(true, est):
    """Mean L1 error over all 3-way marginals, normalised by n."""
    e = [np.abs(marginal(true, S) - marginal(est, S)).sum() / true.sum()
         for S in THREE_WAY]
    return float(np.mean(e))


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
