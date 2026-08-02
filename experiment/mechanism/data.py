"""Load Adult into a dense joint histogram over the five experiment attributes.

  age             73 values  (17..90)   -> rank among observed values
  hours.per.week  94 values  (1..99)    -> rank among observed values
  education.num   16 values  (1..16)    -> index edu - 1
  workclass        9 values             -> index by WORKCLASS below
  income           2 values             -> 0 = <=50K, 1 = >50K

Ordinal attributes are indexed by rank among *observed* values, not by raw
value.  age is missing 89 and hours is missing {69,71,79,83,93}, all single
rare points in the tail.  Ranking keeps the domains gap-free at k = 73 / 94;
the cost is that one index step is one observed value rather than one year,
which differs only next to those gaps.

Joint = 73*94*16*9*2 = 1,976,256 cells -- 16 MB dense.  Small enough that
inference can run to convergence, so nothing is confounded by model
approximation error.
"""
import csv
import numpy as np

CSV = "dataset/adult.csv"

ATTRS = ["age", "hours.per.week", "education.num", "workclass", "income"]
SIZES = [73, 94, 16, 9, 2]

# Ordered so the four partition blocks are contiguous: [0:3] [3:4] [4:6] [6:9]
WORKCLASS = ["Federal-gov", "State-gov", "Local-gov",
             "Private",
             "Self-emp-not-inc", "Self-emp-inc",
             "Without-pay", "Never-worked", "?"]

INCOME = ["<=50K", ">50K"]


def load():
    """Return (records, joint) -- records is (n, 5) int indices, joint is the histogram."""
    rows = list(csv.DictReader(open(CSV)))
    ages = sorted({int(r["age"]) for r in rows})
    hours = sorted({int(r["hours.per.week"]) for r in rows})
    recs = np.empty((len(rows), 5), dtype=np.int64)
    for i, r in enumerate(rows):
        recs[i] = (ages.index(int(r["age"])),
                   hours.index(int(r["hours.per.week"])),
                   int(r["education.num"]) - 1,
                   WORKCLASS.index(r["workclass"]),
                   INCOME.index(r["income"]))
    joint = np.zeros(SIZES)
    np.add.at(joint, tuple(recs.T), 1.0)
    return recs, joint


def marginal(joint, S):
    """Sum the joint down to the axes in S."""
    return joint.sum(axis=tuple(i for i in range(len(SIZES)) if i not in S))


def expand(arr, S):
    """Reshape a marginal so it broadcasts back against the full joint."""
    shape = [1] * len(SIZES)
    for a, i in enumerate(S):
        shape[i] = SIZES[i]
    return arr.reshape(shape)


def sample(joint, n, rng):
    """Draw n synthetic records from a joint distribution."""
    flat = np.maximum(joint.ravel(), 0)
    flat = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, p=flat)
    return np.stack(np.unravel_index(idx, tuple(SIZES)), axis=1)


if __name__ == "__main__":
    recs, joint = load()
    print("records", recs.shape, "joint", joint.shape, "total", joint.sum())
    assert joint.sum() == 32561
    for a, s in zip(ATTRS, SIZES):
        assert recs[:, ATTRS.index(a)].max() < s
    print("income split", joint.sum(axis=(0, 1, 2, 3)))
    print("workclass   ", joint.sum(axis=(0, 1, 2, 4)))
