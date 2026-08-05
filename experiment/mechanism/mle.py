"""Inference: weighted least squares over the measurements.

Every measurement is a marginal of the joint, so a set of measurements is one
over-determined linear system in one unknown -- the joint.  We minimise

    sum_r  || (M_r(model) - y_r) / scale_r ||^2

by exponentiated gradient descent.  Two things make this work:

  * The gradient of the r-th term is  A_r^T (A_r x - y_r)  where A_r sums the
    joint down to marginal r.  A_r^T is just a broadcast back over the summed-
    out axes, which is what `expand` does -- so marginals of different shapes
    all contribute to the same cell-shaped gradient and are fitted jointly.
  * Starting from uniform, the multiplicative update keeps log(model) inside
    the span of the measured marginals, i.e. the log-linear family.  That is
    Private-PGM's model class, reached without building a junction tree --
    affordable here only because the joint is small enough to hold densely.

Measurements are weighted by 1/scale^2, so the warm start and the later rounds
combine correctly despite carrying different noise levels.
"""
import numpy as np

from data import SIZES, expand, marginal


def fit(model, meas, scales, n, iters, t0=0, lr=0.5, graphs=None):
    """Exponentiated gradient descent.  `t0` continues a decaying step size.

    Under a Blowfish policy the measurement of S lives in edge space rather
    than cell space, so `graphs[S]` supplies the transform.  The residual is
    taken where the noise is -- on the edges, where it is isotropic -- and
    mapped back to cells by (P_G^-1)^T.  A missing entry means a stock
    cell-space measurement, so both kinds of measurement mix in one fit.

    `n` is the *measured* record count, not the true one.
    """
    graphs = graphs or {}
    for t in range(iters):
        grad = np.zeros(SIZES)
        for S, y in meas.items():
            G, pred = graphs.get(S), marginal(model, S)
            res = pred - y if G is None else G.back(G.transform(pred) - y)
            grad += expand(res / scales[S] ** 2, S)
        g = np.abs(grad).max()
        if g > 0:
            model *= np.exp(-(lr / (1 + (t0 + t) / 100)) * grad / g)
        model *= n / model.sum()
    return model
