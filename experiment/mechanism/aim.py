"""Minimal AIM: the select-measure-generate loop, under standard DP.

  warm start   measure every 1-way marginal, fit
  each round   SELECT a 2-way marginal by exponential mechanism
               MEASURE it under Gaussian noise
               refit the model on everything measured so far
  finally      a longer fit, then sample synthetic records

Budget is tracked in zCDP throughout: the Gaussian mechanism with noise scale
sigma costs rho = 1/(2 sigma^2), and the exponential mechanism with parameter
eps costs rho = eps^2/8.

Simplifications against the published algorithm: the joint is small enough for
exact inference, so no graphical model is built; the round count is fixed with
no budget annealing; candidates are all 2-way marginals and the workload is all
3-way marginals.
"""
import numpy as np

import mle
import policy
from data import SIZES, marginal

NDIM = len(SIZES)
CANDIDATES = [(i, j) for i in range(NDIM) for j in range(i + 1, NDIM)]
ONE_WAY = [(i,) for i in range(NDIM)]

SENSITIVITY = np.sqrt(2.0)      # bounded DP: one record moves, x changes by e_u - e_v


def cells(S):
    return int(np.prod([SIZES[i] for i in S]))


def cost(S, G):
    """(noise multiplier, expected L1 reconstruction error at unit sigma).

    Stock AIM releases the marginal itself: sensitivity sqrt(2), and one noisy
    cell per cell.  Under a policy it releases x_G instead, so the multiplier
    is the policy sensitivity and the error term is spec 6.2's generalised
    penalty.  Stock is the special case P_G = I.
    """
    return (SENSITIVITY, cells(S)) if G is None else (G.delta, G.penalty)


def measure(x, sigma, rng, G=None):
    """Noisy measurement of a marginal.

    Every record contributes a count of one to exactly one cell, so the whole
    marginal costs the same as a single cell would.  That is why AIM measures
    entire marginals, and why the shape of S does not affect the noise level.

    Under a policy the released quantity is x_G = P_G^-1 x -- weights on the
    policy graph's edges, not counts on its cells -- and the noise goes on
    there, where it stays isotropic.  It is never reconstructed back to cells.
    """
    if G is None:
        return x + rng.normal(0, sigma * SENSITIVITY, size=x.shape)
    xg = G.transform(x)
    return xg + rng.normal(0, sigma * G.delta, size=xg.shape)


def select(true, model, sigma, eps, rng, graphs=None):
    """Exponential mechanism over the candidate marginals.

    AIM's quality score, equation (1):

        q_r = w_r ( ||M_r(D) - M_r(model)||_1  -  sqrt(2/pi) sigma n_r )

    The first term is how badly the current model explains marginal r.  The
    second is what measuring it would cost anyway: sqrt(2/pi)*sigma_eff is the
    expected absolute error of one Gaussian draw, and n_r is the cell count, so
    it discounts marginals that are large enough for the noise to swamp the
    gain.  All workload weights w_r are 1 here.
    """
    q = []
    for S in CANDIDATES:
        delta, pen = cost(S, graphs.get(S) if graphs else None)
        q.append(np.abs(marginal(true, S) - marginal(model, S)).sum()
                 - np.sqrt(2.0 / np.pi) * sigma * delta * pen)
    q = np.array(q)
    # q has sensitivity 2 (a record moving changes one L1 term by at most 2)
    p = np.exp(eps * (q - q.max()) / (2 * 2.0))
    return CANDIDATES[rng.choice(len(CANDIDATES), p=p / p.sum())]


def run(true, rho, seed, rounds=10, iters=30, use_policy=False):
    """One AIM run.  Returns the fitted joint, and the marginals it selected."""
    rng = np.random.default_rng(seed)
    n = true.sum()

    # Neither of these costs budget: the graphs depend only on the policy, and
    # the block totals are invariant under the neighbour relation.  Cached, so
    # a sweep in one process pays the build cost once.
    graphs = ({S: policy.graph(S) for S in ONE_WAY + CANDIDATES}
              if use_policy else None)
    totals = policy.free_totals(true) if use_policy else None

    # budget: 10% warm start over the 1-way marginals, 90% split across rounds,
    # each round half to SELECT and half to MEASURE.
    rho_warm = 0.10 * rho / len(ONE_WAY)
    rho_round = 0.90 * rho / rounds
    sigma_warm = 1.0 / np.sqrt(2 * rho_warm)
    sigma_meas = 1.0 / np.sqrt(2 * (rho_round / 2))
    eps_select = np.sqrt(8 * (rho_round / 2))

    meas, scales, repeats = {}, {}, {}
    for S in ONE_WAY:
        G = graphs.get(S) if graphs else None
        meas[S] = measure(marginal(true, S), sigma_warm, rng, G)
        scales[S] = sigma_warm * cost(S, G)[0]
        repeats[S] = 1

    model = np.full(SIZES, n / np.prod(SIZES))
    model = mle.fit(model, meas, scales, n, iters, graphs=graphs, totals=totals)

    picked = []
    for t in range(rounds):
        S = select(true, model, sigma_meas, eps_select, rng, graphs)
        picked.append(S)
        G = graphs.get(S) if graphs else None
        y = measure(marginal(true, S), sigma_meas, rng, G)
        if S in meas:
            # averaging k equal-variance measurements divides the noise by sqrt(k)
            meas[S] = (meas[S] * repeats[S] + y) / (repeats[S] + 1)
            repeats[S] += 1
        else:
            meas[S], repeats[S] = y, 1
        scales[S] = sigma_meas * cost(S, G)[0] / np.sqrt(repeats[S])
        model = mle.fit(model, meas, scales, n, iters, t0=(t + 1) * iters,
                        graphs=graphs, totals=totals)

    return mle.fit(model, meas, scales, n, iters * 5, t0=(rounds + 1) * iters,
                   graphs=graphs, totals=totals), picked


if __name__ == "__main__":
    import time

    from data import load

    _, true = load()
    for rho in (0.01, 0.04):
        for pol in (False, True):
            t = time.time()
            m, p = run(true, rho, 0, use_policy=pol)
            e1 = np.abs(marginal(true, (0,)) - marginal(m, (0,))).sum() / true.sum()
            e3 = np.abs(marginal(true, (0, 2, 4)) - marginal(m, (0, 2, 4))).sum() / true.sum()
            print(f"rho={rho:<7} policy={str(pol):<6} {time.time()-t:5.1f}s  "
                  f"age-1way L1={e1:.4f}  3way L1={e3:.4f}  distinct={len(set(p))}")
