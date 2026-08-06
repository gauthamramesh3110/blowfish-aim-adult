"""Minimal AIM: the select-measure-generate loop, under standard DP.

  warm start   measure every 1-way marginal, fit
  each round   SELECT a 2-way marginal by exponential mechanism
               MEASURE it under Gaussian noise
               refit the model on everything measured so far
  finally      a longer fit, then sample synthetic records

Two neighbour relations are supported, selected by `bounded`:

  unbounded  a record is added or removed, as in AIM.  A marginal has L2
             sensitivity 1, and n is not public -- it is measured alongside
             the 1-way marginals.
  bounded    n is fixed and public, and a record moves between values.  A
             marginal has L2 sensitivity sqrt(2), and nothing is spent on n.

The relation matters because it decides what a Blowfish policy can relax.
Against bounded DP, which protects every substitution in one step, a threshold
policy removes most of those obligations.  Against unbounded DP it removes
nothing -- see docs/experiment-results.md.

Budget is tracked in zCDP throughout: the Gaussian mechanism with noise scale
sigma costs rho = 1/(2 sigma^2), and the exponential mechanism with parameter
eps costs rho = eps^2/8.

Simplifications against the published algorithm: the joint is small enough for
exact inference, so no graphical model is built; the round count is fixed with
no budget annealing; candidates are all 2-way marginals and the workload is all
3-way marginals.
"""
import collections

import numpy as np

import mle
import policy
from data import SIZES, marginal

Cost = collections.namedtuple("Cost", "delta penalty")

NDIM = len(SIZES)
CANDIDATES = [(i, j) for i in range(NDIM) for j in range(i + 1, NDIM)]
ONE_WAY = [(i,) for i in range(NDIM)]

def sensitivity(bounded):
    """L2 sensitivity of releasing a raw marginal.

    unbounded  a record is added or removed, so x moves by e_u        -> 1
    bounded    n is fixed, so a record moves: x moves by e_u - e_v    -> sqrt(2)
    """
    return np.sqrt(2.0) if bounded else 1.0


def cells(S):
    return int(np.prod([SIZES[i] for i in S]))


def score_sensitivity(graphs, bounded):
    """L1 sensitivity of SELECT's quality score.

    Adding or removing a record moves one cell by 1, so one L1 term moves by 1.
    Any substitution -- whether bounded DP's, or a policy's move along a graph
    edge -- moves two cells by 1 each, hence 2.
    """
    return 2.0 if (graphs or bounded) else 1.0


def cost(S, G, bounded):
    """Noise multiplier, and expected L1 reconstruction error at unit sigma.

    Stock AIM releases the marginal itself: one noisy cell per cell.  Under a
    policy it releases x_G instead, so the multiplier is the policy sensitivity
    and the error term is spec 6.2's generalised penalty.
    """
    return (Cost(sensitivity(bounded), cells(S)) if G is None
            else Cost(G.delta, G.penalty))


def measure(x, sigma, rng, G=None, bounded=False):
    """Noisy measurement of a marginal.

    Every record contributes a count of one to exactly one cell, so the whole
    marginal costs the same as a single cell would.  That is why AIM measures
    entire marginals, and why the shape of S does not affect the noise level.

    Under a policy the released quantity is x_G = P_G^-1 x -- weights on the
    policy graph's edges, not counts on its cells -- and the noise goes on
    there, where it stays isotropic.  It is never reconstructed back to cells.
    """
    if G is None:
        return x + rng.normal(0, sigma * sensitivity(bounded), size=x.shape)
    xg = G.transform(x)
    return xg + rng.normal(0, sigma * G.delta, size=xg.shape)


def select(true, model, sigma, eps, rng, graphs=None, bounded=False):
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
        c = cost(S, graphs.get(S) if graphs else None, bounded)
        q.append(np.abs(marginal(true, S) - marginal(model, S)).sum()
                 - np.sqrt(2.0 / np.pi) * sigma * c.delta * c.penalty)
    q = np.array(q)
    p = np.exp(eps * (q - q.max()) / (2 * score_sensitivity(graphs, bounded)))
    return CANDIDATES[rng.choice(len(CANDIDATES), p=p / p.sum())]


def run(true, rho, seed, rounds=10, iters=30, use_policy=False, bounded=False):
    """One AIM run.  Returns the fitted joint, and the marginals it selected."""
    rng = np.random.default_rng(seed)
    n = true.sum()

    # The graphs cost no budget: they depend only on the policy, never the
    # data.  Cached, so a sweep in one process pays the build cost once.
    graphs = ({S: policy.graph(S, bounded) for S in ONE_WAY + CANDIDATES}
              if use_policy else None)

    # budget: 10% warm start, 90% split across rounds, each round half to
    # SELECT and half to MEASURE.  Under unbounded DP the warm start also pays
    # for `n`, so it is split one extra way.
    releases = len(ONE_WAY) if bounded else len(ONE_WAY) + 1
    rho_warm = 0.10 * rho / releases
    rho_round = 0.90 * rho / rounds
    sigma_warm = 1.0 / np.sqrt(2 * rho_warm)
    sigma_meas = 1.0 / np.sqrt(2 * (rho_round / 2))
    eps_select = np.sqrt(8 * (rho_round / 2))

    # Under bounded DP n is fixed and public.  Under unbounded it is exactly
    # what differs between neighbours, so it costs budget: sensitivity 1, since
    # a bottom edge adds or removes one record and no other edge changes it.
    n_hat = n if bounded else n + rng.normal(0, sigma_warm)

    meas, scales, repeats = {}, {}, {}
    for S in ONE_WAY:
        G = graphs.get(S) if graphs else None
        meas[S] = measure(marginal(true, S), sigma_warm, rng, G, bounded)
        scales[S] = sigma_warm * cost(S, G, bounded).delta
        repeats[S] = 1

    model = np.full(SIZES, n_hat / np.prod(SIZES))
    model = mle.fit(model, meas, scales, n_hat, iters, graphs=graphs)

    picked = []
    for t in range(rounds):
        S = select(true, model, sigma_meas, eps_select, rng, graphs, bounded)
        picked.append(S)
        G = graphs.get(S) if graphs else None
        y = measure(marginal(true, S), sigma_meas, rng, G, bounded)
        if S in meas:
            # averaging k equal-variance measurements divides the noise by sqrt(k)
            meas[S] = (meas[S] * repeats[S] + y) / (repeats[S] + 1)
            repeats[S] += 1
        else:
            meas[S], repeats[S] = y, 1
        scales[S] = sigma_meas * cost(S, G, bounded).delta / np.sqrt(repeats[S])
        model = mle.fit(model, meas, scales, n_hat, iters, t0=(t + 1) * iters,
                        graphs=graphs)

    return mle.fit(model, meas, scales, n_hat, iters * 5,
                   t0=(rounds + 1) * iters, graphs=graphs), picked


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
