"""Blowfish policy graphs, built explicitly -- one P_G per measured marginal.

A Blowfish policy is one graph per attribute.  Two databases are neighbours if
one record moves along a single edge, so an edge (u,v) is a promise that values
u and v stay indistinguishable.  More edges = stronger protection.  Attributes
combine by the graph Cartesian product: two cells are adjacent iff they differ
on exactly one axis, and that axis's two values are joined in its own graph.

The transform (Design paper, Theorem 4.1) is

    W x  =  W_G x_G      with   x_G = P_G^-1 x,   W_G = W P_G

so running a standard-DP mechanism on (W_G, x_G) yields the Blowfish guarantee
on (W, x).  P_G is the signed vertex-edge incidence matrix and its right
inverse is P_G^-1 = P_G^T (P_G P_G^T)^-1 (Design paper, Lemma 4.8).

Neighbours differ by adding or removing one record, as in AIM.  That is the
Design paper's Case I -- the graph carries an extra vertex `bottom` meaning
"this record is absent", every cell is joined to it, and a bottom-edge
contributes a column with a single +1 rather than a +1/-1 pair.

Two consequences, both good:

  * P_G P_G^T = L_graph + I is positive definite, so nothing has to be grounded
    and there is no per-component bookkeeping.  `bottom` is simply vertex k,
    pinned at zero.
  * n is not public -- it is exactly what differs between neighbours -- so
    aim.py measures it rather than reading it.  Partition block totals are
    likewise not free; they cost budget like anything else.

The product graph is materialised as an explicit edge list and L is inverted
densely.  P_G itself is never built: every operation is a gather or scatter
over the edge list.  The worst case is age x hours -- 6,862 cells, 104,532
edges, about 6s to build and 377 MB for Z.
"""
import numpy as np

from data import SIZES

NDIM = len(SIZES)


# ---- per-attribute graphs (spec section 4) ---------------------------------

def complete(k):
    """Full protection: every pair indistinguishable, standard-DP strength."""
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def threshold(k, theta):
    """Values within theta of each other are indistinguishable."""
    return [(i, j) for i in range(k) for j in range(i + 1, min(i + theta + 1, k))]


def blocks_graph(blocks):
    """Partition: complete inside each block, nothing across it."""
    return [(i, j) for lo, hi in blocks
            for i in range(lo, hi) for j in range(i + 1, hi)]


# One entry per attribute of data.ATTRS, matching spec section 4.
WORKCLASS_BLOCKS = [(0, 3), (3, 4), (4, 6), (6, 9)]   # gov / private / self / unclear

POLICY = [
    ("threshold", 7),                 # age            -- within 7 years
    ("threshold", 8),                 # hours.per.week -- within 8 hours
    ("threshold", 2),                 # education.num  -- within 2 levels
    ("partition", WORKCLASS_BLOCKS),  # workclass      -- 4 blocks
    ("full", None),                   # income         -- complete graph K_2
]


def axis_edges(a):
    """The policy graph on attribute a, as an edge list."""
    kind, arg = POLICY[a]
    if kind == "threshold":
        return threshold(SIZES[a], arg)
    if kind == "partition":
        return blocks_graph(arg)
    return complete(SIZES[a])


# ---- the policy graph of one marginal --------------------------------------

def product_edges(S):
    """Edges of the Cartesian product graph, as two arrays of cell indices.

    Two cells are joined iff they differ on exactly one axis, and that axis's
    two values are joined in its own graph.  `np.take(idx, u, axis=ax)` is the
    slice of cells whose axis-`ax` value is `u`, so one per-axis edge becomes a
    whole column of cell edges at once.

    The product is enumerated directly rather than factored per axis.
    """
    shape = [SIZES[i] for i in S]
    idx = np.arange(int(np.prod(shape))).reshape(shape)
    U, V = [], []
    for ax, a in enumerate(S):
        for u, v in axis_edges(a):
            U.append(np.take(idx, u, axis=ax).ravel())
            V.append(np.take(idx, v, axis=ax).ravel())
    return np.concatenate(U), np.concatenate(V)


class Graph:
    """P_G for one marginal: edge list, inverse Laplacian, sensitivity.

    Vertex `k` is `bottom`.  It carries no row of P_G and its level is pinned
    at zero, so a bottom-edge column reads as a bare +1 -- exactly the Case I
    construction, and the reason no grounding is needed.
    """

    def __init__(self, S):
        self.S = S
        self.shape = [SIZES[i] for i in S]
        self.k = k = int(np.prod(self.shape))
        self.bottom = k                 # one extra vertex, pinned at level 0

        # the policy edges, plus one bottom-edge per cell (the record may be
        # present or absent).  `to_bottom` splits the two kinds thereafter.
        gu, gv = product_edges(S)
        self.U = np.concatenate([gu, np.arange(k)])
        self.V = np.concatenate([gv, np.full(k, self.bottom)])
        self.to_bottom = self.V == self.bottom
        u, v = self.U[~self.to_bottom], self.V[~self.to_bottom]

        # L = P_G P_G^T.  A bottom-edge column is a bare +1, so it lands on the
        # diagonal only -- which is exactly why L = L_graph + I is invertible.
        deg = np.bincount(self.U, minlength=k) + np.bincount(v, minlength=k)
        L = np.zeros((k, k))
        L[np.arange(k), np.arange(k)] = deg
        np.add.at(L, (u, v), -1.0)
        np.add.at(L, (v, u), -1.0)
        self.Z = np.linalg.inv(L)

        # Delta_2 = max column L2 norm of P_G^-1 P_G.  Expanding one column,
        # ||P_G^T Z c||^2 = c^T Z c -- the edge's effective resistance.
        r = np.empty(len(self.U))
        r[~self.to_bottom] = self.Z[u, u] + self.Z[v, v] - 2 * self.Z[u, v]
        b = self.U[self.to_bottom]      # bottom reads as zero, leaving Z[u,u]
        r[self.to_bottom] = self.Z[b, b]
        self.delta = float(np.sqrt(r.max()))

        # SELECT penalty, spec 6.2: sum_i sqrt(inv(A^T A)[i,i]) with A = P_G^-1.
        # A^T A = Z P_G P_G^T Z = Z, so inv(A^T A) = L and the diagonal is the
        # degree.  Collapses to the cell count when the only edges are the
        # bottom ones, which is stock AIM's A = I.
        self.penalty = float(np.sqrt(deg).sum())

    def __repr__(self):
        return (f"Graph{self.S} cells={self.k} edges={len(self.U)} "
                f"delta={self.delta:.4f}")

    def levels(self, x):
        """Z @ x, with `bottom` appended at zero so z[U]-z[V] just works."""
        z = np.zeros(self.bottom + 1)
        z[:self.k] = self.Z @ x.ravel()
        return z

    def transform(self, x):
        """Cell-space marginal -> edge weights x_G = P_G^-1 x."""
        z = self.levels(x)
        return z[self.U] - z[self.V]

    def back(self, r):
        """Edge-space residual -> cell-shaped gradient, i.e. (P_G^-1)^T r."""
        w = (np.bincount(self.U, weights=r, minlength=self.bottom + 1)
             - np.bincount(self.V, weights=r, minlength=self.bottom + 1))
        return (self.Z @ w[:self.k]).reshape(self.shape)


_CACHE = {}


def graph(S):
    """Policy graph for marginal S, built once and reused.

    Data independent -- it depends only on the policy, so building it costs no
    privacy budget and it can be cached freely.
    """
    if S not in _CACHE:
        _CACHE[S] = Graph(S)
    return _CACHE[S]


if __name__ == "__main__":
    import time

    from data import ATTRS, load, marginal

    print("--- 1-way marginals ---")
    print(f"{'attr':<16}{'k':>5}{'edges':>8}{'Delta2':>10}{'penalty':>10}")
    for a, name in enumerate(ATTRS):
        g = graph((a,))
        print(f"{name:<16}{g.k:>5}{len(g.U):>8}{g.delta:>10.4f}{g.penalty:>10.1f}")

    print("\n--- 2-way candidates ---")
    print(f"{'S':<10}{'cells':>8}{'edges':>9}{'Delta2':>10}{'build':>9}")
    for i in range(NDIM):
        for j in range(i + 1, NDIM):
            t = time.time()
            g = graph((i, j))
            print(f"{str((i,j)):<10}{g.k:>8}{len(g.U):>9}"
                  f"{g.delta:>10.4f}{time.time()-t:>8.1f}s")

    print("\n--- round trip: P_G x_G must return every cell, not just some ---")
    _, true = load()
    for S in [(0,), (3,), (2, 4), (0, 3)]:
        g = graph(S)
        x = marginal(true, S)
        xg = g.transform(x)
        back = (np.bincount(g.U, weights=xg, minlength=g.k + 1)
                - np.bincount(g.V, weights=xg, minlength=g.k + 1))[:g.k]
        print(f"  S={str(S):<8} max |P_G x_G - x| = "
              f"{np.abs(back - x.ravel()).max():.3e}")

    print("\n--- a tree must give exactly 1 (Design paper, Lemma 4.9) ---")
    # a path with bottom attached to one end only is a tree; our graphs attach
    # bottom to every cell, so this checks the machinery, not the policy.
    g = graph((4,))
    print(f"  income (K_2 + 2 bottom edges): Delta2 = {g.delta:.4f}")
