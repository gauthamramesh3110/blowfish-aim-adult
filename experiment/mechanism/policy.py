"""Blowfish policy graphs, built explicitly -- one P_G per measured marginal.

A Blowfish policy is one graph per attribute.  Two databases are neighbours if
one record moves along a single edge, so an edge (u,v) is a promise that values
u and v stay indistinguishable.  More edges = stronger protection.  Attributes
combine by the graph Cartesian product: two cells are adjacent iff they differ
on exactly one axis, and that axis's two values are joined in its own graph.

The transform (Design paper, Theorem 4.1) is

    W x  =  W_G x_G      with   x_G = P_G^-1 x,   W_G = W P_G

so running a standard-DP mechanism on (W_G, x_G) yields the Blowfish guarantee
on (W, x).  P_G is the signed vertex-edge incidence matrix -- one row per cell,
one column per edge, +1 and -1 at its endpoints -- and its right inverse is
P_G^-1 = P_G^T (P_G P_G^T)^-1 (Design paper, Lemma 4.8).

Bounded DP is used, so n is public.  The plain incidence matrix is rank
deficient -- the all-ones vector spans the null space of each component -- so
one vertex per connected component is folded into a bottom symbol (Design
paper, Case II).  P_G P_G^T is then the *grounded Laplacian*, which is
invertible.  A record can never leave its component, so component totals are
invariant under the neighbour relation: sensitivity 0, released exactly at no
privacy cost (spec 4.2).  Here the only partitioned attribute is workclass, so
that free information is exactly its four block totals.

No Kronecker structure is used.  The product graph is materialised as an
explicit edge list and the grounded Laplacian is inverted densely.  P_G itself
is never built: every operation is a gather or scatter over the edge list.  The
worst case is age x hours -- 6,862 cells, 97,670 edges, 4s and 1.3 GB.
"""
import numpy as np

from data import SIZES, expand

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

PARTITIONED = [a for a, (kind, _) in enumerate(POLICY) if kind == "partition"]


def axis_edges(a):
    """The policy graph on attribute a, as an edge list."""
    kind, arg = POLICY[a]
    if kind == "threshold":
        return threshold(SIZES[a], arg)
    if kind == "partition":
        return blocks_graph(arg)
    return complete(SIZES[a])


def block_of(a):
    """Map each value of a partitioned axis to its block index."""
    b = np.zeros(SIZES[a], int)
    for bi, (lo, hi) in enumerate(POLICY[a][1]):
        b[lo:hi] = bi
    return b


# ---- free information: partition block totals ------------------------------

def free_totals(joint):
    """Counts per partition block.  Invariant under the policy, so exact.

    A record can only move along an edge, and there are no edges across blocks,
    so these numbers are identical in every pair of neighbouring databases.
    Sensitivity 0 means they cost no budget (spec 4.2, 6.1 step 3).
    """
    out = {}
    for a in PARTITIONED:
        m = joint.sum(axis=tuple(i for i in range(NDIM) if i != a))
        out[a] = np.bincount(block_of(a), weights=m, minlength=len(POLICY[a][1]))
    return out


def renormalise(model, totals):
    """Rescale the model so each block carries its exact total.

    This replaces `model *= n/model.sum()` when a policy is in force: the same
    idea, but pinning every block rather than only the grand total.
    """
    for a, t in totals.items():
        b = block_of(a)
        cur = np.bincount(b, weights=model.sum(
            axis=tuple(i for i in range(NDIM) if i != a)), minlength=len(t))
        model *= expand(np.where(cur > 0, t / np.maximum(cur, 1e-12), 1.0)[b], (a,))
    return model


# ---- the policy graph of one marginal --------------------------------------

def components(k, U, V):
    """Connected-component root of every vertex, by union-find."""
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(U.tolist(), V.tolist()):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    return np.array([find(i) for i in range(k)])


class Graph:
    """P_G for one marginal: edge list, grounded Laplacian inverse, sensitivity."""

    def __init__(self, S):
        self.S = S
        self.shape = [SIZES[i] for i in S]
        self.k = int(np.prod(self.shape))

        # Cartesian product, built directly -- no Kronecker algebra.
        idx = np.arange(self.k).reshape(self.shape)
        U, V = [], []
        for ax, a in enumerate(S):
            for u, v in axis_edges(a):
                U.append(np.take(idx, u, axis=ax).ravel())
                V.append(np.take(idx, v, axis=ax).ravel())
        self.U = np.concatenate(U) if U else np.zeros(0, int)
        self.V = np.concatenate(V) if V else np.zeros(0, int)

        # fold one vertex per component into bottom (Design paper, Case II)
        root = components(self.k, self.U, self.V)
        self.ground = np.sort(np.unique(root, return_index=True)[1])
        self.keep = np.setdiff1d(np.arange(self.k), self.ground)
        self.pos = np.full(self.k, -1)
        self.pos[self.keep] = np.arange(len(self.keep))

        # grounded Laplacian L = P_G P_G^T, formed from the edge list
        m = len(self.keep)
        pu, pv = self.pos[self.U], self.pos[self.V]
        ku, kv = pu >= 0, pv >= 0
        L = np.zeros((m, m))
        deg = (np.bincount(pu[ku], minlength=m) + np.bincount(pv[kv], minlength=m))
        L[np.arange(m), np.arange(m)] = deg
        both = ku & kv
        np.add.at(L, (pu[both], pv[both]), -1.0)
        np.add.at(L, (pv[both], pu[both]), -1.0)
        self.Z = np.linalg.inv(L)

        # Delta_2 = max column L2 norm of P_G^-1 P_G.  Expanding that column,
        # ||P_G^T Z (e_u - e_v)||^2 = (e_u - e_v)^T Z (e_u - e_v), the effective
        # resistance of the edge.  Grounded vertices read as zero.
        su, sv = np.maximum(pu, 0), np.maximum(pv, 0)
        zuu = np.where(ku, self.Z[su, su], 0.0)
        zvv = np.where(kv, self.Z[sv, sv], 0.0)
        zuv = np.where(both, self.Z[su, sv], 0.0)
        self.delta = float(np.sqrt((zuu + zvv - 2 * zuv).max()))

        # SELECT penalty, spec 6.2: sum_i sqrt(inv(A^T A)[i,i]) with A = P_G^-1.
        # A^T A = Z P_G P_G^T Z = Z, so inv(A^T A) = L and the diagonal is the
        # degree.  Collapses to the cell count when the graph is a perfect
        # matching, which is stock AIM's A = I.
        self.penalty = float(np.sqrt(deg).sum())

    def __repr__(self):
        return (f"Graph{self.S} cells={self.k} edges={len(self.U)} "
                f"comps={len(self.ground)} delta={self.delta:.4f}")

    def transform(self, x):
        """Cell-space marginal -> edge weights x_G = P_G^-1 x."""
        z = np.zeros(self.k)
        z[self.keep] = self.Z @ x.ravel()[self.keep]
        return z[self.U] - z[self.V]

    def back(self, r):
        """Edge-space residual -> cell-shaped gradient, i.e. (P_G^-1)^T r."""
        w = (np.bincount(self.U, weights=r, minlength=self.k)
             - np.bincount(self.V, weights=r, minlength=self.k))
        g = np.zeros(self.k)
        g[self.keep] = self.Z @ w[self.keep]
        return g.reshape(self.shape)


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

    from data import ATTRS, load

    print("--- spec section 7 reference values, 1-way ---")
    print(f"{'attr':<16}{'k':>5}{'edges':>8}{'comps':>7}{'Delta2':>10}")
    for a, name in enumerate(ATTRS):
        g = graph((a,))
        print(f"{name:<16}{g.k:>5}{len(g.U):>8}{len(g.ground):>7}{g.delta:>10.4f}")

    print("\n--- 2-way candidates ---")
    print(f"{'S':<10}{'cells':>8}{'edges':>9}{'comps':>7}{'Delta2':>10}{'build':>9}")
    for i in range(NDIM):
        for j in range(i + 1, NDIM):
            t = time.time()
            g = graph((i, j))
            print(f"{str((i,j)):<10}{g.k:>8}{len(g.U):>9}{len(g.ground):>7}"
                  f"{g.delta:>10.4f}{time.time()-t:>8.1f}s")

    print("\n--- round trip: P_G x_G must return the kept cells exactly ---")
    _, true = load()
    from data import marginal
    for S in [(0,), (3,), (2, 4), (0, 3)]:
        g = graph(S)
        x = marginal(true, S)
        xg = g.transform(x)
        back = (np.bincount(g.U, weights=xg, minlength=g.k)
                - np.bincount(g.V, weights=xg, minlength=g.k))
        err = np.abs(back[g.keep] - x.ravel()[g.keep]).max()
        print(f"  S={str(S):<8} max |P_G x_G - x| over kept cells = {err:.3e}")

    print("\n--- free block totals (workclass) ---")
    print(" ", free_totals(true)[3])
