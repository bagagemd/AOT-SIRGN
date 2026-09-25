import itertools

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class loader:
    """Builds a temporal adjacency structure from an edge list."""

    def __init__(self):
        self.countID = 0
        self.G = {}
        self.co = {}     # original node label -> internal id
        self.revco = {}  # internal id -> original node label

    def nodeID(self, x):
        """Return the internal id for node label ``x``, assigning a new one if needed."""
        if x not in self.co:
            self.co[x] = self.countID
            self.revco[self.countID] = x
            self.countID += 1
        return self.co[x]

    def read(self, file):
        """Load edges from a DataFrame with columns (source, target, timestamp)."""
        x = file.values
        for a in range(x.shape[0]):
            i = self.nodeID(x[a, 0])
            j = self.nodeID(x[a, 1])
            self.addEdge((i, j, float(x[a, 2])))
        self.fixG()

    def storeEmb(self, file, data):
        """Write embeddings as ``<original_node_label> <v1> <v2> ...``, one node per line."""
        with open(file, "w") as f:
            for a in range(data.shape[0]):
                row = " ".join(str(val) for val in data[a])
                f.write(f"{int(self.revco[a])} {row}\n")

    def fixG(self):
        """Convert each node's {timestamp: set(neighbors)} into a timestamp-descending list."""
        for g in range(len(self.G)):
            timestamps = sorted(self.G[g], reverse=True)
            self.G[g] = [(t, np.array(list(self.G[g][t]))) for t in timestamps]

    def addEdge(self, s):
        """Add an undirected temporal edge ``(u, v, timestamp)``."""
        u, v, t = s
        self.G.setdefault(u, {}).setdefault(t, set()).add(v)
        self.G.setdefault(v, {}).setdefault(t, set()).add(u)


def temporalAggregation1(embd, G, v, alpha, ordr):
    k = embd.shape[1]
    T = len(G[v])

    h1 = np.zeros((1, k))
    w = []
    for _, nbrs in G[v]:
        wi = embd[nbrs].sum(axis=0)
        h1 += wi
        w.append(wi.reshape((k, 1)))

    h2 = np.zeros((k, k))
    z = np.zeros((1, k))
    for j in range(1, T):
        t_j = G[v][j][0]
        t_prev = G[v][j - 1][0]
        z = np.exp((t_j - t_prev) / alpha) * (w[j - 1].T + z)
        h2 += w[j] * z

    iprev = [np.zeros((1, k))]
    z1 = np.zeros((1, k))
    for j in range(1, T):
        t_j = G[v][j][0]
        t_prev = G[v][j - 1][0]
        z1 = np.exp((t_j - t_prev) / alpha) * (w[j - 1].T + z1)
        iprev.append(z1.copy())

    allA = []
    for c in range(2, ordr):
        z2 = np.zeros((k ** (c - 1), k))
        icurr = [z2.flatten()]
        for l in range(1, T):
            t_l = G[v][l][0]
            t_prev = G[v][l - 1][0]
            z2 = np.exp(t_l - t_prev) * ((w[l - 1] * iprev[l - 1]).T + z2)
            icurr.append(z2.flatten())

        A = np.zeros((k, k ** c))
        for m in range(T):
            A += w[m] * icurr[m]
        allA.append(A.flatten())
        iprev = icurr

    allA = list(itertools.chain.from_iterable(allA))
    return np.hstack([h1.flatten(), h2.flatten(), allA])


def temporalAggregation(embd, G, alpha, ordr):
    """Run ``temporalAggregation1`` over every node and stack the results."""
    return np.vstack([temporalAggregation1(embd, G, v, alpha, ordr) for v in range(len(G))])


def getnumber(emb):
    """Number of distinct rows in ``emb`` (i.e. distinct structural roles)."""
    return len(np.unique(emb, axis=0))


def temporalSirGN(G, n, ordr, alpha, iter=5):
    """Temporal SirGN embedding.

    Parameters
    ----------
    G : dict
        Temporal graph from ``loader.G``.
    n : int
        Number of clusters per iteration.
    ordr : int
        Maximum aggregation order (orders 1..max(ordr, 2) are computed).
    alpha : float
        Time-decay scale.
    iter : int
        Number of refinement iterations.
    """
    nv = len(G)
    embd = np.full((nv, n), 1 / n)
    emb = temporalAggregation(embd, G, alpha, ordr)

    for i in range(iter):
        print(i)
        emb1 = MinMaxScaler().fit_transform(emb)
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=1).fit(emb1)
        dist = kmeans.transform(emb1)

        # Soft cluster membership: closer centroids get larger weight, rows sum to 1
        M = dist.max(axis=1, keepdims=True)
        m = dist.min(axis=1, keepdims=True)
        subx = (M - dist) / (M - m)
        subx = subx / subx.sum(axis=1, keepdims=True)

        emb = temporalAggregation(subx, G, alpha, ordr)
    return emb
