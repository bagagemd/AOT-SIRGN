# Arbitrary Order Temporal SirGN

Structural node embeddings for temporal graphs, using SirGN with arbitrary-order temporal aggregation.

Each iteration clusters nodes by their current representation. It then rebuilds each node's representation from its neighbors' soft cluster memberships over time. Interactions are weighted by an exponential time decay, and aggregation can go up to any order.

## Requirements

- Python 3
- numpy
- scikit-learn
- pandas (for loading edge lists)

```bash
pip install numpy scikit-learn pandas
```

## Input format

A whitespace-separated edge list with one temporal edge per line:

```
source target timestamp
```

Edges are treated as undirected.

## Usage

```python
import pandas as pd
from temporal_sirgn import loader, temporalSirGN

edges = pd.read_csv("edges.txt", sep=" ", header=None)

l = loader()
l.read(edges)

emb = temporalSirGN(l.G, n=10, ordr=3, alpha=1.0, iter=5)
l.storeEmb("embeddings.txt", emb)
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `n`       | Number of clusters per iteration |
| `ordr`    | Maximum aggregation order (orders 1 through `ordr`; minimum 2) |
| `alpha`   | Time-decay scale |
| `iter`    | Number of refinement iterations |

## Output

`storeEmb` writes one line per node:

```
<node_id> <v1> <v2> ... <vd>
```
