import torch

from torch_geometric.loader import DataLoader

from gnn.dataset import load_language_graphs
from gnn.model import CodeGNN

from semantic.combined_encoder import CombinedEncoder


# -------------------------
# Load graph
# -------------------------

graphs = load_language_graphs(
    "datasets/processed/ast/python/graphs.json"
)

graph = graphs[0]

loader = DataLoader(
    [graph],
    batch_size=1
)

batch = next(iter(loader))


# -------------------------
# Load trained GNN
# -------------------------

gnn = CodeGNN()

gnn.load_state_dict(
    torch.load(
        "gnn/code_gnn.pth",
        map_location="cpu"
    )
)


# -------------------------
# Combined encoder
# -------------------------

encoder = CombinedEncoder(
    gnn
)


# -------------------------
# Example code
# -------------------------

code = """
def factorial(n):
    if n == 0:
        return 1

    return n * factorial(n - 1)
"""


embedding = encoder.encode(
    code,
    batch
)


print(
    "Combined embedding shape:",
    embedding.shape
)