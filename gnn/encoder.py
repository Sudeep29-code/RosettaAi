import torch

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from gnn.model import ASTGNN


class GNNEncoder:

    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=128
    ):

        self.model = ASTGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

    def encode(
        self,
        node_features,
        edges
    ):

        x = torch.tensor(
            node_features,
            dtype=torch.float
        )

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

        data = Data(
            x=x,
            edge_index=edge_index
        )

        node_embeddings = self.model(
            data.x,
            data.edge_index
        )

        # One embedding for the entire graph
        batch = torch.zeros(
            node_embeddings.size(0),
            dtype=torch.long
        )

        graph_embedding = global_mean_pool(
            node_embeddings,
            batch
        )

        return graph_embedding