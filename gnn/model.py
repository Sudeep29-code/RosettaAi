import torch
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    global_mean_pool
)


class CodeGNN(torch.nn.Module):

    def __init__(
        self,
        input_dim=1,
        hidden_dim=64,
        embedding_dim=128,
        num_classes=4
    ):

        super().__init__()

        self.conv1 = GCNConv(
            input_dim,
            hidden_dim
        )

        self.conv2 = GCNConv(
            hidden_dim,
            embedding_dim
        )

        self.classifier = torch.nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):

        x = self.conv1(
            x,
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        embedding = global_mean_pool(
            x,
            batch
        )

        output = self.classifier(
            embedding
        )

        return output, embedding