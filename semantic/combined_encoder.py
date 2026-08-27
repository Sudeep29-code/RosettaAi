import torch
import torch.nn as nn

from semantic.codebert import CodeBERTEncoder


class CombinedEncoder(nn.Module):

    def __init__(
        self,
        gnn_model
    ):

        super().__init__()

        self.gnn = gnn_model

        self.codebert = CodeBERTEncoder()

        self.projection = nn.Linear(
            128 + 768,
            256
        )

    def encode(
        self,
        code,
        graph
    ):

        # -------------------------
        # GNN representation
        # -------------------------

        self.gnn.eval()

        with torch.no_grad():

            _, gnn_embedding = self.gnn(
                graph.x,
                graph.edge_index,
                graph.batch
            )

        # -------------------------
        # CodeBERT representation
        # -------------------------

        codebert_embedding = (
            self.codebert.encode(code)
        )

        # -------------------------
        # Combine
        # -------------------------

        combined = torch.cat(
            [
                gnn_embedding,
                codebert_embedding
            ],
            dim=1
        )

        # -------------------------
        # Project
        # -------------------------

        semantic_embedding = (
            self.projection(combined)
        )

        return semantic_embedding