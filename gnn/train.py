import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool

from gnn.model import ASTGNN
from gnn.training_data import build_training_data


INPUT_DIM = 10
HIDDEN_DIM = 128
OUTPUT_DIM = 128

EPOCHS = 50
LEARNING_RATE = 0.001


def get_graph_embedding(model, x, edge_index):

    node_embeddings = model(
        x,
        edge_index
    )

    batch = torch.zeros(
        node_embeddings.size(0),
        dtype=torch.long
    )

    graph_embedding = global_mean_pool(
        node_embeddings,
        batch
    )

    return graph_embedding


def train():

    data = build_training_data()

    model = ASTGNN(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(
        "\nStarting GNN training..."
    )

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        # Group graphs by algorithm
        algorithm_embeddings = {}

        for item in data:

            embedding = get_graph_embedding(
                model,
                item["x"],
                item["edge_index"]
            )

            algorithm = item["algorithm"]

            if algorithm not in algorithm_embeddings:
                algorithm_embeddings[algorithm] = []

            algorithm_embeddings[algorithm].append(
                embedding
            )

        # Contrastive loss
        loss = torch.tensor(
            0.0,
            requires_grad=True
        )

        for embeddings in algorithm_embeddings.values():

            if len(embeddings) < 2:
                continue

            for i in range(
                len(embeddings)
            ):

                for j in range(
                    i + 1,
                    len(embeddings)
                ):

                    similarity = F.cosine_similarity(
                        embeddings[i],
                        embeddings[j]
                    )

                    # Encourage same-algorithm
                    # representations to be similar
                    loss = loss + (
                        1 - similarity
                    )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss = loss.item()

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"Loss: {total_loss:.4f}"
            )

    torch.save(
        model.state_dict(),
        "gnn/ast_gnn.pth"
    )

    print(
        "\nGNN training completed."
    )

    print(
        "Model saved to gnn/ast_gnn.pth"
    )


if __name__ == "__main__":

    train()