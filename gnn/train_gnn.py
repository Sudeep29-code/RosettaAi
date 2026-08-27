import torch

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

from gnn.dataset import load_language_graphs
from gnn.model import CodeGNN


DATASETS = [
    "datasets/processed/ast/python/graphs.json",
    "datasets/processed/ast/java/graphs.json",
    "datasets/processed/ast/javascript/graphs.json",
    "datasets/processed/ast/cpp/graphs.json",
]


def load_all_graphs():

    all_graphs = []

    for path in DATASETS:

        graphs = load_language_graphs(path)

        all_graphs.extend(graphs)

    return all_graphs


def calculate_accuracy(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            output, _ = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            predictions = output.argmax(
                dim=1
            )

            correct += (
                predictions == batch.y
            ).sum().item()

            total += batch.y.size(0)

    if total == 0:
        return 0.0

    return (
        correct / total
    ) * 100


def main():

    print("Loading graphs...")

    graphs = load_all_graphs()

    print(
        "Total graphs:",
        len(graphs)
    )

    # --------------------------------
    # Train / validation split
    # --------------------------------

    train_size = int(
        0.8 * len(graphs)
    )

    validation_size = (
        len(graphs) - train_size
    )

    train_dataset, validation_dataset = random_split(
        graphs,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        "Training graphs:",
        len(train_dataset)
    )

    print(
        "Validation graphs:",
        len(validation_dataset)
    )

    # --------------------------------
    # Data loaders
    # --------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        shuffle=False
    )

    # --------------------------------
    # Model
    # --------------------------------

    model = CodeGNN(
        num_classes=4
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    loss_function = torch.nn.CrossEntropyLoss()

    epochs = 20

    best_accuracy = 0.0

    # --------------------------------
    # Training
    # --------------------------------

    for epoch in range(epochs):

        model.train()

        total_loss = 0.0

        for batch in train_loader:

            optimizer.zero_grad()

            output, embedding = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            loss = loss_function(
                output,
                batch.y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(train_loader)
        )

        validation_accuracy = calculate_accuracy(
            model,
            validation_loader
        )

        if validation_accuracy > best_accuracy:

            best_accuracy = validation_accuracy

            torch.save(
                model.state_dict(),
                "gnn/code_gnn.pth"
            )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {average_loss:.4f} "
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

    print()
    print("GNN training completed.")

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        "Best model saved to "
        "gnn/code_gnn.pth"
    )


if __name__ == "__main__":
    main()