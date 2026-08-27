import torch

from gnn.model import CodeGNN
from gnn.dataset import graph_to_data


MODEL_PATH = "gnn/code_gnn.pth"


LANGUAGE_NAMES = {
    0: "Python",
    1: "Java",
    2: "JavaScript",
    3: "C++"
}


def run_graph_test(graph_item):
    """
    Run GNN prediction on a graph.
    """

    data = graph_to_data(graph_item)

    model = CodeGNN(
        num_classes=4
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    # Add batch dimension
    batch = torch.zeros(
        data.x.size(0),
        dtype=torch.long
    )

    with torch.no_grad():

        output, embedding = model(
            data.x,
            data.edge_index,
            batch
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        prediction = output.argmax(
            dim=1
        ).item()

    print()
    print("Predicted language:",
          LANGUAGE_NAMES[prediction])

    print(
        "Confidence:",
        f"{probabilities[0][prediction].item() * 100:.2f}%"
    )

    print(
        "Embedding shape:",
        embedding.shape
    )

    print()
    print("First 10 embedding values:")

    print(
        embedding[0][:10]
    )

    return prediction, embedding


def test_graph():
    """
    Pytest test for GNN language prediction.
    """

    example = {
        "language": "python",

        "graph": {
            "nodes": [

                {
                    "id": 0,
                    "type": "FUNCTION",
                    "value": "add"
                },

                {
                    "id": 1,
                    "type": "PARAMETER",
                    "value": "a"
                },

                {
                    "id": 2,
                    "type": "PARAMETER",
                    "value": "b"
                },

                {
                    "id": 3,
                    "type": "RETURN",
                    "value": "return"
                },

                {
                    "id": 4,
                    "type": "OPERATION",
                    "value": "+"
                }
            ],

            "edges": [

                {
                    "source": 0,
                    "target": 1
                },

                {
                    "source": 0,
                    "target": 2
                },

                {
                    "source": 0,
                    "target": 3
                },

                {
                    "source": 3,
                    "target": 4
                }
            ]
        }
    }

    prediction, embedding = run_graph_test(
        example
    )

    # Basic sanity checks
    assert prediction in LANGUAGE_NAMES

    assert embedding is not None

    assert embedding.shape[0] == 1


if __name__ == "__main__":

    example = {
        "language": "python",

        "graph": {
            "nodes": [
                {
                    "id": 0,
                    "type": "FUNCTION",
                    "value": "add"
                },
                {
                    "id": 1,
                    "type": "PARAMETER",
                    "value": "a"
                },
                {
                    "id": 2,
                    "type": "PARAMETER",
                    "value": "b"
                },
                {
                    "id": 3,
                    "type": "RETURN",
                    "value": "return"
                },
                {
                    "id": 4,
                    "type": "OPERATION",
                    "value": "+"
                }
            ],

            "edges": [
                {
                    "source": 0,
                    "target": 1
                },
                {
                    "source": 0,
                    "target": 2
                },
                {
                    "source": 0,
                    "target": 3
                },
                {
                    "source": 3,
                    "target": 4
                }
            ]
        }
    }

    run_graph_test(example)