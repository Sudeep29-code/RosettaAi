import json
import torch

from parsers.parser_factory import parse_code
from semantic.normalizer import ASTNormalizer

from gnn.features import (
    create_node_features,
    create_edges
)


DATASET_PATH = "datasets/algorithms/multilingual_algorithms.json"

LANGUAGES = [
    "python",
    "java",
    "cpp",
    "javascript"
]


def load_dataset():

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def graph_to_tensors(code, language):

    graph = parse_code(
        code,
        language
    )

    normalizer = ASTNormalizer()

    graph = normalizer.normalize(graph)

    features = create_node_features(
        graph
    )

    edges = create_edges(
        graph
    )

    x = torch.tensor(
        features,
        dtype=torch.float
    )

    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()

    return x, edge_index


def build_training_data():

    dataset = load_dataset()

    training_data = []

    for item in dataset:

        algorithm = item["algorithm"]

        for language in LANGUAGES:

            code = item[language]

            x, edge_index = graph_to_tensors(
                code,
                language
            )

            training_data.append({
                "algorithm": algorithm,
                "language": language,
                "x": x,
                "edge_index": edge_index
            })

    return training_data


if __name__ == "__main__":

    data = build_training_data()

    print(
        "Total training graphs:",
        len(data)
    )

    for item in data:

        print(
            item["algorithm"],
            "|",
            item["language"],
            "| nodes:",
            item["x"].shape[0]
        )