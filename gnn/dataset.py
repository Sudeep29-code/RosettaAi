import json
import torch

from torch_geometric.data import Data


NODE_TYPES = {
    "FUNCTION": 0,
    "PARAMETER": 1,
    "CONSTANT": 2,
    "VARIABLE": 3,
    "OPERATION": 4,
    "FUNCTION_CALL": 5,
    "CONDITION": 6,
    "RETURN": 7,
}


LANGUAGE_MAP = {
    "python": 0,
    "java": 1,
    "javascript": 2,
    "cpp": 3,
}


def load_graphs(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def graph_to_data(item):

    graph = item["graph"]

    nodes = graph["nodes"]
    edges = graph["edges"]

    # --------------------------------
    # Node features
    # --------------------------------

    node_features = []

    for node in nodes:

        node_type = node.get(
            "type",
            "VARIABLE"
        )

        feature = NODE_TYPES.get(
            node_type,
            len(NODE_TYPES)
        )

        node_features.append(
            [float(feature)]
        )

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    # --------------------------------
    # Edge index
    # --------------------------------

    edge_list = []

    for edge in edges:

        edge_list.append([
            edge["source"],
            edge["target"]
        ])

    if edge_list:

        edge_index = torch.tensor(
            edge_list,
            dtype=torch.long
        ).t().contiguous()

    else:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    # --------------------------------
    # Language
    # --------------------------------

    language = item["language"]

    y = torch.tensor(
        [LANGUAGE_MAP[language]],
        dtype=torch.long
    )

    return Data(
        x=x,
        edge_index=edge_index,
        y=y
    )


def load_language_graphs(path):

    graphs = load_graphs(path)

    return [
        graph_to_data(graph)
        for graph in graphs
    ]