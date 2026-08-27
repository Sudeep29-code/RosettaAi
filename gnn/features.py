NODE_TYPES = {
    "FUNCTION": 0,
    "PARAMETER": 1,
    "CONDITION": 2,
    "COMPARISON": 3,
    "RETURN": 4,
    "OPERATION": 5,
    "VARIABLE": 6,
    "CONSTANT": 7,
    "FUNCTION_CALL": 8,
    "UNKNOWN": 9
}


def create_node_features(graph):

    features = []

    for node in graph["nodes"]:

        node_type = node["type"]

        index = NODE_TYPES.get(
            node_type,
            NODE_TYPES["UNKNOWN"]
        )

        feature = [0] * len(NODE_TYPES)

        feature[index] = 1

        features.append(feature)

    return features


def create_edges(graph):

    edges = []

    for edge in graph["edges"]:

        if isinstance(edge, dict):

            source = edge["source"]
            target = edge["target"]

        else:

            source = edge[0]
            target = edge[1]

        edges.append(
            [source, target]
        )

    return edges