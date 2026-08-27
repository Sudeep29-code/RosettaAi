from gnn.dataset import load_language_graphs


path = (
    "datasets/processed/"
    "ast/python/graphs.json"
)


graphs = load_language_graphs(
    path
)


print(
    "Number of graphs:",
    len(graphs)
)


graph = graphs[0]


print(
    "Node feature shape:",
    graph.x.shape
)


print(
    "Edge shape:",
    graph.edge_index.shape
)


print(
    "Language label:",
    graph.y
)