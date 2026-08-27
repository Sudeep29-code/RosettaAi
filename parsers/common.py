class CommonAST:

    def __init__(self):

        self.nodes = []
        self.edges = []


    def add_node(
        self,
        node_type,
        value=None
    ):

        node_id = len(
            self.nodes
        )

        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "value": value
        })

        return node_id


    def add_edge(
        self,
        source,
        target
    ):

        self.edges.append({
            "source": source,
            "target": target
        })


    def get_graph(self):

        return {
            "nodes": self.nodes,
            "edges": self.edges
        }