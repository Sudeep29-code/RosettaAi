import os
import torch

from gnn.model import CodeGNN
from parsers.parser_factory import parse_code
from semantic.normalizer import ASTNormalizer
from translation.rules import TRANSLATION_RULES


class SemanticTranslator:

    def __init__(self):

        self.normalizer = ASTNormalizer()

        self.model = None
        self.gnn_available = False

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "gnn",
            "code_gnn.pth"
        )

        try:

            self.model = CodeGNN(
                num_classes=4
            )

            self.model.load_state_dict(
                torch.load(
                    model_path,
                    map_location="cpu"
                )
            )

            self.model.eval()

            self.gnn_available = True

            print("GNN model loaded successfully")

        except Exception as e:

            print(
                "Warning: GNN model could not be loaded."
            )

            print(
                "Reason:",
                e
            )

            self.model = None
            self.gnn_available = False


    # ============================================
    # BUILD GRAPH
    # ============================================

    def build_graph(self, code, language):

        graph = parse_code(
            code,
            language
        )

        graph = self.normalizer.normalize(
            graph,
            language
        )

        return graph


    # ============================================
    # GRAPH TO TENSORS
    # ============================================

    def graph_to_tensors(self, graph):

        nodes = graph.get(
            "nodes",
            []
        )

        edges = graph.get(
            "edges",
            []
        )

        node_types = {
            "FUNCTION": 0,
            "PARAMETER": 1,
            "CONSTANT": 2,
            "VARIABLE": 3,
            "OPERATION": 4,
            "FUNCTION_CALL": 5,
            "CONDITION": 6,
            "RETURN": 7,
            "ASSIGNMENT": 8,
            "ELSE": 9,
            "LOOP": 10,
            "UNKNOWN": 11
        }

        features = []

        for node in nodes:

            node_type = node.get(
                "type",
                "UNKNOWN"
            )

            value = node_types.get(
                node_type,
                node_types["UNKNOWN"]
            )

            features.append(
                [float(value)]
            )

        # Prevent empty graph errors

        if not features:

            features = [[0.0]]

        x = torch.tensor(
            features,
            dtype=torch.float
        )

        edge_list = []

        for edge in edges:

            if isinstance(edge, dict):

                source = edge.get("source")
                target = edge.get("target")

            else:

                source = edge[0]
                target = edge[1]

            if source is not None and target is not None:

                edge_list.append(
                    [source, target]
                )

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

        batch = torch.zeros(
            x.shape[0],
            dtype=torch.long
        )

        return x, edge_index, batch


    # ============================================
    # GET EMBEDDING
    # ============================================

    def get_embedding(self, code, language):

        graph = self.build_graph(
            code,
            language
        )

        # If GNN is unavailable,
        # return a safe empty embedding

        if not self.gnn_available:

            output = torch.zeros(
                (1, 4)
            )

            embedding = torch.zeros(
                (1, 128)
            )

            return output, embedding

        x, edge_index, batch = (
            self.graph_to_tensors(graph)
        )

        with torch.no_grad():

            output, embedding = self.model(
                x,
                edge_index,
                batch
            )

        return output, embedding


    # ============================================
    # TRANSLATE
    # ============================================

    def translate(
        self,
        code,
        source,
        target
    ):

        source = source.lower()
        target = target.lower()

        if source == target:

            return {
                "source_language": source,
                "target_language": target,
                "translated_code": code
            }

        if source not in TRANSLATION_RULES:

            raise ValueError(
                f"Unsupported source language: {source}"
            )

        if target not in TRANSLATION_RULES[source]:

            raise ValueError(
                f"Unsupported target language: {target}"
            )

        output, embedding = self.get_embedding(
            code,
            source
        )

        result = code

        rules = TRANSLATION_RULES[
            source
        ][target]

        for old, new in rules.items():

            result = result.replace(
                old,
                new
            )

        return {
            "source_language": source,
            "target_language": target,
            "translated_code": result,
            "embedding": embedding,
            "prediction": torch.argmax(
                output,
                dim=1
            ).item()
        }