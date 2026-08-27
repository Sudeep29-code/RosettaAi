VALID_NODE_TYPES = {
    "FUNCTION",
    "PARAMETER",
    "CONDITION",
    "ELSE",
    "COMPARISON",
    "RETURN",
    "OPERATION",
    "VARIABLE",
    "CONSTANT",
    "FUNCTION_CALL",
    "METHOD_CALL",
    "DICTIONARY",
    "DICT_ENTRY",
    "SUBSCRIPT",
    "LIST",
    "ASSIGNMENT",
    "LOOP",
    "BREAK",
    "CONTINUE",
    "UNKNOWN"
}


CPP_NODE_MAP = {

    "function_definition": "FUNCTION",
    "function_declarator": "FUNCTION",

    "parameter_declaration": "PARAMETER",

    "identifier": "VARIABLE",

    "number_literal": "CONSTANT",
    "string_literal": "CONSTANT",
    "true": "CONSTANT",
    "false": "CONSTANT",
    "null": "CONSTANT",

    "if_statement": "CONDITION",

    "for_statement": "LOOP",
    "while_statement": "LOOP",
    "do_statement": "LOOP",

    "return_statement": "RETURN",

    "call_expression": "FUNCTION_CALL",

    "binary_expression": "OPERATION",
    "unary_expression": "OPERATION",

    # C++ assignment
    "assignment_expression": "ASSIGNMENT",

    "comparison_expression": "COMPARISON",
}


class ASTNormalizer:

    def normalize(self, graph, language=None):

        normalized_nodes = []
        normalized_edges = []

        language = (
            language.lower()
            if language
            else None
        )

        # =====================================================
        # NORMALIZE NODES
        # =====================================================

        for node in graph.get("nodes", []):

            node_type = node.get(
                "type",
                "UNKNOWN"
            )

            # -------------------------------------------------
            # C++ Tree-sitter → Common AST
            # -------------------------------------------------

            if language == "cpp":

                node_type = CPP_NODE_MAP.get(
                    node_type,
                    "UNKNOWN"
                )

            # -------------------------------------------------
            # Already normalized nodes
            # -------------------------------------------------

            elif node_type not in VALID_NODE_TYPES:

                node_type = "UNKNOWN"

            # -------------------------------------------------
            # Create normalized node
            # -------------------------------------------------

            normalized_node = {
                "id": node.get("id"),
                "type": node_type,
                "value": node.get("value")
            }

            # -------------------------------------------------
            # Preserve value_type
            #
            # Important for:
            # "Hello" → String
            # 10      → Integer
            # True    → Boolean
            # -------------------------------------------------

            if "value_type" in node:

                normalized_node["value_type"] = (
                    node["value_type"]
                )

            normalized_nodes.append(
                normalized_node
            )

        # =====================================================
        # NORMALIZE EDGES
        # =====================================================

        for edge in graph.get("edges", []):

            # -------------------------------------------------
            # Dictionary format
            # -------------------------------------------------

            if isinstance(edge, dict):

                if (
                    "source" in edge
                    and "target" in edge
                ):

                    normalized_edges.append({
                        "source": edge["source"],
                        "target": edge["target"]
                    })

            # -------------------------------------------------
            # Tuple / list format
            # -------------------------------------------------

            elif isinstance(
                edge,
                (list, tuple)
            ):

                if len(edge) == 2:

                    normalized_edges.append({
                        "source": edge[0],
                        "target": edge[1]
                    })

        # =====================================================
        # RETURN NORMALIZED GRAPH
        # =====================================================

        return {
            "nodes": normalized_nodes,
            "edges": normalized_edges
        }


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    example_graph = {

        "nodes": [

            {
                "id": 0,
                "type": "FUNCTION",
                "value": "greet"
            },

            {
                "id": 1,
                "type": "RETURN",
                "value": None
            },

            {
                "id": 2,
                "type": "CONSTANT",
                "value": "Hello",
                "value_type": "STRING"
            }

        ],

        "edges": [
            (0, 1),
            (1, 2)
        ]
    }

    normalizer = ASTNormalizer()

    result = normalizer.normalize(
        example_graph,
        "python"
    )

    print(result)