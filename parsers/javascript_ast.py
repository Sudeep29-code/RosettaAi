from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjavascript

from parsers.common import CommonAST


JS_LANGUAGE = Language(tsjavascript.language())
parser = Parser(JS_LANGUAGE)


def analyze_javascript_code(code):

    graph = CommonAST()

    tree = parser.parse(code.encode())

    root = tree.root_node

    process_node(root, graph, None)

    return graph.get_graph()


def process_node(node, graph, parent_id):

    node_type = node.type

    # Function
    if node_type in (
        "function_declaration",
        "function_expression",
        "arrow_function"
    ):

        function_name = find_function_name(node)

        function_id = graph.add_node(
            "FUNCTION",
            function_name
        )

        if parent_id is not None:
            graph.add_edge(parent_id, function_id)

        parent_id = function_id

    # Parameter
    elif node_type == "formal_parameters":

        for child in node.children:

            if child.type == "identifier":

                parameter_id = graph.add_node(
                    "PARAMETER",
                    child.text.decode()
                )

                if parent_id is not None:
                    graph.add_edge(
                        parent_id,
                        parameter_id
                    )

    # Return
    elif node_type == "return_statement":

        return_id = graph.add_node(
            "RETURN"
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                return_id
            )

        parent_id = return_id

    # If
    elif node_type == "if_statement":

        condition_id = graph.add_node(
            "CONDITION"
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                condition_id
            )

        parent_id = condition_id

    # Identifier
    elif node_type == "identifier":

        variable_id = graph.add_node(
            "VARIABLE",
            node.text.decode()
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                variable_id
            )

        return

    # Number
    elif node_type in (
        "number",
        "number_literal"
    ):

        constant_id = graph.add_node(
            "CONSTANT",
            node.text.decode()
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                constant_id
            )

        return

    # Binary expression
    elif node_type == "binary_expression":

        operation_id = graph.add_node(
            "OPERATION",
            node.text.decode()
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                operation_id
            )

        parent_id = operation_id

    # Function call
    elif node_type == "call_expression":

        call_id = graph.add_node(
            "FUNCTION_CALL"
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                call_id
            )

        parent_id = call_id

    # Process children
    for child in node.children:

        process_node(
            child,
            graph,
            parent_id
        )


def find_function_name(node):

    name_node = node.child_by_field_name("name")

    if name_node:

        return name_node.text.decode()

    return None


if __name__ == "__main__":

    code = """
    function factorial(n) {

        if (n === 0) {
            return 1;
        }

        return n * factorial(n - 1);
    }
    """

    result = analyze_javascript_code(code)

    print("\\nNODES:")

    for node in result["nodes"]:
        print(node)

    print("\\nEDGES:")

    for edge in result["edges"]:
        print(edge)