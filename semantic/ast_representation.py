import ast


class ASTRepresentation(ast.NodeVisitor):

    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node_type, value=None):
        node_id = len(self.nodes)

        self.nodes.append({
            "id": node_id,
            "type": node_type,
            "value": value
        })

        return node_id

    def visit_FunctionDef(self, node):
        function_id = self.add_node(
            "FUNCTION",
            node.name
        )

        for arg in node.args.args:
            parameter_id = self.add_node(
                "PARAMETER",
                arg.arg
            )

            self.edges.append(
                (function_id, parameter_id)
            )

        for statement in node.body:
            statement_id = self.process_statement(statement)

            if statement_id is not None:
                self.edges.append(
                    (function_id, statement_id)
                )

    def process_statement(self, node):

        if isinstance(node, ast.If):

            if_id = self.add_node("CONDITION")

            condition = self.process_expression(node.test)

            if condition is not None:
                self.edges.append(
                    (if_id, condition)
                )

            for statement in node.body:
                statement_id = self.process_statement(statement)

                if statement_id is not None:
                    self.edges.append(
                        (if_id, statement_id)
                    )

            return if_id

        elif isinstance(node, ast.Return):

            return_id = self.add_node("RETURN")

            expression = self.process_expression(node.value)

            if expression is not None:
                self.edges.append(
                    (return_id, expression)
                )

            return return_id

        return None

    def process_expression(self, node):

        if isinstance(node, ast.Constant):

            return self.add_node(
                "CONSTANT",
                str(node.value)
            )

        elif isinstance(node, ast.Name):

            return self.add_node(
                "VARIABLE",
                node.id
            )

        elif isinstance(node, ast.BinOp):

            operator = type(node.op).__name__

            operation_id = self.add_node(
                "OPERATION",
                operator
            )

            left = self.process_expression(node.left)
            right = self.process_expression(node.right)

            if left is not None:
                self.edges.append(
                    (operation_id, left)
                )

            if right is not None:
                self.edges.append(
                    (operation_id, right)
                )

            return operation_id

        elif isinstance(node, ast.Call):

            function_name = None

            if isinstance(node.func, ast.Name):
                function_name = node.func.id

            call_id = self.add_node(
                "FUNCTION_CALL",
                function_name
            )

            for argument in node.args:

                argument_id = self.process_expression(argument)

                if argument_id is not None:
                    self.edges.append(
                        (call_id, argument_id)
                    )

            return call_id

        elif isinstance(node, ast.Compare):

            compare_id = self.add_node(
                "COMPARISON"
            )

            left = self.process_expression(node.left)

            if left is not None:
                self.edges.append(
                    (compare_id, left)
                )

            for comparator in node.comparators:

                right = self.process_expression(comparator)

                if right is not None:
                    self.edges.append(
                        (compare_id, right)
                    )

            return compare_id

        return None


def analyze_code(code):

    tree = ast.parse(code)

    analyzer = ASTRepresentation()

    analyzer.visit(tree)

    return {
        "nodes": analyzer.nodes,
        "edges": analyzer.edges
    }


if __name__ == "__main__":

    code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

    result = analyze_code(code)

    print("\nNODES:")

    for node in result["nodes"]:
        print(node)

    print("\nEDGES:")

    for edge in result["edges"]:
        print(edge)