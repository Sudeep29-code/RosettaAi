import javalang
from parsers.common import CommonAST


def analyze_java_code(code):

    # javalang expects a complete Java class.
    # Our dataset may contain only a method.
    if "class " not in code:
        code = f"""
        class RosettaTemp {{
            {code}
        }}
        """

    tree = javalang.parse.parse(code)

    graph = CommonAST()

    for _, node in tree:

        if isinstance(node, javalang.tree.MethodDeclaration):

            function_id = graph.add_node(
                "FUNCTION",
                node.name
            )

            for parameter in node.parameters:

                parameter_id = graph.add_node(
                    "PARAMETER",
                    parameter.name
                )

                graph.add_edge(
                    function_id,
                    parameter_id
                )

            if node.body:

                for statement in node.body:

                    process_statement(
                        statement,
                        graph,
                        function_id
                    )

    return graph.get_graph()


def process_statement(statement, graph, parent_id):

    # IF statement
    if isinstance(statement, javalang.tree.IfStatement):

        condition_id = graph.add_node(
            "CONDITION"
        )

        graph.add_edge(
            parent_id,
            condition_id
        )

        process_expression(
            statement.condition,
            graph,
            condition_id
        )

        # Then statement
        process_statement(
            statement.then_statement,
            graph,
            condition_id
        )

        return condition_id

    # Block statement
    elif isinstance(statement, javalang.tree.BlockStatement):

        if statement.statements:

            for child in statement.statements:

                process_statement(
                    child,
                    graph,
                    parent_id
                )

    # Return statement
    elif isinstance(statement, javalang.tree.ReturnStatement):

        return_id = graph.add_node(
            "RETURN"
        )

        graph.add_edge(
            parent_id,
            return_id
        )

        if statement.expression:

            process_expression(
                statement.expression,
                graph,
                return_id
            )

        return return_id


def process_expression(expression, graph, parent_id):

    # Variable
    if isinstance(expression, javalang.tree.MemberReference):

        variable_id = graph.add_node(
            "VARIABLE",
            expression.member
        )

        graph.add_edge(
            parent_id,
            variable_id
        )

    # Literal
    elif isinstance(expression, javalang.tree.Literal):

        constant_id = graph.add_node(
            "CONSTANT",
            expression.value
        )

        graph.add_edge(
            parent_id,
            constant_id
        )

    # Binary operation
    elif isinstance(expression, javalang.tree.BinaryOperation):

        operation_id = graph.add_node(
            "OPERATION",
            expression.operator
        )

        graph.add_edge(
            parent_id,
            operation_id
        )

        process_expression(
            expression.operandl,
            graph,
            operation_id
        )

        process_expression(
            expression.operandr,
            graph,
            operation_id
        )

    # Method/function call
    elif isinstance(expression, javalang.tree.MethodInvocation):

        call_id = graph.add_node(
            "FUNCTION_CALL",
            expression.member
        )

        graph.add_edge(
            parent_id,
            call_id
        )

        for argument in expression.arguments:

            process_expression(
                argument,
                graph,
                call_id
            )


if __name__ == "__main__":

    code = """
    class Main {

        static int factorial(int n) {

            if (n == 0) {
                return 1;
            }

            return n * factorial(n - 1);
        }
    }
    """

    result = analyze_java_code(code)

    print("\\nNODES:")

    for node in result["nodes"]:
        print(node)

    print("\\nEDGES:")

    for edge in result["edges"]:
        print(edge)