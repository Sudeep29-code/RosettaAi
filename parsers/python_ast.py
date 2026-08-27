import ast

from parsers.common import CommonAST


def analyze_code(code):

    try:
        tree = ast.parse(code)

    except SyntaxError as e:
        raise ValueError(
            f"Invalid Python code: {e}"
        )

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    non_functions = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]

    if not functions and non_functions:
        # Pure top-level script
        clean_lines = []
        for line in code.split("\n"):
            clean_lines.append("    " + line if line.strip() else "")
        wrapped_code = "def main():\n" + "\n".join(clean_lines)
        try:
            tree = ast.parse(wrapped_code)
        except Exception:
            pass
    elif functions and non_functions:
        # Mixed script (functions + top-level test statements)
        main_def = ast.FunctionDef(
            name="main",
            args=ast.arguments(
                posonlyargs=[],
                args=[],
                vararg=None,
                kwonlyargs=[],
                kw_defaults=[],
                kwarg=None,
                defaults=[]
            ),
            body=non_functions,
            decorator_list=[]
        )
        tree.body = functions + [main_def]

    graph = CommonAST()

    for node in tree.body:

        process_node(
            node,
            graph
        )

    return graph.get_graph()


def process_node(
    node,
    graph,
    parent_id=None
):

    # =====================================================
    # FUNCTION
    # =====================================================

    if isinstance(node, ast.FunctionDef):

        function_id = graph.add_node(
            "FUNCTION",
            node.name
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                function_id
            )

        for arg in node.args.args:

            parameter_id = graph.add_node(
                "PARAMETER",
                arg.arg
            )

            graph.add_edge(
                function_id,
                parameter_id
            )

        for child in node.body:

            process_node(
                child,
                graph,
                function_id
            )

        return


    # =====================================================
    # RETURN
    # =====================================================

    if isinstance(node, ast.Return):

        return_id = graph.add_node(
            "RETURN"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                return_id
            )

        if node.value is not None:

            process_node(
                node.value,
                graph,
                return_id
            )

        return


    # =====================================================
    # ASSIGNMENT
    # =====================================================

    if isinstance(node, ast.Assign):

        # Handle tuple unpack e.g. a, b = 0, 1 or a, b = b, a + b
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            target_tuple = node.targets[0]
            val_tuple = node.value if isinstance(node.value, ast.Tuple) else None

            if val_tuple and len(target_tuple.elts) == len(val_tuple.elts):
                target_names = [e.id for e in target_tuple.elts if isinstance(e, ast.Name)]

                # Check if first target appears in subsequent expressions (simultaneous swap)
                needs_temp = False
                if len(target_names) >= 2:
                    first_name = target_names[0]
                    for later_val in val_tuple.elts[1:]:
                        for sub in ast.walk(later_val):
                            if isinstance(sub, ast.Name) and sub.id == first_name:
                                needs_temp = True
                                break

                if needs_temp:
                    # 1. temp = first_target
                    temp_assign = graph.add_node("ASSIGNMENT")
                    if parent_id is not None:
                        graph.add_edge(parent_id, temp_assign)
                    temp_var = graph.add_node("VARIABLE", "temp")
                    graph.add_edge(temp_assign, temp_var)
                    process_node(target_tuple.elts[0], graph, temp_assign)

                    # 2. first_target = first_val
                    first_assign = graph.add_node("ASSIGNMENT")
                    if parent_id is not None:
                        graph.add_edge(parent_id, first_assign)
                    process_node(target_tuple.elts[0], graph, first_assign)
                    process_node(val_tuple.elts[0], graph, first_assign)

                    # 3. subsequent targets using temp
                    for t, v in zip(target_tuple.elts[1:], val_tuple.elts[1:]):
                        sub_assign = graph.add_node("ASSIGNMENT")
                        if parent_id is not None:
                            graph.add_edge(parent_id, sub_assign)
                        process_node(t, graph, sub_assign)

                        # Check if v uses first_name
                        if isinstance(v, ast.BinOp) and isinstance(v.left, ast.Name) and v.left.id == target_names[0]:
                            op_id = graph.add_node("OPERATION", type(v.op).__name__)
                            graph.add_edge(sub_assign, op_id)
                            temp_ref = graph.add_node("VARIABLE", "temp")
                            graph.add_edge(op_id, temp_ref)
                            process_node(v.right, graph, op_id)
                        else:
                            process_node(v, graph, sub_assign)
                    return
                else:
                    # Independent assignments
                    for t, v in zip(target_tuple.elts, val_tuple.elts):
                        sub_assign = graph.add_node("ASSIGNMENT")
                        if parent_id is not None:
                            graph.add_edge(parent_id, sub_assign)
                        process_node(t, graph, sub_assign)
                        process_node(v, graph, sub_assign)
                    return

        assignment_id = graph.add_node(
            "ASSIGNMENT"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                assignment_id
            )

        for target in node.targets:

            process_node(
                target,
                graph,
                assignment_id
            )

        process_node(
            node.value,
            graph,
            assignment_id
        )

        return


    # =====================================================
    # VARIABLE
    # =====================================================

    if isinstance(node, ast.Name):

        variable_id = graph.add_node(
            "VARIABLE",
            node.id
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                variable_id
            )

        return


    # =====================================================
    # CONSTANT
    # IMPORTANT: preserve value_type
    # =====================================================

    if isinstance(node, ast.Constant):

        value = node.value

        if isinstance(value, bool):

            value_type = "BOOLEAN"
            stored_value = str(value)

        elif isinstance(value, str):

            value_type = "STRING"
            stored_value = value

        elif isinstance(value, int):

            value_type = "INTEGER"
            stored_value = str(value)

        elif isinstance(value, float):

            value_type = "FLOAT"
            stored_value = str(value)

        elif value is None:

            value_type = "NULL"
            stored_value = "None"

        else:

            value_type = "UNKNOWN"
            stored_value = str(value)

        constant_id = graph.add_node(
            "CONSTANT",
            stored_value
        )

        # Store the type information
        graph.nodes[constant_id][
            "value_type"
        ] = value_type

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                constant_id
            )

        return


    # =====================================================
    # BINARY OPERATION
    # a + b
    # a - b
    # =====================================================

    if isinstance(node, ast.BinOp):

        operation_id = graph.add_node(
            "OPERATION",
            type(node.op).__name__
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                operation_id
            )

        process_node(
            node.left,
            graph,
            operation_id
        )

        process_node(
            node.right,
            graph,
            operation_id
        )

        return


    # =====================================================
    # UNARY OPERATION
    # =====================================================

    if isinstance(node, ast.UnaryOp):

        operation_id = graph.add_node(
            "OPERATION",
            type(node.op).__name__
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                operation_id
            )

        process_node(
            node.operand,
            graph,
            operation_id
        )

        return


    # =====================================================
    # COMPARISON
    # =====================================================

    if isinstance(node, ast.Compare):

        operation = "Compare"

        if node.ops:

            operation = type(
                node.ops[0]
            ).__name__

        operation_id = graph.add_node(
            "OPERATION",
            operation
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                operation_id
            )

        process_node(
            node.left,
            graph,
            operation_id
        )

        for comparator in node.comparators:

            process_node(
                comparator,
                graph,
                operation_id
            )

        return


    # =====================================================
    # BOOLEAN OPERATION
    # =====================================================

    if isinstance(node, ast.BoolOp):

        operation_id = graph.add_node(
            "OPERATION",
            type(node.op).__name__
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                operation_id
            )

        for value in node.values:

            process_node(
                value,
                graph,
                operation_id
            )

        return


    # =====================================================
    # IF
    # =====================================================

    if isinstance(node, ast.If):

        condition_id = graph.add_node(
            "CONDITION",
            "if"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                condition_id
            )

        process_node(
            node.test,
            graph,
            condition_id
        )

        for child in node.body:

            process_node(
                child,
                graph,
                condition_id
            )

        if node.orelse:

            else_id = graph.add_node(
                "ELSE",
                "else"
            )

            graph.add_edge(
                condition_id,
                else_id
            )

            for child in node.orelse:

                process_node(
                    child,
                    graph,
                    else_id
                )

        return


    # =====================================================
    # FOR LOOP
    # =====================================================

    if isinstance(node, ast.For):

        loop_id = graph.add_node(
            "LOOP",
            "for"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                loop_id
            )

        process_node(
            node.target,
            graph,
            loop_id
        )

        process_node(
            node.iter,
            graph,
            loop_id
        )

        for child in node.body:

            process_node(
                child,
                graph,
                loop_id
            )

        return


    # =====================================================
    # WHILE LOOP
    # =====================================================

    if isinstance(node, ast.While):

        loop_id = graph.add_node(
            "LOOP",
            "while"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                loop_id
            )

        process_node(
            node.test,
            graph,
            loop_id
        )

        for child in node.body:

            process_node(
                child,
                graph,
                loop_id
            )

        return


    # =====================================================
    # EXPRESSION
    # =====================================================

    if isinstance(node, ast.Expr):

        process_node(
            node.value,
            graph,
            parent_id
        )

        return


    # =====================================================
    # FUNCTION CALL & METHOD CALL
    # =====================================================

    if isinstance(node, ast.Call):

        if isinstance(node.func, ast.Attribute):
            # Method call: receiver.method(args...)
            method_id = graph.add_node(
                "METHOD_CALL",
                node.func.attr
            )

            if parent_id is not None:
                graph.add_edge(parent_id, method_id)

            # Child 0 is receiver object (e.g. text, text.lower(), frequency)
            process_node(
                node.func.value,
                graph,
                method_id
            )

            # Subsequent children are arguments
            for argument in node.args:
                process_node(
                    argument,
                    graph,
                    method_id
                )

            return

        call_name = "function"

        if isinstance(node.func, ast.Name):
            call_name = node.func.id

        call_id = graph.add_node(
            "FUNCTION_CALL",
            call_name
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                call_id
            )

        for argument in node.args:
            process_node(
                argument,
                graph,
                call_id
            )

        return

    # =====================================================
    # SUBSCRIPT (e.g. frequency[word])
    # =====================================================

    if isinstance(node, ast.Subscript):

        subscript_id = graph.add_node(
            "SUBSCRIPT",
            "subscript"
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                subscript_id
            )

        # Child 0: collection/dict
        process_node(
            node.value,
            graph,
            subscript_id
        )

        # Child 1: key/index
        process_node(
            node.slice,
            graph,
            subscript_id
        )

        return

    # =====================================================
    # DICTIONARY (e.g. {})
    # =====================================================

    if isinstance(node, ast.Dict):

        dict_id = graph.add_node(
            "DICTIONARY",
            "dict"
        )

        if parent_id is not None:
            graph.add_edge(
                parent_id,
                dict_id
            )

        for k, v in zip(node.keys, node.values):
            if k is not None:
                entry_id = graph.add_node("DICT_ENTRY", "entry")
                graph.add_edge(dict_id, entry_id)
                process_node(k, graph, entry_id)
                process_node(v, graph, entry_id)

        return

    # =====================================================
    # ATTRIBUTE
    # =====================================================

    if isinstance(node, ast.Attribute):

        attribute_id = graph.add_node(
            "VARIABLE",
            node.attr
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                attribute_id
            )

        return


    # =====================================================
    # BREAK & CONTINUE
    # =====================================================

    if isinstance(node, ast.Break):

        break_id = graph.add_node(
            "BREAK",
            "break"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                break_id
            )

        return

    if isinstance(node, ast.Continue):

        continue_id = graph.add_node(
            "CONTINUE",
            "continue"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                continue_id
            )

        return

    # =====================================================
    # LIST
    # =====================================================

    if isinstance(node, ast.List):

        list_id = graph.add_node(
            "LIST",
            "list"
        )

        if parent_id is not None:

            graph.add_edge(
                parent_id,
                list_id
            )

        for element in node.elts:

            process_node(
                element,
                graph,
                list_id
            )

        return


    # =====================================================
    # UNKNOWN
    # =====================================================

    unknown_id = graph.add_node(
        "UNKNOWN",
        type(node).__name__
    )

    if parent_id is not None:

        graph.add_edge(
            parent_id,
            unknown_id
        )