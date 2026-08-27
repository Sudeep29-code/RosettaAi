class ASTTranslator:

    # =========================================================
    # VALUE CONVERSION
    # =========================================================

    def convert_value(self, value, target):

        if value is None:
            return ""

        value = str(value)

        conversions = {
            "java": {
                "True": "true",
                "False": "false",
                "None": "null"
            },

            "javascript": {
                "True": "true",
                "False": "false",
                "None": "null"
            },

            "cpp": {
                "True": "true",
                "False": "false",
                "None": "nullptr"
            },

            "python": {
                "true": "True",
                "false": "False",
                "null": "None",
                "nullptr": "None"
            }
        }

        return conversions.get(target, {}).get(
            value,
            value
        )

    # =========================================================
    # EXPRESSION BUILDER
    # =========================================================

    def get_expression(
        self,
        node_id,
        node_map,
        children,
        target
    ):

        node = node_map.get(node_id)

        if not node:
            return ""

        node_type = node.get("type")
        value = node.get("value")

        # -----------------------------------------------------
        # VARIABLE
        # -----------------------------------------------------

        if node_type == "VARIABLE":

            return self.convert_value(
                value,
                target
            )

        # -----------------------------------------------------
        # CONSTANT
        # -----------------------------------------------------

        if node_type == "CONSTANT":

            return self.convert_value(
                value,
                target
            )

        # -----------------------------------------------------
        # OPERATION
        # -----------------------------------------------------

        if node_type == "OPERATION":

            operator_map = {
                "Add": "+",
                "Sub": "-",
                "Mult": "*",
                "Div": "/",
                "Mod": "%",
                "Eq": "==",
                "NotEq": "!=",
                "Lt": "<",
                "LtE": "<=",
                "Gt": ">",
                "GtE": ">=",
                "And": "&&",
                "Or": "||"
            }

            operator = operator_map.get(
                value,
                value
            )

            child_ids = children.get(
                node_id,
                []
            )

            expressions = []

            for child_id in child_ids:

                expression = self.get_expression(
                    child_id,
                    node_map,
                    children,
                    target
                )

                if expression:
                    expressions.append(expression)

            if len(expressions) >= 2:

                return (
                    f"{expressions[0]} "
                    f"{operator} "
                    f"{expressions[1]}"
                )

            return operator

        return ""

    # =========================================================
    # TRANSLATE GRAPH
    # =========================================================

    def translate_graph(
        self,
        graph,
        source,
        target
    ):

        source = source.lower()
        target = target.lower()

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # -----------------------------------------------------
        # NODE MAP
        # -----------------------------------------------------

        node_map = {
            node["id"]: node
            for node in nodes
        }

        # -----------------------------------------------------
        # CHILDREN MAP
        # -----------------------------------------------------

        children = {}

        for edge in edges:

            if isinstance(edge, (list, tuple)):

                parent = edge[0]
                child = edge[1]

            elif isinstance(edge, dict):

                parent = edge["source"]
                child = edge["target"]

            else:
                continue

            children.setdefault(
                parent,
                []
            ).append(child)

        # -----------------------------------------------------
        # FIND FUNCTION
        # -----------------------------------------------------

        function_node = None

        for node in nodes:

            if node.get("type") == "FUNCTION":

                function_node = node
                break

        output = []

        # Current indentation level
        indent = 0

        def emit(line=""):

            output.append(
                "    " * indent + line
            )

        # =====================================================
        # FUNCTION
        # =====================================================

        if function_node:

            function_name = function_node.get(
                "value",
                "function"
            )

            parameter_ids = children.get(
                function_node["id"],
                []
            )

            parameters = []

            for child_id in parameter_ids:

                child = node_map.get(child_id)

                if not child:
                    continue

                if child.get("type") == "PARAMETER":

                    parameters.append(
                        child.get("value", "")
                    )

            # ---------------------------------------------
            # JAVA
            # ---------------------------------------------

            if target == "java":

                typed_parameters = []

                for parameter in parameters:

                    typed_parameters.append(
                        f"Object {parameter}"
                    )

                params = ", ".join(
                    typed_parameters
                )

                emit(
                    f"public static Object "
                    f"{function_name}({params}) {{"
                )

            # ---------------------------------------------
            # JAVASCRIPT
            # ---------------------------------------------

            elif target == "javascript":

                params = ", ".join(parameters)

                emit(
                    f"function {function_name}({params}) {{"
                )

            # ---------------------------------------------
            # C++
            # ---------------------------------------------

            elif target == "cpp":

                typed_parameters = []

                for parameter in parameters:

                    typed_parameters.append(
                        f"auto {parameter}"
                    )

                params = ", ".join(
                    typed_parameters
                )

                emit(
                    f"auto {function_name}({params}) {{"
                )

            # ---------------------------------------------
            # PYTHON
            # ---------------------------------------------

            else:

                params = ", ".join(parameters)

                emit(
                    f"def {function_name}({params}):"
                )

            indent += 1

        # =====================================================
        # PROCESS TOP-LEVEL FUNCTION CHILDREN
        # =====================================================

        if function_node:

            function_children = children.get(
                function_node["id"],
                []
            )

            for child_id in function_children:

                child = node_map.get(child_id)

                if not child:
                    continue

                child_type = child.get("type")

                # Parameters are already handled
                if child_type == "PARAMETER":
                    continue

                # -------------------------------------------------
                # CONDITION
                # -------------------------------------------------

                if child_type == "CONDITION":

                    condition_children = children.get(
                        child_id,
                        []
                    )

                    condition_expression = ""

                    # Find the operation/expression
                    for condition_child_id in condition_children:

                        condition_child = node_map.get(
                            condition_child_id
                        )

                        if not condition_child:
                            continue

                        condition_type = condition_child.get(
                            "type"
                        )

                        if condition_type in (
                            "OPERATION",
                            "VARIABLE",
                            "CONSTANT"
                        ):

                            condition_expression = (
                                self.get_expression(
                                    condition_child_id,
                                    node_map,
                                    children,
                                    target
                                )
                            )

                            break

                    emit(
                        f"if ({condition_expression}) {{"
                    )

                    indent += 1

                    # ---------------------------------------------
                    # Process condition body
                    # ---------------------------------------------

                    for condition_body_id in condition_children:

                        body_node = node_map.get(
                            condition_body_id
                        )

                        if not body_node:
                            continue

                        body_type = body_node.get(
                            "type"
                        )

                        # Skip expression nodes
                        if body_type in (
                            "OPERATION",
                            "VARIABLE",
                            "CONSTANT"
                        ):
                            continue

                        # -----------------------------------------
                        # RETURN INSIDE IF
                        # -----------------------------------------

                        if body_type == "RETURN":

                            return_children = children.get(
                                condition_body_id,
                                []
                            )

                            if return_children:

                                expression = self.get_expression(
                                    return_children[0],
                                    node_map,
                                    children,
                                    target
                                )

                                emit(
                                    f"return {expression};"
                                )

                            else:

                                emit("return;")

                    indent -= 1

                    emit("}")

                    continue

                # -------------------------------------------------
                # ASSIGNMENT
                # -------------------------------------------------

                if child_type == "ASSIGNMENT":

                    assignment_children = children.get(
                        child_id,
                        []
                    )

                    variable_name = None
                    expression = None

                    for assignment_child_id in assignment_children:

                        assignment_child = node_map.get(
                            assignment_child_id
                        )

                        if not assignment_child:
                            continue

                        assignment_type = (
                            assignment_child.get("type")
                        )

                        if assignment_type == "VARIABLE":

                            if variable_name is None:

                                variable_name = (
                                    assignment_child.get(
                                        "value"
                                    )
                                )

                            else:

                                expression = (
                                    self.get_expression(
                                        assignment_child_id,
                                        node_map,
                                        children,
                                        target
                                    )
                                )

                        elif assignment_type in (
                            "OPERATION",
                            "CONSTANT"
                        ):

                            expression = (
                                self.get_expression(
                                    assignment_child_id,
                                    node_map,
                                    children,
                                    target
                                )
                            )

                    if variable_name and expression:

                        if target == "java":

                            emit(
                                f"Object {variable_name} "
                                f"= {expression};"
                            )

                        elif target == "javascript":

                            emit(
                                f"let {variable_name} "
                                f"= {expression};"
                            )

                        elif target == "cpp":

                            emit(
                                f"auto {variable_name} "
                                f"= {expression};"
                            )

                        else:

                            emit(
                                f"{variable_name} "
                                f"= {expression}"
                            )

                    continue

                # -------------------------------------------------
                # RETURN
                # -------------------------------------------------

                if child_type == "RETURN":

                    return_children = children.get(
                        child_id,
                        []
                    )

                    if return_children:

                        expression = self.get_expression(
                            return_children[0],
                            node_map,
                            children,
                            target
                        )

                        emit(
                            f"return {expression};"
                        )

                    else:

                        emit("return;")

                    continue

                # -------------------------------------------------
                # FUNCTION CALL
                # -------------------------------------------------

                if child_type == "FUNCTION_CALL":

                    call_name = child.get(
                        "value"
                    )

                    call_children = children.get(
                        child_id,
                        []
                    )

                    arguments = []

                    for argument_id in call_children:

                        argument_node = node_map.get(
                            argument_id
                        )

                        if not argument_node:
                            continue

                        argument_type = (
                            argument_node.get("type")
                        )

                        argument_value = (
                            argument_node.get("value")
                        )

                        # Don't include function-name variable
                        if (
                            argument_type == "VARIABLE"
                            and argument_value == call_name
                        ):
                            continue

                        expression = (
                            self.get_expression(
                                argument_id,
                                node_map,
                                children,
                                target
                            )
                        )

                        if expression:

                            arguments.append(
                                expression
                            )

                    args = ", ".join(arguments)

                    if (
                        source == "python"
                        and call_name == "print"
                    ):

                        if target == "java":

                            emit(
                                f"System.out.println({args});"
                            )

                        elif target == "javascript":

                            emit(
                                f"console.log({args});"
                            )

                        elif target == "cpp":

                            emit(
                                f"cout << {args};"
                            )

                        else:

                            emit(
                                f"print({args})"
                            )

                    else:

                        emit(
                            f"{call_name}({args});"
                        )

        # =====================================================
        # CLOSE FUNCTION
        # =====================================================

        if function_node:

            indent -= 1

            emit("}")

        return "\n".join(output)