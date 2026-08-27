class ASTTranslator:
    """
    Graph-based source-code translator.

    Supported targets:
        - Java
        - C++
        - JavaScript
        - Python

    Supported graph nodes:
        - FUNCTION
        - PARAMETER
        - VARIABLE
        - CONSTANT
        - ASSIGNMENT
        - OPERATION
        - CONDITION
        - IF
        - ELSE
        - LOOP
        - RETURN
        - FUNCTION_CALL
        - UNKNOWN (list)
    """

    def __init__(self):
        self.target = "java"
        self.indent_level = 0

        self.variables = {}
        self.parameters = {}
        self.declared_variables = set()

        self.function_return_type = "void"

    # ============================================================
    # PUBLIC API
    # ============================================================

    def translate_graph(
        self,
        graph,
        source="python",
        target="java"
    ):
        """
        Translate a parser graph into the requested target language.
        """

        if not isinstance(graph, dict):
            return ""

        source = (source or "python").lower()
        target = (target or "java").lower()

        # --------------------------------------------------------
        # Same language
        # --------------------------------------------------------

        if source == target:
            return graph.get("source_code", "")

        self.target = target

        self.indent_level = 0
        self.variables = {}
        self.parameters = {}
        self.declared_variables = set()
        self.function_return_type = "void"

        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not nodes:
            return ""

        node_map = {
            node.get("id"): node
            for node in nodes
            if isinstance(node, dict)
        }

        # --------------------------------------------------------
        # Build children map
        # --------------------------------------------------------

        children = {}

        for edge in edges:
            if not isinstance(edge, dict):
                continue

            source_id = edge.get("source")
            target_id = edge.get("target")

            children.setdefault(
                source_id,
                []
            ).append(target_id)

        function_nodes = [
            node
            for node in nodes
            if node.get("type") == "FUNCTION"
        ]

        if not function_nodes:
            return ""

        # Translate all functions
        translated_functions = []
        for fn_node in function_nodes:
            fn_code = self._translate_single_function(
                fn_node,
                nodes,
                edges,
                node_map,
                children
            )
            if fn_code.strip():
                translated_functions.append(fn_code)

        if not translated_functions:
            return ""

        if self.target == "java":
            # Check if functions already have class wrapper
            all_bodies = "\n\n".join(translated_functions)
            needs_map_import = "Map<" in all_bodies or "HashMap" in all_bodies
            needs_arrays_import = "Arrays." in all_bodies or "String[]" in all_bodies
            needs_list_import = "List<" in all_bodies

            imports = []
            if needs_map_import:
                imports.extend(["import java.util.Map;", "import java.util.HashMap;"])
            if needs_arrays_import:
                imports.append("import java.util.Arrays;")
            if needs_list_import:
                imports.append("import java.util.List;")

            import_str = ("\n".join(sorted(set(imports))) + "\n\n") if imports else ""

            indented_fns = []
            for fn in translated_functions:
                indented_fns.append("\n".join("    " + line if line.strip() else "" for line in fn.split("\n")))
            class_body = "\n\n".join(indented_fns)

            return f"{import_str}public class Solution {{\n{class_body}\n}}"

        elif self.target == "cpp":
            # Put helper functions first, main at the bottom
            helper_fns = [fn for fn in translated_functions if not fn.startswith("int main")]
            main_fns = [fn for fn in translated_functions if fn.startswith("int main")]
            combined = helper_fns + main_fns
            all_cpp = "\n\n".join(combined)

            headers = ["#include <iostream>"]
            if "string" in all_cpp:
                headers.append("#include <string>")
            if "vector" in all_cpp:
                headers.append("#include <vector>")
            if "unordered_map" in all_cpp or "map" in all_cpp:
                headers.append("#include <unordered_map>")
            if "stringstream" in all_cpp or "ss" in all_cpp or "split_words" in all_cpp:
                headers.append("#include <sstream>")
            if "transform" in all_cpp or "tolower" in all_cpp or "to_lower" in all_cpp:
                headers.append("#include <algorithm>")
            if "pow(" in all_cpp:
                headers.append("#include <cmath>")

            headers.append("using namespace std;\n")

            helper_definitions = []
            if "to_lower(" in all_cpp:
                helper_definitions.append(
                    "string to_lower(string s) {\n    for (char &c : s) c = tolower(c);\n    return s;\n}"
                )
            if "split_words(" in all_cpp:
                helper_definitions.append(
                    "vector<string> split_words(const string& s) {\n    vector<string> words;\n    stringstream ss(s);\n    string word;\n    while (ss >> word) {\n        words.push_back(word);\n    }\n    return words;\n}"
                )

            helper_str = ("\n\n".join(helper_definitions) + "\n\n") if helper_definitions else ""
            return "\n".join(headers) + "\n" + helper_str + all_cpp

        elif self.target == "javascript":
            # If main function exists, unpack statements to run at top level
            js_code_parts = []
            for fn_node, fn_str in zip(function_nodes, translated_functions):
                if fn_node.get("value") == "main":
                    # Unpack main body lines
                    main_lines = fn_str.split("\n")[1:-1]
                    unindented = [l[4:] if l.startswith("    ") else l for l in main_lines]
                    js_code_parts.append("\n".join(unindented))
                else:
                    js_code_parts.append(fn_str)

            return "\n\n".join(js_code_parts)

        return "\n\n".join(translated_functions)

    def _translate_single_function(
        self,
        function_node,
        nodes,
        edges,
        node_map,
        children
    ):
        self.indent_level = 0
        self.variables = {}
        self.parameters = {}
        self.declared_variables = set()
        self.function_return_type = "void"

        function_id = function_node.get("id")
        function_name = function_node.get("value") or "translated_function"

        # ========================================================
        # PARAMETERS
        # ========================================================

        parameter_nodes = []

        for child_id in children.get(function_id, []):

            child = node_map.get(child_id)

            if not child:
                continue

            if child.get("type") == "PARAMETER":
                parameter_nodes.append(child)

        for parameter in parameter_nodes:

            name = parameter.get("value")

            if name:
                self.parameters[name] = "unknown"

        # ========================================================
        # TYPE ANALYSIS
        # ========================================================

        self._infer_variables(
            nodes,
            children,
            node_map
        )

        self._infer_parameter_types(
            parameter_nodes,
            nodes,
            children,
            node_map
        )

        self.function_return_type = (
            self._infer_function_return_type(
                nodes,
                children,
                node_map
            )
        )

        # ========================================================
        # FUNCTION HEADER
        # ========================================================

        params = []

        for parameter in parameter_nodes:

            name = parameter.get("value")

            params.append(
                self._parameter_declaration(name)
            )

        return_type = self._function_return_type()

        header = self._function_header(
            function_name,
            params,
            return_type
        )

        # ========================================================
        # BODY
        # ========================================================

        body_nodes = []

        for child_id in children.get(
            function_id,
            []
        ):

            child = node_map.get(child_id)

            if not child:
                continue

            if child.get("type") == "PARAMETER":
                continue

            body_nodes.append(child_id)

        lines = [
            header + " {"
        ]

        self.indent_level = 1

        visited = set()

        for node_id in body_nodes:

            if node_id in visited:
                continue

            generated = self._translate_statement(
                node_id,
                node_map,
                children,
                visited,
                inside_loop=False
            )

            if generated:
                lines.extend(generated)

        self.indent_level = 0

        lines.append("}")

        return "\n".join(lines)

    # ============================================================
    # TYPE INFERENCE
    # ============================================================

    def _infer_variables(
        self,
        nodes,
        children,
        node_map
    ):
        """
        Infer types of variables created by assignments.
        """

        for node in nodes:

            if node.get("type") != "ASSIGNMENT":
                continue

            child_ids = children.get(
                node.get("id"),
                []
            )

            if len(child_ids) >= 2:
                variable_node = node_map.get(child_ids[0])
                expression_node = node_map.get(child_ids[1])
            elif len(child_ids) == 1:
                variable_node = node_map.get(child_ids[0])
                expression_node = None
            else:
                continue

            if not variable_node:
                continue

            name = variable_node.get("value")

            if not name:
                continue

            if expression_node:

                inferred = self._infer_node_type(
                    expression_node,
                    node_map,
                    children
                )

            else:
                inferred = "unknown"

            self.variables[name] = inferred

    # ============================================================

    def _infer_parameter_types(
        self,
        parameter_nodes,
        nodes,
        children,
        node_map
    ):
        """
        Infer parameter types.

        Example:

            def calculate(numbers):
                for x in numbers:

        numbers -> list
        """

        for parameter in parameter_nodes:

            name = parameter.get("value")

            if not name:
                continue

            inferred = "unknown"

            # ----------------------------------------------------
            # Look for usage in loops
            # ----------------------------------------------------

            for node in nodes:

                if node.get("type") != "LOOP":
                    continue

                loop_children = [
                    node_map[c]
                    for c in children.get(
                        node.get("id"),
                        []
                    )
                    if c in node_map
                ]

                variable_nodes = [
                    child
                    for child in loop_children
                    if child.get("type") == "VARIABLE"
                ]

                # Example:
                # LOOP
                #   VARIABLE x
                #   VARIABLE numbers
                #
                # The second variable is the iterable.

                for index, variable in enumerate(
                    variable_nodes
                ):

                    if variable.get("value") != name:
                        continue

                    if index >= 1:
                        inferred = "list"

            # ----------------------------------------------------
            # Look for list marker
            # ----------------------------------------------------

            for node in nodes:

                if node.get("type") != "UNKNOWN":
                    continue

                if node.get("value") != "list":
                    continue

                for child_id in children.get(
                    node.get("id"),
                    []
                ):

                    child = node_map.get(child_id)

                    if (
                        child
                        and child.get("value") == name
                    ):
                        inferred = "list"

            # Check if parameter has string methods called on it or is named text/str/word
            if name in ("text", "s", "str", "string", "line", "sentence", "word"):
                inferred = "string"

            for node in nodes:
                if node.get("type") == "METHOD_CALL":
                    child_ids = children.get(node.get("id"), [])
                    if child_ids:
                        receiver = node_map.get(child_ids[0])
                        if receiver and receiver.get("value") == name:
                            if node.get("value") in ("lower", "upper", "split", "strip", "trim", "replace", "charAt", "indexOf"):
                                inferred = "string"
                            elif node.get("value") in ("get", "keys", "values", "items"):
                                inferred = "dict"
                            elif node.get("value") in ("append", "pop", "insert", "sort"):
                                inferred = "list"

            self.parameters[name] = inferred

    # ============================================================

    def _infer_node_type(
        self,
        node,
        node_map,
        children
    ):
        """
        Infer semantic type of a graph node.
        """

        if not node:
            return "unknown"

        node_type = node.get("type")
        value = node.get("value")

        # --------------------------------------------------------
        # DICTIONARY
        # --------------------------------------------------------

        if node_type == "DICTIONARY":
            return "dict"

        # --------------------------------------------------------
        # METHOD CALL
        # --------------------------------------------------------

        if node_type == "METHOD_CALL":
            if value == "split":
                return "string[]"
            if value in ("lower", "upper", "strip", "trim", "replace"):
                return "string"
            if value in ("get", "count", "length", "size"):
                return "int"
            if value == "append":
                return "void"

        # --------------------------------------------------------
        # SUBSCRIPT
        # --------------------------------------------------------

        if node_type == "SUBSCRIPT":
            return "int"

        # --------------------------------------------------------
        # CONSTANT
        # --------------------------------------------------------

        if node_type == "CONSTANT":

            value_type = str(
                node.get("value_type", "")
            ).upper()

            if value_type in (
                "INTEGER",
                "INT"
            ):
                return "int"

            if value_type in (
                "FLOAT",
                "DOUBLE"
            ):
                return "double"

            if value_type in (
                "STRING",
                "STR"
            ):
                return "string"

            if value_type in (
                "BOOLEAN",
                "BOOL"
            ):
                return "boolean"

            # Fallback based on actual value
            if isinstance(value, bool):
                return "boolean"

            if isinstance(value, int):
                return "int"

            if isinstance(value, float):
                return "double"

            if isinstance(value, str):
                return "string"

        # --------------------------------------------------------
        # VARIABLE
        # --------------------------------------------------------

        if node_type == "VARIABLE":

            name = node.get("value")

            if name in self.variables:
                return self.variables[name]

            if name in self.parameters:
                return self.parameters[name]

            return "unknown"

        # --------------------------------------------------------
        # PARAMETER
        # --------------------------------------------------------

        if node_type == "PARAMETER":

            name = node.get("value")

            return self.parameters.get(
                name,
                "unknown"
            )

        # --------------------------------------------------------
        # LIST
        # --------------------------------------------------------

        if node_type == "UNKNOWN":

            if value == "list":
                return "list"

        # --------------------------------------------------------
        # OPERATION
        # --------------------------------------------------------

        if node_type == "OPERATION":

            # Comparisons
            if value in (
                "Gt",
                "GtE",
                "Lt",
                "LtE",
                "Eq",
                "NotEq",
                "Is",
                "IsNot",
                "In",
                "NotIn"
            ):
                return "boolean"

            # Arithmetic
            if value in (
                "Add",
                "Sub",
                "Mult",
                "Div",
                "Mod"
            ):

                child_ids = children.get(
                    node.get("id"),
                    []
                )

                child_types = []

                for child_id in child_ids:

                    child = node_map.get(child_id)

                    if not child:
                        continue

                    child_types.append(
                        self._infer_node_type(
                            child,
                            node_map,
                            children
                        )
                    )

                if "string" in child_types:
                    return "string"

                if "double" in child_types:
                    return "double"

                if "int" in child_types:
                    return "int"

                # Python division generally produces float.
                if value == "Div":
                    return "double"

                return "double"

        # --------------------------------------------------------
        # CONDITION
        # --------------------------------------------------------

        if node_type in (
            "COMPARISON",
            "CONDITION",
            "CONDITION_BLOCK"
        ):
            return "boolean"

        # --------------------------------------------------------
        # FUNCTION CALL
        # --------------------------------------------------------

        if node_type == "FUNCTION_CALL":

            if value == "print":
                return "void"

            if value in ("int", "len", "range", "abs", "floor", "ceil", "round", "parseInt"):
                return "int"

            if value in ("float", "double", "sqrt", "pow", "sin", "cos", "parseFloat"):
                return "double"

            if value in ("str", "input", "toString"):
                return "string"

            if value in ("bool", "isinstance", "Boolean"):
                return "boolean"

            if hasattr(self, "function_return_types") and value in self.function_return_types:
                return self.function_return_types[value]

            if "dict" in str(value).lower() or "map" in str(value).lower() or "count" in str(value).lower() or "freq" in str(value).lower():
                return "dict"

        return "unknown"

    # ============================================================
    # FUNCTION RETURN TYPE
    # ============================================================

    def _infer_function_return_type(
        self,
        nodes,
        children,
        node_map
    ):
        """
        Infer the return type of the function.
        """

        return_nodes = [
            node
            for node in nodes
            if node.get("type") == "RETURN"
        ]

        if not return_nodes:
            return "void"

        detected_types = []

        for return_node in return_nodes:

            child_ids = children.get(
                return_node.get("id"),
                []
            )

            if not child_ids:
                continue

            for child_id in child_ids:

                child = node_map.get(child_id)

                if not child:
                    continue

                inferred = self._infer_node_type(
                    child,
                    node_map,
                    children
                )

                if inferred != "unknown":
                    detected_types.append(inferred)

        # --------------------------------------------------------
        # If return is an unknown variable, look at variable type.
        # --------------------------------------------------------

        for return_node in return_nodes:

            for child_id in children.get(
                return_node.get("id"),
                []
            ):

                child = node_map.get(child_id)

                if not child:
                    continue

                if child.get("type") != "VARIABLE":
                    continue

                name = child.get("value")

                variable_type = self.variables.get(
                    name,
                    "unknown"
                )

                if variable_type != "unknown":
                    detected_types.append(
                        variable_type
                    )

                parameter_type = self.parameters.get(
                    name,
                    "unknown"
                )

                if parameter_type != "unknown":
                    detected_types.append(
                        parameter_type
                    )

        # --------------------------------------------------------
        # Resolve priority
        # --------------------------------------------------------

        if "dict" in detected_types:
            return "dict"

        if "string[]" in detected_types:
            return "string[]"

        if "list" in detected_types:
            return "list"

        if "string" in detected_types:
            return "string"

        if "boolean" in detected_types:
            return "boolean"

        if "double" in detected_types:
            return "double"

        if "int" in detected_types:
            return "int"

        if return_nodes:
            return "double"

        return "void"

    # ============================================================
    # LANGUAGE TYPE HELPERS
    # ============================================================

    def _java_type(self, value_type):

        mapping = {
            "int": "int",
            "double": "double",
            "float": "double",
            "string": "String",
            "boolean": "boolean",
            "bool": "boolean",
            "list": "int[]",
            "string[]": "String[]",
            "array": "String[]",
            "dict": "Map<String, Integer>",
            "map": "Map<String, Integer>",
            "void": "void",
            "unknown": "double"
        }

        return mapping.get(
            value_type,
            "double"
        )

    # ============================================================

    def _cpp_type(self, value_type):

        mapping = {
            "int": "int",
            "double": "double",
            "float": "double",
            "string": "string",
            "boolean": "bool",
            "bool": "bool",
            "list": "vector<int>",
            "string[]": "vector<string>",
            "array": "vector<string>",
            "dict": "unordered_map<string, int>",
            "map": "unordered_map<string, int>",
            "void": "void",
            "unknown": "double"
        }

        return mapping.get(
            value_type,
            "double"
        )

    # ============================================================
    # PARAMETERS
    # ============================================================

    def _parameter_declaration(
        self,
        name
    ):

        inferred = self.parameters.get(
            name,
            "unknown"
        )

        if self.target == "java":

            return (
                f"{self._java_type(inferred)} "
                f"{name}"
            )

        if self.target == "cpp":

            return (
                f"{self._cpp_type(inferred)} "
                f"{name}"
            )

        return name

    # ============================================================

    def _function_return_type(self):

        if self.target == "java":

            return self._java_type(
                self.function_return_type
            )

        if self.target == "cpp":

            return self._cpp_type(
                self.function_return_type
            )

        return None

    # ============================================================

    def _function_header(
        self,
        name,
        params,
        return_type
    ):

        if name == "main":
            if self.target == "java":
                return "public static void main(String[] args)"
            if self.target == "cpp":
                return "int main()"
            if self.target == "javascript":
                return "function main()"
            return "def main():"

        if self.target == "java":

            return (
                f"public static {return_type} "
                f"{name}({', '.join(params)})"
            )

        if self.target == "cpp":

            return (
                f"{return_type} "
                f"{name}({', '.join(params)})"
            )

        if self.target == "javascript":

            return (
                f"function {name}"
                f"({', '.join(params)})"
            )

        # Python
        return (
            f"def {name}"
            f"({', '.join(params)}):"
        )

    # ============================================================
    # STATEMENTS
    # ============================================================

    def _translate_statement(
        self,
        node_id,
        node_map,
        children,
        visited,
        inside_loop=False
    ):

        node = node_map.get(node_id)

        if not node:
            return []

        node_type = node.get("type")

        # --------------------------------------------------------
        # ASSIGNMENT
        # --------------------------------------------------------

        if node_type == "ASSIGNMENT":

            visited.add(node_id)

            return self._translate_assignment(
                node,
                node_map,
                children,
                visited,
                inside_loop
            )

        # --------------------------------------------------------
        # RETURN
        # --------------------------------------------------------

        if node_type == "RETURN":

            visited.add(node_id)

            return self._translate_return(
                node,
                node_map,
                children
            )

        # --------------------------------------------------------
        # LOOP
        # --------------------------------------------------------

        if node_type == "LOOP":

            visited.add(node_id)

            return self._translate_loop(
                node,
                node_map,
                children,
                visited
            )

        # --------------------------------------------------------
        # FUNCTION CALL & METHOD CALL
        # --------------------------------------------------------

        if node_type == "FUNCTION_CALL":

            visited.add(node_id)

            return self._translate_function_call(
                node,
                node_map,
                children
            )

        if node_type == "METHOD_CALL":

            visited.add(node_id)

            expr = self._translate_expression(
                node,
                node_map,
                children
            )

            if self.target == "python":
                return [self._indent() + expr]

            return [self._indent() + expr + ";"]

        # --------------------------------------------------------
        # IF
        # --------------------------------------------------------

        if node_type in (
            "IF",
            "CONDITION_BLOCK",
            "CONDITION"
        ):

            visited.add(node_id)

            return self._translate_if(
                node,
                node_map,
                children,
                visited
            )

        # --------------------------------------------------------
        # BREAK & CONTINUE
        # --------------------------------------------------------

        if node_type == "BREAK":

            visited.add(node_id)

            if self.target == "python":
                return [self._indent() + "break"]

            return [self._indent() + "break;"]

        if node_type == "CONTINUE":

            visited.add(node_id)

            if self.target == "python":
                return [self._indent() + "continue"]

            return [self._indent() + "continue;"]

        return []

    # ============================================================
    # ASSIGNMENT
    # ============================================================

    def _translate_assignment(
        self,
        node,
        node_map,
        children,
        visited,
        inside_loop=False
    ):

        child_ids = children.get(
            node.get("id"),
            []
        )

        if len(child_ids) >= 2:
            variable_node = node_map.get(child_ids[0])
            expression_node = node_map.get(child_ids[1])
        elif len(child_ids) == 1:
            variable_node = node_map.get(child_ids[0])
            expression_node = None
        else:
            return []

        if not variable_node:
            return []

        indent = self._indent()

        expression = self._translate_expression(
            expression_node,
            node_map,
            children
        )

        if variable_node.get("type") == "SUBSCRIPT":
            sub_child_ids = children.get(variable_node.get("id"), [])
            if len(sub_child_ids) >= 2:
                receiver = self._translate_expression(node_map.get(sub_child_ids[0]), node_map, children)
                slice_idx = self._translate_expression(node_map.get(sub_child_ids[1]), node_map, children)
                var_type = self.variables.get(receiver, self.parameters.get(receiver, "dict"))
                if self.target == "java":
                    return [indent + f"{receiver}.put({slice_idx}, {expression});"]
                elif self.target == "cpp" or self.target == "javascript":
                    return [indent + f"{receiver}[{slice_idx}] = {expression};"]
                else:
                    return [indent + f"{receiver}[{slice_idx}] = {expression}"]

        name = variable_node.get("value")

        inferred_type = self._infer_node_type(
            expression_node,
            node_map,
            children
        )

        previous_type = self.variables.get(
            name
        )

        if inferred_type == "unknown":
            inferred_type = (
                previous_type
                or "unknown"
            )

        self.variables[name] = inferred_type

        # --------------------------------------------------------
        # PYTHON
        # --------------------------------------------------------

        if self.target == "python":

            return [
                indent +
                f"{name} = {expression}"
            ]

        # --------------------------------------------------------
        # Check if already declared
        # --------------------------------------------------------

        already_declared = (name in self.declared_variables) or (name in self.parameters)
        self.declared_variables.add(name)

        if already_declared:

            return [
                indent +
                f"{name} = {expression};"
            ]

        # --------------------------------------------------------
        # JAVA
        # --------------------------------------------------------

        if self.target == "java":

            java_type = self._java_type(
                self.variables.get(
                    name,
                    inferred_type
                )
            )

            return [
                indent +
                f"{java_type} "
                f"{name} = {expression};"
            ]

        # --------------------------------------------------------
        # C++
        # --------------------------------------------------------

        if self.target == "cpp":

            cpp_type = self._cpp_type(
                self.variables.get(
                    name,
                    inferred_type
                )
            )

            return [
                indent +
                f"{cpp_type} "
                f"{name} = {expression};"
            ]

        # --------------------------------------------------------
        # JAVASCRIPT
        # --------------------------------------------------------

        return [
            indent +
            f"let {name} = {expression};"
        ]

    # ============================================================
    # RETURN
    # ============================================================

    def _translate_return(
        self,
        node,
        node_map,
        children
    ):

        child_ids = children.get(
            node.get("id"),
            []
        )

        indent = self._indent()

        if not child_ids:

            if self.target == "python":

                return [
                    indent +
                    "return"
                ]

            return [
                indent +
                "return;"
            ]

        expression_node = node_map.get(
            child_ids[0]
        )

        expression = self._translate_expression(
            expression_node,
            node_map,
            children
        )

        if self.target == "python":

            return [
                indent +
                f"return {expression}"
            ]

        return [
            indent +
            f"return {expression};"
        ]

    # ============================================================
    # LOOPS
    # ============================================================

    def _translate_loop(
        self,
        node,
        node_map,
        children,
        visited
    ):

        loop_kind = str(
            node.get(
                "value",
                "for"
            )
        ).lower()

        child_ids = children.get(
            node.get("id"),
            []
        )

        if loop_kind == "for":

            return self._translate_for_loop(
                node,
                child_ids,
                node_map,
                children,
                visited
            )

        if loop_kind == "while":

            return self._translate_while_loop(
                node,
                child_ids,
                node_map,
                children,
                visited
            )

        return []

    # ============================================================
    # FOR LOOP
    # ============================================================

    def _translate_for_loop(
        self,
        node,
        child_ids,
        node_map,
        children,
        visited
    ):

        variable = None
        iterable = None
        range_call = None
        body_nodes = []

        for child_id in child_ids:

            child = node_map.get(child_id)

            if not child:
                continue

            child_type = child.get("type")

            if child_type == "VARIABLE":

                if variable is None:
                    variable = child
                elif iterable is None and range_call is None:
                    iterable = child
                else:
                    body_nodes.append(child_id)

            elif child_type == "FUNCTION_CALL" and child.get("value") == "range":
                range_call = child

            elif child_type in (
                "FUNCTION_CALL",
                "ASSIGNMENT",
                "RETURN",
                "LOOP",
                "CONDITION",
                "IF",
                "CONDITION_BLOCK"
            ):

                body_nodes.append(child_id)

        if not variable:
            return []

        variable_name = variable.get(
            "value",
            "item"
        )

        if range_call:
            range_args = []
            for arg_id in children.get(range_call.get("id"), []):
                arg_node = node_map.get(arg_id)
                if arg_node:
                    range_args.append(self._translate_expression(arg_node, node_map, children))

            start_bound = "0"
            end_bound = range_args[0] if range_args else "10"
            if len(range_args) >= 2:
                start_bound = range_args[0]
                end_bound = range_args[1]

            if self.target == "python":
                header = self._indent() + f"for {variable_name} in range({end_bound}):"
            elif self.target == "java":
                header = self._indent() + f"for (int {variable_name} = {start_bound}; {variable_name} < {end_bound}; {variable_name}++) {{"
            elif self.target == "cpp":
                header = self._indent() + f"for (int {variable_name} = {start_bound}; {variable_name} < {end_bound}; {variable_name}++) {{"
            else:
                header = self._indent() + f"for (let {variable_name} = {start_bound}; {variable_name} < {end_bound}; {variable_name}++) {{"

        else:
            iterable_name = (
                iterable.get("value")
                if iterable
                else "items"
            )

            # --------------------------------------------------------
            # Python
            # --------------------------------------------------------

            if self.target == "python":

                lines = [
                    self._indent() +
                    f"for {variable_name} "
                    f"in {iterable_name}:"
                ]

                self.indent_level += 1

                for body_id in body_nodes:

                    generated = self._translate_statement(
                        body_id,
                        node_map,
                        children,
                        visited,
                        inside_loop=True
                    )

                    lines.extend(generated)

                self.indent_level -= 1

                return lines

            # --------------------------------------------------------
            # Java
            # --------------------------------------------------------

            if self.target == "java":

                iterable_type = self.parameters.get(
                    iterable_name,
                    self.variables.get(
                        iterable_name,
                        "unknown"
                    )
                )

                if iterable_type in ("string[]", "array") or "word" in variable_name.lower() or "str" in variable_name.lower():
                    loop_type = "String"
                elif iterable_type == "list":
                    loop_type = "int"
                else:
                    loop_type = "String"

                header = (
                    self._indent() +
                    f"for ({loop_type} "
                    f"{variable_name} : "
                    f"{iterable_name}) {{"
                )

            # --------------------------------------------------------
            # C++
            # --------------------------------------------------------

            elif self.target == "cpp":

                header = (
                    self._indent() +
                    f"for (auto "
                    f"{variable_name} : "
                    f"{iterable_name}) {{"
                )

            # --------------------------------------------------------
            # JavaScript
            # --------------------------------------------------------

            else:

                header = (
                    self._indent() +
                    f"for (const "
                    f"{variable_name} of "
                    f"{iterable_name}) {{"
                )

        lines = [header]

        self.indent_level += 1

        for body_id in body_nodes:

            generated = self._translate_statement(
                body_id,
                node_map,
                children,
                visited,
                inside_loop=True
            )

            lines.extend(generated)

        self.indent_level -= 1

        lines.append(
            self._indent() +
            "}"
        )

        return lines

    # ============================================================
    # WHILE LOOP
    # ============================================================

    def _translate_while_loop(
        self,
        node,
        child_ids,
        node_map,
        children,
        visited
    ):

        condition = None
        body_nodes = []

        for child_id in child_ids:

            child = node_map.get(child_id)

            if not child:
                continue

            child_type = child.get("type")

            if child_type in (
                "COMPARISON",
                "CONDITION",
                "OPERATION"
            ):

                if condition is None:
                    condition = child
                    continue

            if child_type in (
                "FUNCTION_CALL",
                "ASSIGNMENT",
                "RETURN",
                "LOOP",
                "IF",
                "CONDITION",
                "CONDITION_BLOCK"
            ):

                body_nodes.append(child_id)

        condition_code = (
            self._translate_expression(
                condition,
                node_map,
                children
            )
            if condition
            else "true"
        )

        # --------------------------------------------------------
        # Python
        # --------------------------------------------------------

        if self.target == "python":

            lines = [
                self._indent() +
                f"while {condition_code}:"
            ]

            self.indent_level += 1

            for body_id in body_nodes:

                generated = self._translate_statement(
                    body_id,
                    node_map,
                    children,
                    visited,
                    inside_loop=True
                )

                lines.extend(generated)

            self.indent_level -= 1

            return lines

        # --------------------------------------------------------
        # C-style languages
        # --------------------------------------------------------

        lines = [
            self._indent() +
            f"while ({condition_code}) {{"
        ]

        self.indent_level += 1

        for body_id in body_nodes:

            generated = self._translate_statement(
                body_id,
                node_map,
                children,
                visited,
                inside_loop=True
            )

            lines.extend(generated)

        self.indent_level -= 1

        lines.append(
            self._indent() +
            "}"
        )

        return lines

    # ============================================================
    # IF / ELSE
    # ============================================================

    def _translate_if(
        self,
        node,
        node_map,
        children,
        visited
    ):

        child_ids = children.get(
            node.get("id"),
            []
        )

        if not child_ids:
            return []

        condition = node_map.get(child_ids[0])
        body_nodes = []
        else_nodes = []

        for child_id in child_ids[1:]:

            child = node_map.get(child_id)

            if not child:
                continue

            child_type = child.get("type")

            if child_type == "ELSE":

                else_nodes.extend(
                    children.get(
                        child_id,
                        []
                    )
                )

            else:

                body_nodes.append(child_id)

        condition_code = (
            self._translate_expression(
                condition,
                node_map,
                children
            )
            if condition
            else "true"
        )

        # --------------------------------------------------------
        # Python
        # --------------------------------------------------------

        if self.target == "python":

            lines = [
                self._indent() +
                f"if {condition_code}:"
            ]

            self.indent_level += 1

            for body_id in body_nodes:

                generated = self._translate_statement(
                    body_id,
                    node_map,
                    children,
                    visited
                )

                lines.extend(generated)

            self.indent_level -= 1

            if else_nodes:

                lines.append(
                    self._indent() +
                    "else:"
                )

                self.indent_level += 1

                for body_id in else_nodes:

                    generated = self._translate_statement(
                        body_id,
                        node_map,
                        children,
                        visited
                    )

                    lines.extend(generated)

                self.indent_level -= 1

            return lines

        # --------------------------------------------------------
        # Java / C++ / JavaScript
        # --------------------------------------------------------

        lines = [
            self._indent() +
            f"if ({condition_code}) {{"
        ]

        self.indent_level += 1

        for body_id in body_nodes:

            generated = self._translate_statement(
                body_id,
                node_map,
                children,
                visited
            )

            lines.extend(generated)

        self.indent_level -= 1

        lines.append(
            self._indent() +
            "}"
        )

        # --------------------------------------------------------
        # ELSE
        # --------------------------------------------------------

        if else_nodes:

            lines.append(
                self._indent() +
                "else {"
            )

            self.indent_level += 1

            for body_id in else_nodes:

                generated = self._translate_statement(
                    body_id,
                    node_map,
                    children,
                    visited
                )

                lines.extend(generated)

            self.indent_level -= 1

            lines.append(
                self._indent() +
                "}"
            )

        return lines

    # ============================================================
    # FUNCTION CALL
    # ============================================================

    def _translate_function_call(
        self,
        node,
        node_map,
        children
    ):

        name = node.get("value")

        args = []

        for child_id in children.get(
            node.get("id"),
            []
        ):

            child = node_map.get(child_id)

            if child:

                args.append(
                    self._translate_expression(
                        child,
                        node_map,
                        children
                    )
                )

        # --------------------------------------------------------
        # print()
        # --------------------------------------------------------

        if name == "print":

            if self.target == "java":
                if not args:
                    argument = '""'
                elif len(args) == 1:
                    argument = args[0]
                else:
                    argument = ' + " " + '.join(args)

                return [
                    self._indent() +
                    f"System.out.println({argument});"
                ]

            if self.target == "cpp":

                if not args:
                    output = "std::cout"

                else:
                    output = (
                        "std::cout << " +
                        ' << " " << '.join(args)
                    )

                return [
                    self._indent() +
                    output +
                    " << std::endl;"
                ]

            if self.target == "javascript":
                argument = ", ".join(args)

                return [
                    self._indent() +
                    f"console.log("
                    f"{argument});"
                ]

            # Python
            argument = ", ".join(args)
            return [
                self._indent() +
                f"print({argument})"
            ]

        # --------------------------------------------------------
        # Normal function call
        # --------------------------------------------------------

        call = (
            f"{name}"
            f"({', '.join(args)})"
        )

        if self.target == "python":

            return [
                self._indent() +
                call
            ]

        return [
            self._indent() +
            call +
            ";"
        ]

    # ============================================================
    # EXPRESSIONS
    # ============================================================

    def _translate_expression(
        self,
        node,
        node_map,
        children
    ):

        if not node:
            return ""

        node_type = node.get("type")
        value = node.get("value")

        # --------------------------------------------------------
        # CONSTANT
        # --------------------------------------------------------

        if node_type == "CONSTANT":

            value_type = str(
                node.get(
                    "value_type",
                    ""
                )
            ).upper()

            # String
            if value_type in (
                "STRING",
                "STR"
            ):

                text = str(value)

                text = text.strip(
                    "\"'"
                )

                escaped = (
                    text
                    .replace(
                        "\\",
                        "\\\\"
                    )
                    .replace(
                        '"',
                        '\\"'
                    )
                )

                return (
                    '"'
                    +
                    escaped
                    +
                    '"'
                )

            # Boolean
            if value_type in (
                "BOOLEAN",
                "BOOL"
            ):

                is_true = (
                    str(value).lower()
                    == "true"
                )

                if self.target == "python":

                    return (
                        "True"
                        if is_true
                        else "False"
                    )

                return (
                    "true"
                    if is_true
                    else "false"
                )

            return str(value)

        # --------------------------------------------------------
        # VARIABLE
        # --------------------------------------------------------

        if node_type == "VARIABLE":

            return str(value)

        # --------------------------------------------------------
        # METHOD CALL (e.g. text.lower().split() or frequency.get(word, 0))
        # --------------------------------------------------------

        if node_type == "METHOD_CALL":
            child_ids = children.get(node.get("id"), [])
            if not child_ids:
                return f"{value}()"
            receiver_node = node_map.get(child_ids[0])
            receiver_expr = self._translate_expression(receiver_node, node_map, children)
            args = [self._translate_expression(node_map.get(cid), node_map, children) for cid in child_ids[1:] if node_map.get(cid)]

            if value == "lower":
                if self.target == "java" or self.target == "javascript":
                    return f"{receiver_expr}.toLowerCase()"
                if self.target == "cpp":
                    return f"to_lower({receiver_expr})"
                return f"{receiver_expr}.lower()"

            if value == "upper":
                if self.target == "java" or self.target == "javascript":
                    return f"{receiver_expr}.toUpperCase()"
                return f"{receiver_expr}.upper()"

            if value == "split":
                if self.target == "java":
                    return f"{receiver_expr}.split(\"\\\\s+\")"
                if self.target == "javascript":
                    return f"{receiver_expr}.split(/\\s+/)"
                if self.target == "cpp":
                    return f"split_words({receiver_expr})"
                return f"{receiver_expr}.split()"

            if value == "get":
                if len(args) == 2:
                    if self.target == "java":
                        return f"{receiver_expr}.getOrDefault({args[0]}, {args[1]})"
                    if self.target == "javascript":
                        return f"({receiver_expr}[{args[0]}] !== undefined ? {receiver_expr}[{args[0]}] : {args[1]})"
                    if self.target == "cpp":
                        return f"({receiver_expr}.count({args[0]}) ? {receiver_expr}[{args[0]}] : {args[1]})"
                    return f"{receiver_expr}.get({args[0]}, {args[1]})"
                elif len(args) == 1:
                    if self.target == "java":
                        return f"{receiver_expr}.get({args[0]})"
                    if self.target == "javascript" or self.target == "cpp":
                        return f"{receiver_expr}[{args[0]}]"
                    return f"{receiver_expr}.get({args[0]})"

            if value == "append":
                if self.target == "java":
                    return f"{receiver_expr}.add({', '.join(args)})"
                if self.target == "cpp":
                    return f"{receiver_expr}.push_back({', '.join(args)})"
                if self.target == "javascript":
                    return f"{receiver_expr}.push({', '.join(args)})"
                return f"{receiver_expr}.append({', '.join(args)})"

            args_str = ", ".join(args)
            return f"{receiver_expr}.{value}({args_str})"

        # --------------------------------------------------------
        # DICTIONARY (e.g. {})
        # --------------------------------------------------------

        if node_type == "DICTIONARY":
            if self.target == "java":
                return "new HashMap<>()"
            if self.target == "cpp":
                return "{}"
            return "{}"

        # --------------------------------------------------------
        # SUBSCRIPT (e.g. frequency[word])
        # --------------------------------------------------------

        if node_type == "SUBSCRIPT":
            child_ids = children.get(node.get("id"), [])
            if len(child_ids) >= 2:
                receiver_expr = self._translate_expression(node_map.get(child_ids[0]), node_map, children)
                slice_expr = self._translate_expression(node_map.get(child_ids[1]), node_map, children)
                var_type = self.variables.get(receiver_expr, self.parameters.get(receiver_expr, "unknown"))
                if self.target == "java" and "Map" in self._java_type(var_type):
                    return f"{receiver_expr}.get({slice_expr})"
                return f"{receiver_expr}[{slice_expr}]"

        # --------------------------------------------------------
        # LIST
        # --------------------------------------------------------

        if node_type in ("LIST", "UNKNOWN"):

            if value == "list":

                elements = []

                for child_id in children.get(
                    node.get("id"),
                    []
                ):

                    child = node_map.get(child_id)

                    if child:

                        elements.append(
                            self._translate_expression(
                                child,
                                node_map,
                                children
                            )
                        )

                if self.target == "java":

                    return (
                        "new int[]{"
                        +
                        ", ".join(elements)
                        +
                        "}"
                    )

                if self.target == "cpp":

                    return (
                        "{"
                        +
                        ", ".join(elements)
                        +
                        "}"
                    )

                return (
                    "["
                    +
                    ", ".join(elements)
                    +
                    "]"
                )

        # --------------------------------------------------------
        # OPERATION
        # --------------------------------------------------------

        if node_type == "OPERATION":

            operation_map = {

                # Arithmetic
                "Add": "+",
                "Sub": "-",
                "Mult": "*",
                "Div": "/",
                "Mod": "%",

                # Comparisons
                "Gt": ">",
                "GtE": ">=",
                "Lt": "<",
                "LtE": "<=",
                "Eq": "==",
                "NotEq": "!=",

                "Is": "==",
                "IsNot": "!=",
                "In": "in",
                "NotIn": "not in"
            }

            operator = operation_map.get(
                value,
                value
            )

            parts = []

            for child_id in children.get(
                node.get("id"),
                []
            ):

                child = node_map.get(child_id)

                if child:

                    parts.append(
                        self._translate_expression(
                            child,
                            node_map,
                            children
                        )
                    )

            if value == "USub":
                if parts:
                    return f"-{parts[0]}"
                return "-"

            if value == "Not":
                if parts:
                    return f"!({parts[0]})" if self.target != "python" else f"not {parts[0]}"
                return "!"

            if value == "FloorDiv":
                if len(parts) == 2:
                    if self.target == "javascript":
                        return f"Math.floor(({parts[0]}) / {parts[1]})"
                    return f"(({parts[0]}) / {parts[1]})"

            if value in ("In", "NotIn"):
                if len(parts) == 2:
                    item, coll = parts[0], parts[1]
                    if self.target == "java":
                        return f"{coll}.containsKey({item})" if value == "In" else f"!{coll}.containsKey({item})"
                    elif self.target == "cpp":
                        return f"{coll}.count({item})" if value == "In" else f"!{coll}.count({item})"
                    elif self.target == "javascript":
                        return f"({item} in {coll})" if value == "In" else f"!({item} in {coll})"
                    return f"{item} in {coll}" if value == "In" else f"{item} not in {coll}"

            if value == "Pow":
                if len(parts) == 2:
                    if self.target == "java":
                        return f"Math.pow({parts[0]}, {parts[1]})"
                    if self.target == "cpp":
                        return f"pow({parts[0]}, {parts[1]})"
                    if self.target == "javascript":
                        return f"Math.pow({parts[0]}, {parts[1]})"
                    return f"{parts[0]} ** {parts[1]}"

            if len(parts) == 2:

                # ------------------------------------------------
                # JavaScript strict equality
                # ------------------------------------------------

                if self.target == "javascript":

                    if value == "Eq":
                        operator = "==="

                    elif value == "NotEq":
                        operator = "!=="

                return (
                    f"{parts[0]} "
                    f"{operator} "
                    f"{parts[1]}"
                )

            return " ".join(parts)

        # --------------------------------------------------------
        # CONDITION
        # --------------------------------------------------------

        if node_type in (
            "COMPARISON",
            "CONDITION",
            "CONDITION_BLOCK"
        ):

            parts = []

            for child_id in children.get(
                node.get("id"),
                []
            ):

                child = node_map.get(child_id)

                if child:

                    parts.append(
                        self._translate_expression(
                            child,
                            node_map,
                            children
                        )
                    )

            # CONDITION -> OPERATION
            if len(parts) == 1:
                return parts[0]

            if len(parts) >= 2:

                operator = value or ">"

                return (
                    f"{parts[0]} "
                    f"{operator} "
                    f"{parts[1]}"
                )

            return "true"

        # --------------------------------------------------------
        # FUNCTION CALL
        # --------------------------------------------------------

        if node_type == "FUNCTION_CALL":

            args = []

            for child_id in children.get(
                node.get("id"),
                []
            ):

                child = node_map.get(child_id)

                if child:

                    args.append(
                        self._translate_expression(
                            child,
                            node_map,
                            children
                        )
                    )

            if value == "print":

                if self.target == "java":
                    arg_str = ' + " " + '.join(args) if len(args) > 1 else (args[0] if args else '""')
                    return f"System.out.println({arg_str})"

                if self.target == "cpp":

                    if not args:
                        return "std::cout"

                    return (
                        "std::cout << "
                        +
                        ' << " " << '.join(args)
                        +
                        " << std::endl"
                    )

                if self.target == "javascript":

                    return (
                        "console.log("
                        +
                        ", ".join(args)
                        +
                        ")"
                    )

                return (
                    "print("
                    +
                    ", ".join(args)
                    +
                    ")"
                )

            if value == "len":
                if args:
                    arg = args[0]
                    if self.target == "java":
                        var_type = self.variables.get(arg, self.parameters.get(arg, "unknown"))
                        if var_type == "string":
                            return f"{arg}.length()"
                        elif "[]" in self._java_type(var_type) or var_type == "list":
                            return f"{arg}.length"
                        elif "Map" in self._java_type(var_type) or "List" in self._java_type(var_type):
                            return f"{arg}.size()"
                        return f"{arg}.length"
                    elif self.target == "cpp":
                        return f"{arg}.size()"
                    elif self.target == "javascript":
                        return f"{arg}.length"
                    return f"len({arg})"

            if value == "str":
                if args:
                    arg = args[0]
                    if self.target == "java":
                        return f"String.valueOf({arg})"
                    elif self.target == "cpp":
                        return f"to_string({arg})"
                    elif self.target == "javascript":
                        return f"String({arg})"
                    return f"str({arg})"

            if value == "input":
                prompt_arg = ", ".join(args)
                if self.target == "java":
                    return "scanner.nextLine()"
                if self.target == "javascript":
                    return f"prompt({prompt_arg})" if prompt_arg else "prompt()"
                if self.target == "cpp":
                    return "std::cin"
                return f"input({prompt_arg})"

            if value == "int":
                arg_expr = ", ".join(args)
                if self.target == "java":
                    if "scanner" in arg_expr or "input" in arg_expr:
                        return "scanner.nextInt()"
                    if "Math.pow" in arg_expr or "pow" in arg_expr or "." in arg_expr:
                        return f"(int) ({arg_expr})"
                    return f"(int) ({arg_expr})" if arg_expr else "0"
                if self.target == "javascript":
                    if "prompt" in arg_expr or "input" in arg_expr:
                        return f"parseInt({arg_expr})"
                    return f"Math.floor({arg_expr})" if arg_expr else "0"
                if self.target == "cpp":
                    if "cin" in arg_expr or "input" in arg_expr:
                        return "0"
                    return f"static_cast<int>({arg_expr})" if arg_expr else "0"
                return f"int({arg_expr})"

            args_str = ", ".join(args)
            return f"{value}({args_str})"

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        if value is None:
            return ""

        return str(value)

    # ============================================================
    # HELPERS
    # ============================================================

    def _indent(self):

        return (
            "    "
            *
            self.indent_level
        )