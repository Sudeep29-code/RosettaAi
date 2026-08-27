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

    "assignment_expression": "OPERATION",

}


def normalize_cpp_type(node_type):

    return CPP_NODE_MAP.get(
        node_type,
        "OTHER"
    )