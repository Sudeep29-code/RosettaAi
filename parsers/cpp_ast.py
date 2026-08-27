from tree_sitter import Language, Parser
import tree_sitter_cpp

from parsers.common import CommonAST


CPP_LANGUAGE = Language(
    tree_sitter_cpp.language()
)

parser = Parser(
    CPP_LANGUAGE
)


def analyze_cpp_code(code):

    tree = parser.parse(
        code.encode("utf-8")
    )

    root = tree.root_node

    common_ast = CommonAST()

    def visit(node, parent_id=None):

        # Use the node type as the AST node type
        node_type = node.type

        # Get source text
        value = node.text.decode(
            "utf-8",
            errors="ignore"
        ) if node.text else None

        current_id = common_ast.add_node(
            node_type,
            value
        )

        # Connect node to its parent
        if parent_id is not None:

            common_ast.add_edge(
                parent_id,
                current_id
            )

        # Visit children
        for child in node.children:

            visit(
                child,
                current_id
            )

        return current_id

    visit(root)

    return common_ast.get_graph()