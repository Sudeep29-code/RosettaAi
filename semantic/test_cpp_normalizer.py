from parsers.cpp_ast import analyze_cpp_code
from semantic.normalizer import ASTNormalizer


code = """
int factorial(int n) {

    if (n == 0) {
        return 1;
    }

    return n * factorial(n - 1);
}
"""


graph = analyze_cpp_code(code)

normalizer = ASTNormalizer()

normalized = normalizer.normalize(
    graph,
    "cpp"
)


for node in normalized["nodes"][:20]:

    print(node)