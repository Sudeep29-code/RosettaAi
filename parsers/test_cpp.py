from parsers.cpp_ast import analyze_cpp_code


code = """
int factorial(int n) {

    if (n == 0) {
        return 1;
    }

    return n * factorial(n - 1);
}
"""


graph = analyze_cpp_code(code)


print(
    "Number of nodes:",
    len(graph["nodes"])
)

print(
    "Number of edges:",
    len(graph["edges"])
)

print(
    "\nFirst 10 nodes:"
)

for node in graph["nodes"][:10]:

    print(node)