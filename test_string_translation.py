from parsers.parser_factory import parse_code
from translation.ast_translator import ASTTranslator


code = '''def greet():
    return "Hello"
'''

graph = parse_code(
    code,
    "python"
)

print("--- GRAPH ---")
print(graph)

translator = ASTTranslator()

result = translator.translate_graph(
    graph,
    "python",
    "java"
)

print("\n--- TRANSLATION ---")
print(result)