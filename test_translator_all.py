from parsers.parser_factory import parse_code
from translation.ast_translator import ASTTranslator


code = '''
def calculate(numbers):
    total = 0

    for x in numbers:
        total = total + x

    if total > 10:
        print(total)
    else:
        print(0)

    return total
'''


graph = parse_code(code, "python")

print("========== GRAPH ==========")
print(graph)

translator = ASTTranslator()

for target in ["java", "cpp", "javascript"]:

    print()
    print("=" * 60)
    print(f"========== {target.upper()} ==========")
    print("=" * 60)

    try:
        result = translator.translate_graph(
            graph,
            "python",
            target
        )

        print(result)

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")