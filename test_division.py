from translation.rosetta_engine import RosettaEngine

e = RosettaEngine()

code = """def divide(a, b):
    result = a / b
    return result
"""

r = e.translate(
    code,
    "python",
    "java"
)

print(r["ast_translation"])
