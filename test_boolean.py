from translation.rosetta_engine import RosettaEngine

e = RosettaEngine()

code = """def is_positive(x):
    return x > 0
"""

r = e.translate(
    code,
    "python",
    "java"
)

print(r["ast_translation"])