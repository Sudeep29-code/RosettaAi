from translation.rosetta_engine import RosettaEngine

engine = RosettaEngine()

code = """def check_age(age):
    if age >= 18:
        return True
    else:
        return False
"""

result = engine.translate(
    code,
    "python",
    "java"
)

print(result["ast_translation"])