from translation.rosetta_engine import RosettaEngine


engine = RosettaEngine()


code = """def add(a, b):
    result = a + b
    return result
"""


languages = [
    "java",
    "javascript",
    "cpp",
    "python"
]


print("=" * 60)
print("SOURCE CODE")
print("=" * 60)
print(code)


for target in languages:

    print("\n" + "=" * 60)
    print(f"PYTHON -> {target.upper()}")
    print("=" * 60)

    try:

        result = engine.translate(
            code,
            "python",
            target
        )

        print(result["ast_translation"])

    except Exception as e:

        print("ERROR:", e)