from translation.rosetta_engine import RosettaEngine


code = """def calculate(numbers):
    total = 0

    for x in numbers:
        total = total + x

    if total > 10:
        print(total)
    else:
        print(0)

    return total
"""


print("=" * 60)
print("INITIALIZING ENGINE")
print("=" * 60)

engine = RosettaEngine()


for target in ["java", "cpp", "javascript"]:

    print("\n" + "=" * 60)
    print(f"PYTHON -> {target.upper()}")
    print("=" * 60)

    try:

        result = engine.translate(
            code,
            "python",
            target
        )

        print("\n--- TRANSLATED CODE ---")
        print(result["translated_code"])

        print("\n--- AST TRANSLATION ---")
        print(result["ast_translation"])

        print("\n--- RULE TRANSLATION ---")
        print(result["rule_translation"])

        print("\n--- EMBEDDING SHAPE ---")
        print(result["embedding_shape"])

        print("\n--- PREDICTED LANGUAGE ---")
        print(result["predicted_language"])

    except Exception as e:

        print("\nERROR:")
        print(type(e).__name__, e)