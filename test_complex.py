from translation.rosetta_engine import RosettaEngine


engine = RosettaEngine()


tests = {
    "If Else": """
def check_age(age):
    if age >= 18:
        return True
    else:
        return False
""",

    "Multiple Operations": """
def calculate(a, b):
    x = a + b
    y = x * 2
    return y
""",

    "Multiple Returns": """
def maximum(a, b):
    if a > b:
        return a
    return b
""",

    "Print": """
def greet(name):
    print(name)
    return True
"""
}


languages = [
    "JAVA",
    "JAVASCRIPT",
    "CPP",
    "PYTHON"
]


for test_name, source_code in tests.items():

    print("=" * 60)
    print(test_name)
    print("=" * 60)

    print("\nSOURCE CODE")
    print("-" * 60)
    print(source_code.strip())

    for target in languages:

        print("\n--- PYTHON -> " + target + " ---")

        try:

            result = engine.translate(
                source_code,
                "python",
                target.lower()
            )

            print(result["translated_code"])

        except Exception as e:

            print(
                "ERROR:",
                type(e).__name__,
                e
            )