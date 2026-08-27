from translation.rosetta_engine import RosettaEngine
from evaluation.evaluate import calculate_codebleu
from fastapi.testclient import TestClient
from app import app
import json


def run_demonstration():
    print("=" * 70)
    print(" ROSETTA AI: LIVE DEMO & INPUT/OUTPUT WALKTHROUGH")
    print("=" * 70)

    # ----------------------------------------------------
    # 1. CORE ENGINE DEMO (Python SDK)
    # ----------------------------------------------------
    engine = RosettaEngine()

    sample_input_python = """def calculate_discount(prices, min_spend):
    total = 0
    for p in prices:
        if p >= min_spend:
            total = total + p
    return total"""

    print("\n" + "=" * 70)
    print(" 1. INPUT CODE (Python):")
    print("=" * 70)
    print(sample_input_python)

    targets = ["java", "cpp", "javascript"]

    for target in targets:
        print("\n" + "-" * 70)
        print(f" OUTPUT TRANSLATION -> {target.upper()}")
        print("-" * 70)

        result = engine.translate(
            code=sample_input_python,
            source_language="python",
            target_language=target,
            strategy="hybrid",
            refactor=True
        )

        print(result["translated_code"])
        print("\n[Metadata]")
        print(f"- Strategy used: {result['strategy_used']}")
        print(f"- Syntax valid: {result['syntax_valid']}")
        print(f"- Embedding shape: {result['embedding_shape']}")

    # ----------------------------------------------------
    # 2. FASTAPI REST API DEMO
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(" 2. FASTAPI REST API CALL DEMO (POST /translate)")
    print("=" * 70)

    client = TestClient(app)

    api_payload = {
        "code": "def is_even(num):\n    if num % 2 == 0:\n        return True\n    return False",
        "source_language": "python",
        "target_language": "java",
        "strategy": "hybrid",
        "refactor": True
    }

    print("\n[HTTP REQUEST PAYLOAD]:")
    print(json.dumps(api_payload, indent=2))

    response = client.post("/translate", json=api_payload)
    print("\n[HTTP RESPONSE (Status 200 OK)]:")
    print(json.dumps(response.json(), indent=2))

    # ----------------------------------------------------
    # 3. EVALUATION METRICS DEMO (POST /evaluate)
    # ----------------------------------------------------
    print("\n" + "=" * 70)
    print(" 3. CODE EVALUATION DEMO (POST /evaluate)")
    print("=" * 70)

    eval_payload = {
        "reference_code": "public static boolean is_even(int num) {\n    if (num % 2 == 0) {\n        return true;\n    }\n    return false;\n}",
        "candidate_code": response.json()["translated_code"],
        "language": "java"
    }

    eval_response = client.post("/evaluate", json=eval_payload)
    print("\n[EVALUATION METRICS RESULT]:")
    print(json.dumps(eval_response.json(), indent=2))


if __name__ == "__main__":
    run_demonstration()

