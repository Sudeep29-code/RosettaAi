import os
import json

from datasets import load_from_disk

from preprocessing.cleaner import clean_code, is_valid_example
from parsers.parser_factory import parse_code
from semantic.normalizer import ASTNormalizer


INPUT_PATH = "datasets/raw/multipl_e/cpp"
OUTPUT_PATH = "datasets/processed/ast/cpp"


def main():

    print("Loading C++ MultiPL-E dataset...")

    dataset = load_from_disk(INPUT_PATH)

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    normalizer = ASTNormalizer()

    processed = []

    print("Total C++ examples:", len(dataset))

    for index, example in enumerate(dataset):

        code = example.get("prompt", "")

        if not code:
            continue

        code = clean_code(code)

        if not code:
            continue

        try:

            graph = parse_code(
                code,
                "cpp"
            )

            graph = normalizer.normalize(
                graph,
                "cpp"
            )

            processed.append({
                "id": index,
                "language": "cpp",
                "function_name": example.get(
                    "name",
                    ""
                ),
                "graph": graph
            })

        except Exception as error:

            print(
                f"FAILED: C++ example {index}: "
                f"{repr(error)}"
            )

            continue

    output_file = os.path.join(
        OUTPUT_PATH,
        "graphs.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            processed,
            file,
            indent=2
        )

    print()
    print("C++ processing completed.")
    print("Processed examples:", len(processed))
    print("Saved to:", output_file)


if __name__ == "__main__":
    main()