import os
import json

from datasets import load_from_disk

from preprocessing.cleaner import (
    clean_code,
    is_valid_example
)

from parsers.parser_factory import parse_code

from semantic.normalizer import ASTNormalizer


LANGUAGES = [
    "python",
    "java",
    "javascript",
    "cpp"
]

BASE_INPUT_PATH = "datasets/raw/codesearchnet"
BASE_OUTPUT_PATH = "datasets/processed/ast"


def process_language(language):

    input_path = os.path.join(
        BASE_INPUT_PATH,
        language
    )

    output_path = os.path.join(
        BASE_OUTPUT_PATH,
        language
    )

    dataset = load_from_disk(
        input_path
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    normalizer = ASTNormalizer()

    processed = []

    print(f"\nProcessing {language}...")

    for index, example in enumerate(dataset):

        if not is_valid_example(example):
            continue

        code = clean_code(
            example["func_code_string"]
        )

        try:

            graph = parse_code(
                code,
                language
            )

            graph = normalizer.normalize(
    graph,
    language
)

            processed.append({
                "id": index,
                "language": language,
                "function_name": example.get(
                    "func_name",
                    ""
                ),
                "graph": graph
            })

        except Exception as error:

            print(
                f"FAILED: {language} "
                f"example {index}: {repr(error)}"
            )

            continue

    output_file = os.path.join(
        output_path,
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

    print(
        f"{language} completed."
    )

    print(
        "Processed examples:",
        len(processed)
    )

    print(
        "Saved to:",
        output_file
    )


def main():

    for language in LANGUAGES:

        process_language(
            language
        )

    print(
        "\nAll languages processed."
    )


if __name__ == "__main__":

    main()