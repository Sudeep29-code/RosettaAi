import json
import os

from datasets import load_from_disk

from parsers.parser_factory import parse_code
from semantic.normalizer import ASTNormalizer


DATASETS = {
    "python": "datasets/raw/codesearchnet/python",
    "java": "datasets/raw/codesearchnet/java",
    "javascript": "datasets/raw/codesearchnet/javascript",
    "cpp": "datasets/raw/codesearchnet/cpp",
}


OUTPUT_BASE = "datasets/processed/ast"


def process_language(language, dataset_path):

    print()
    print("=" * 60)
    print(f"Processing {language.upper()}")
    print("=" * 60)

    dataset = load_from_disk(dataset_path)

    print(f"Total samples found: {len(dataset)}")

    graphs = []

    success_count = 0
    failed_count = 0

    normalizer = ASTNormalizer()

    for index, item in enumerate(dataset):

        code = item.get("func_code_string")

        if not code:
            failed_count += 1
            continue

        try:

            graph = parse_code(
                code,
                language
            )

            graph = normalizer.normalize(
                graph
            )

            graphs.append({
                "language": language,
                "graph": graph
            })

            success_count += 1

        except Exception as error:

            failed_count += 1

            print(
                f"Skipping sample {index}: "
                f"{str(error)[:100]}"
            )

    output_directory = os.path.join(
        OUTPUT_BASE,
        language
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_path = os.path.join(
        output_directory,
        "graphs.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            graphs,
            file,
            indent=2
        )

    print()
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Saved to: {output_path}")


def main():

    print()
    print("=" * 60)
    print("CODESEARCHNET PREPROCESSING")
    print("=" * 60)

    for language, dataset_path in DATASETS.items():

        process_language(
            language,
            dataset_path
        )

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()