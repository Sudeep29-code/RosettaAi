from datasets import load_from_disk

from parsers.parser_factory import parse_code


DATASETS = {
    "python": "datasets/raw/codesearchnet/python",
    "java": "datasets/raw/codesearchnet/java",
    "javascript": "datasets/raw/codesearchnet/javascript",
    "cpp": "datasets/raw/codesearchnet/cpp",
}


def test_language(language, path):

    print()
    print("=" * 60)
    print("TESTING:", language.upper())
    print("=" * 60)

    dataset = load_from_disk(path)

    code = dataset[0]["func_code_string"]

    print("\nCODE:")
    print("-" * 60)
    print(code[:500])

    try:

        graph = parse_code(
            code,
            language
        )

        print("\nPARSING SUCCESSFUL")

        print("\nGRAPH:")
        print(graph)

    except Exception as e:

        print("\nPARSING FAILED")

        print(
            type(e).__name__,
            ":",
            str(e)
        )


def main():

    for language, path in DATASETS.items():

        test_language(
            language,
            path
        )


if __name__ == "__main__":
    main()