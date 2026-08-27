from datasets import load_from_disk


LANGUAGES = [
    "python",
    "java",
    "javascript"
]


for language in LANGUAGES:

    path = (
        f"datasets/raw/codesearchnet/"
        f"{language}"
    )

    dataset = load_from_disk(path)

    print("\n==============================")
    print("LANGUAGE:", language)
    print("==============================")

    print("Columns:")
    print(dataset.column_names)

    print("\nFirst example:")

    example = dataset[0]

    for key, value in example.items():

        print(
            f"\n{key}:"
        )

        print(
            str(value)[:500]
        )