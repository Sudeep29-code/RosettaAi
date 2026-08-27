import os
from datasets import load_dataset


OUTPUT_PATH = "datasets/raw/multipl_e/cpp"


def main():

    print("Downloading MultiPL-E C++ dataset...")

    dataset = load_dataset(
        "nuprl/MultiPL-E",
        "humaneval-cpp",
        split="test"
    )

    os.makedirs(
        OUTPUT_PATH,
        exist_ok=True
    )

    dataset.save_to_disk(
        OUTPUT_PATH
    )

    print()
    print("C++ dataset downloaded.")
    print("Examples:", len(dataset))
    print("Saved to:", OUTPUT_PATH)
    print("Columns:", dataset.column_names)


if __name__ == "__main__":
    main()