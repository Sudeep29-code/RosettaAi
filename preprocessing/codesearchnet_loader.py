from datasets import load_dataset
import os


LANGUAGES = [
    "python",
    "java",
    "javascript"
]

OUTPUT_DIR = "datasets/raw/codesearchnet"


def load_codesearchnet(language):

    print(f"\nLoading CodeSearchNet: {language}")

    dataset = load_dataset(
        "code-search-net/code_search_net",
        language,
        split="train[:100]"
    )

    print(
        f"{language}: {len(dataset)} examples"
    )

    return dataset


def save_dataset(dataset, language):

    output_path = os.path.join(
        OUTPUT_DIR,
        language
    )

    os.makedirs(
        output_path,
        exist_ok=True
    )

    dataset.save_to_disk(
        output_path
    )

    print(
        f"Saved {language} dataset to {output_path}"
    )


if __name__ == "__main__":

    for language in LANGUAGES:

        dataset = load_codesearchnet(
            language
        )

        save_dataset(
            dataset,
            language
        )